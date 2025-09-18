from functools import partial
import math
import json
import os
import numpy as np
import torch
from torch import nn


class LowRankAD(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers = -1, crossling=False, dtype=torch.bfloat16, **kwargs):
        super().__init__()
        self.num_in = 3 if crossling else 2
        self.crossling = crossling
        self.layer_emb = None
        rank=kwargs.get('rank', 128)
        print("Initialize with rank: ", rank)
        if num_layers > 0:
            self.layer_emb = nn.Embedding(num_layers, hidden_dim, dtype=dtype)
            self.num_in += 1
        self.A = nn.Parameter(torch.randn(input_dim * self.num_in, rank))
        
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, hidden_dim))
    
        self.dropout = nn.Dropout(0.2)


    def forward(self, h, v, layer_idx, v_s = None): 
        if v_s is not None and self.crossling:
            x = torch.concat((h, v, v_s), dim=2).to(h.dtype)
        else:
            x = torch.concat((h, v), dim=2).to(h.dtype)
        if self.layer_emb is not None:
            l_emb = self.layer_emb(layer_idx)
            x = torch.concat((x, l_emb), dim=2).to(h.dtype)
        
        dtype = x.dtype
        x = x.float() @ self.A @self.B
        x1 = self.dropout(x)
        x1 = x1.to(dtype)
        return x1


intervention_map = {
    "low_rank_ad": LowRankAD,
    "low_rank_2": LowRankAD, # for backward compatibility
}

class Steer(nn.Module):
    _keys_to_ignore_on_save = ['interventions']
    def __init__(self, model, path, lang="id", arithmetic="naive", skip_layers=[], **kwargs):
        super().__init__()
        self.model = model
        self.hooks = []
        self.vectors = {}
        self.lang = lang
        self.arithmetic = arithmetic
        self.skip_layers = skip_layers
        self.anchor = kwargs.get("anchor", None)
        self.alpha = kwargs.get("alpha", 1)
        self.beta = kwargs.get("beta", 1)
        self.apply_naive_before = kwargs.get("apply_naive_before", True)
        self.parallel = kwargs.get("parallel", False)

        self.train_mode = None

        self.scaling_mode  = kwargs.get("scaling_mode", None)
        self.restore_norm = kwargs.get("restore_norm", False)
        self.layer_wise_AD = kwargs.get("layer_wise_AD", False)
        self.crossling = kwargs.get("crossling", True)

        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

        if arithmetic == "intervene":
            ad_class = intervention_map[kwargs.get("intervention", "low_rank_ad")]
            self.intervention_type = kwargs.get("intervention", "low_rank_ad")
            self.rank = kwargs.get("rank", 128)
            
            if self.layer_wise_AD:
                self.adaptive_alpha = nn.ModuleList([ ad_class(self.hidden_size, self.hidden_size, num_layers=-1, crossling=self.crossling is not None, rank=self.rank) for _ in range(self.num_layers)])
            else:
                self.adaptive_alpha = ad_class(self.hidden_size, self.hidden_size, num_layers=self.num_layers, crossling=self.crossling is not None, rank=self.rank)

        self.remove_content = kwargs.get("remove_content", False)
        self._init_vector(path, remove_content=self.remove_content)
        

    def set_lang(self, lang, anchor=None):
        self.clear()
        if type(lang) == torch.Tensor:
            vectors = []

            for l in lang.tolist():
                if isinstance(l, list):
                    l = l[0]
                vectors.append(self.vectors[self.id2lang[l]])
            vector = np.stack(vectors, axis=1)
        else:
            vector = self.vectors[lang]
        source_vector = None
        if anchor is not None:
            if type(anchor) == torch.Tensor:
                source_vector = []

                for l in anchor.tolist():
                    if isinstance(l, list):
                        l = l[0]
                    source_vector.append(self.vectors[self.id2lang[l]])
                source_vector = np.stack(source_vector, axis=1)
            else:
                source_vector = self.vectors[anchor]
        self.add_steering(vector, lang, source_vector, skip_layers=self.skip_layers)
        self.lang = lang
        self.anchor = anchor


    def _init_vector(self, path, remove_content):
        for file_name in os.listdir(path):
            if file_name.endswith(".npy"):
                vector = np.load(os.path.join(path, file_name))
                lang_name = file_name[:-4]
                self.vectors[lang_name] = vector
            elif file_name.endswith(".pt"):
                vector = torch.load(os.path.join(path, file_name), weights_only=True).numpy()
                lang_name = file_name.split(".")[0]
                self.vectors[lang_name] = vector
        
        if remove_content:
            avg = np.stack([v[:, :1] for v in self.vectors.values()]).mean(axis=0)
            for k, v in self.vectors.items():
                self.vectors[k] = v - avg
                    
                
      
    def _get_vector_subsequence(self, v, shape, adjust_len=True):
        s = shape[1]
        
        tmp = v
        if adjust_len:
            tmp = tmp.repeat(1, s, 1)
        return tmp

    def naive_steer(self, module, args, module_output, vector, source_vector, lang, average=False, layer_idx=None):
        hidden_state = module_output[0]
       

        if not hasattr(self, "position_idx") or (layer_idx == 0 and hidden_state.shape[1] > 1):
            self.position_idx = 0
        if  len(vector.shape) == 2:
            vector = vector.unsqueeze(0)
        b, s_len, d = vector.shape
        s = hidden_state.shape[1]

       
        v = self._get_vector_subsequence(vector, hidden_state.shape)
        if source_vector is not None:
            if len(source_vector.shape) == 2:
                source_vector = source_vector.unsqueeze(0)
            v_source = self._get_vector_subsequence(source_vector, hidden_state.shape)
            v = v - self.beta * v_source
            
        alpha = self.alpha 
        if self.scaling_mode == "relative_norm":
            l = torch.linalg.norm(hidden_state, dim=-1, keepdim=True) 
            v = v / torch.linalg.norm(v, dim=-1, keepdim=True) * alpha * l
        elif self.scaling_mode== "norm":
            if isinstance(alpha, torch.Tensor):
                v = (v / torch.linalg.norm(v, dim=-1, keepdim=True)) * torch.unsqueeze(alpha, dim=2)
            else:
                v =(v / torch.linalg.norm(v, dim=-1, keepdim=True)) * alpha
    
        elif self.scaling_mode == "factor":
            v = v * alpha
        elif self.scaling_mode is not None:
            raise ValueError("Unknown scaling mode: {}".format(self.scaling_mode))
    
        out = (hidden_state + v).to(hidden_state.dtype)
        if self.position_idx == 0:
            out[:, :1] = hidden_state[:, :1]
        
        if layer_idx == self.num_layers - 1:
            if self.position_idx == 0:
                self.prompt_length = s
            self.position_idx += s

        if self.restore_norm:
            out = out / torch.linalg.norm(out, dim=-1, keepdim=True) * torch.linalg.norm(hidden_state, dim=-1, keepdim=True)
        
        if isinstance(module_output, tuple):
            out = (out,)
        return out


    def intervene(self, module, args, module_return, vector, source_vector, lang, layer_idx):
        hidden_state = module_return[0]
        
        input_state = hidden_state
        
        if  len(vector.shape) == 2:
            vector = vector.unsqueeze(0)
        b, s_len, d = vector.shape
        s = hidden_state.shape[1]

        adjust_len = True
        
        target_v = self._get_vector_subsequence(vector, hidden_state.shape, adjust_len=adjust_len)
        if source_vector is not None:
            if len(source_vector.shape) == 2:
                source_vector = source_vector.unsqueeze(0)
            v_source = self._get_vector_subsequence(source_vector, hidden_state.shape, adjust_len=adjust_len)
           
        else:
            v_source = None
        layer = torch.tensor(layer_idx, device=hidden_state.device).repeat(hidden_state.shape[0], hidden_state.shape[1])
        
        if self.apply_naive_before:
            v = target_v
            v = v - self.beta * v_source if v_source is not None else v
            if self.scaling_mode == "relative_norm":
                l = torch.linalg.norm(hidden_state, dim=-1, keepdim=True) 
                v = v / torch.linalg.norm(v, dim=-1, keepdim=True) * self.alpha * l
            elif self.scaling_mode== "norm":
                if isinstance(self.alpha, torch.Tensor):
                    v = (v / torch.linalg.norm(v, dim=-1, keepdim=True)) * torch.unsqueeze(self.alpha, dim=2)
                else:
                    v =(v / torch.linalg.norm(v, dim=-1, keepdim=True)) * self.alpha
        
            elif self.scaling_mode == "factor":
                v = v * self.alpha
                
            hidden_state2 = (hidden_state + v).to(hidden_state.dtype)
            if self.parallel:
                ad_input = hidden_state
            else:
                ad_input = hidden_state2
        if self.layer_wise_AD:
            vec = self.adaptive_alpha[layer_idx](ad_input, target_v, None, v_s=v_source )
        else:
            vec = self.adaptive_alpha(hidden_state, target_v, layer, v_s=v_source )
        
        
        out =  (vec +  hidden_state2).to(hidden_state.dtype)
        if self.restore_norm:
            out = out / torch.linalg.norm(out, dim=-1, keepdim=True) * torch.linalg.norm(hidden_state, dim=-1, keepdim=True)

        return (out,)
        




    def add_steering(self, vector, lang, source_vector=None, skip_layers=[]):
        layers =  self.model.model.layers
        offset = 1

        
        for layer_idx, layer in enumerate(layers): 
            module = layer
            if skip_layers is not None and layer_idx + offset in skip_layers:
                continue
            v = torch.tensor(vector[layer_idx + offset], device="cuda")
            v_s = None
            if source_vector is not None:
                v_s = torch.tensor(source_vector[layer_idx + offset], device="cuda")
            
            if self.arithmetic == "intervene" or self.arithmetic == "alpha":
                handle = module.register_forward_hook(partial(self.intervene, vector=v, source_vector=v_s, lang=lang, layer_idx=layer_idx))
                self.hooks.append(handle)
            elif self.arithmetic == "naive":
                handle = module.register_forward_hook(partial(self.naive_steer, vector=v, source_vector=v_s, lang=lang, layer_idx=layer_idx))
                self.hooks.append(handle)
            elif self.arithmetic == "identity":
                pass
            else:
                raise NotImplementedError("Only naive steering is supported at the moment")
            
    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def forward(self, *args, **kwargs):
        lang = kwargs.pop("lang", self.lang)
        source_lang = kwargs.pop("source_lang", self.anchor)
       
        if isinstance(lang, torch.Tensor) or (lang is not None and lang != self.lang) :
            self.set_lang(lang, source_lang)
        
        output = self.model(*args, **kwargs)

        return output
    

    def train_steer(self):
        for k, v in self.named_parameters():
            if "adaptive_alpha" in k:
                v.requires_grad = True
            else:
                v.requires_grad = False
        self.train_mode = "steer"

    def _get_config(self):
        config = {
            "arithmetic": self.arithmetic,
            "alpha": self.alpha,
            "beta": self.beta,
            "layer_wise_AD": self.layer_wise_AD,
            "skip_layers": self.skip_layers,
            "intervention_type": self.intervention_type,
            "remove_content": self.remove_content,
            "cross_lingual": self.crossling,
            "rank": self.rank,
            "apply_naive_before": self.apply_naive_before,
            "parallel": self.parallel,
        }
        return config

    def save_intervention(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            raise ValueError(f"Directory {path} already exists")
        config = self._get_config()
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
        state_dict = self.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if "adaptive_alpha" in k}
        torch.save(state_dict, os.path.join(path, "model.pth"))

    def load_intervention(self, path):
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        self.arithmetic = config["arithmetic"]
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.layer_wise_AD = config["layer_wise_AD"]
        self.intervention_type = config["intervention_type"]
        self.skip_layers = config["skip_layers"]
        self.average_content = config["average_content"]
        self.remove_content = config["remove_content"]
        self.apply_naive_before = config["apply_naive_before"] if "apply_naive_before" in config else True
        self.parallel = config["parallel"] if "parallel" in config else True
        if "num_buckets" in config:
            self.num_buckets = config["num_buckets"] 
        if "rank" in config:
            self.rank = config["rank"]
        else:
            self.rank = 128
        print("Loading intervention with config: ", config)
        self.crossling = config["cross_lingual"] if "cross_lingual" in config else False
        ad_class = intervention_map[self.intervention_type]

        if self.layer_wise_AD:
            self.adaptive_alpha = nn.ModuleList([ ad_class(self.hidden_size, self.hidden_size, num_layers=-1, crossling=self.crossling, rank=self.rank) for _ in range(self.num_layers)])
        else:
            self.adaptive_alpha = ad_class(self.hidden_size, self.hidden_size, num_layers=self.num_layers, crossling=self.crossling, rank=self.rank)
        if self.intervention_type == "hybrid_low_rank_ad":
            # initialize B
            self.shared_B = nn.Parameter(torch.randn(self.rank, self.hidden_size))
            self.shared_B = nn.init.xavier_uniform_(self.shared_B)
            for i in range(self.num_layers):
                self.adaptive_alpha[i].B = self.shared_B
        
        state_dict = torch.load(os.path.join(path, "model.pth"))
        
        self.load_state_dict(state_dict, strict=False)