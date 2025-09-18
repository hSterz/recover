import os
import datasets
import torch
from tqdm import tqdm
import pandas as pd

from language_utils import lang_map, lang_string
from few_shot_samples import few_shot_sample

class Evaluator:
    def __init__(self, model, tokenizer, dataset_name, crosslingual=True, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_name = dataset_name
        self.model_name = self.model.model.config.name_or_path

        self.prompt_key = kwargs.get("prompt_key", "prompt")
        self.lang_key = kwargs.get("lang_key", "language")
        self.source_lang_key = kwargs.get("source_lang_key", "source_language")
        self.version = kwargs.get("version", "v?")
        self.crosslingual = crosslingual
        self.output_file = kwargs.get("output_file", "{}/{}/{}/predictions.txt".format(self.dataset_name, self.model_name.split("/")[-1], self.version))
        self.include_lang = kwargs.get("include_lang", True)
        self.shots = kwargs.get("shots", 0)
        self.use_steer = kwargs.get("use_steer", False)
        # Gemma does not support system prompts
        self.use_system_prompt = kwargs.get("use_system_prompt", True) and "gemma" not in self.model_name
        self.system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")

        self.save_steps = kwargs.get("save_steps", 100)
        self.predictions = []

    def load_dataset(self):
        self.dataset = datasets.load_dataset(self.dataset_name)

    def _load_predictions(self):
        if os.path.exists(self.output_file):
            df = pd.read_csv(self.output_file)
            for _, row in df.iterrows():
                self.predictions.append({
                    "prompt": row["prompt"],
                    "language": row["language"],
                    "source_language": row["source_language"],
                    "completion": row["completion"],
                    "model": row["model"],
                    "task": row["task"],
                    "source":row["source"],
                })
        print("Loaded {} predictions from {}".format(len(self.predictions), self.output_file) if self.predictions else "No previous predictions found.")

    def evaluate(self, max_new_tokens):
        self.model.eval()

        self._load_predictions()

        for i, sample in tqdm(enumerate(self.dataset.select(range(len(self.predictions), len(self.dataset)))), total=len(self.dataset) - len(self.predictions), desc="Evaluating"):
            user_input = sample[self.prompt_key]
            lang = sample[self.lang_key]
            anchor = sample.get(self.source_lang_key, "en")
            completion = self.predict(user_input, lang, anchor, max_new_tokens)
            self.predictions.append({
                "prompt": user_input,
                "language": lang,
                "source": sample.get("source", self.dataset_name),
                "source_language": anchor,
                "completion": completion,
                "model": self.model_name,
                "task": "crosslingual" if self.crosslingual else "monolingual",
                "source": sample["source"] if "source" in sample else self.dataset_name
            })
            if i % self.save_steps == 0:
                self.save_predictions()

    def save_predictions(self):
        # Save the predictions to a file
        if not os.path.exists(os.path.dirname(self.output_file)):
            os.makedirs(os.path.dirname(self.output_file))
        # save the predictions using pandas
        df = pd.DataFrame(self.predictions)
        df.to_csv(self.output_file, index=False, header=True)

    def predict(self, text, lang, anchor, max_new_tokens):
        if self.include_lang == False:
            
            text = text.replace(lang_string[lang], "-")
        if self.use_steer:
            if self.crosslingual:
                self.model.set_lang(lang_map[lang], anchor=lang_map[anchor])
            else:
                if self.model.adaptive_alpha is not None:
                    self.model.set_lang(lang_map[lang], anchor=lang_map[lang])
                else:
                    self.model.set_lang(lang_map[lang], anchor=None)
        
        if self.shots > 0:
            samples = few_shot_sample[lang][:self.shots]
            few_shot_prompt = "\n".join([f"Q: {sample['question']}\nA: {sample['answer']}" for sample in samples])
            text = few_shot_prompt + "\nQ:" + text + "\nA:"
        
        if  self.use_system_prompt:
            base_prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": text}], 
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            base_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], 
                tokenize=False,
                add_generation_prompt=True
            )
        input_data = self.tokenizer(base_prompt, return_tensors="pt").to("cuda")
        
        prompt_length = input_data["input_ids"].shape[1]
        
        with torch.no_grad():
            output = self.model.model.generate(
                **input_data,
                max_new_tokens=max_new_tokens,
                use_cache=False if "gemma" in self.model_name else True,
            )
        output_text = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
           
        return output_text
