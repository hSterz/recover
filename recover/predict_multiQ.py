import os 
import argparse
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from steering import Steer
import pandas as pd


from prediction_utils import Evaluator

class LCBEvaluator(Evaluator):
    def __init__(self, model, tokenizer, dataset_name, crosslingual=True, **kwargs):
        super().__init__(model, tokenizer, dataset_name, crosslingual=crosslingual, **kwargs)
    
    def load_dataset(self, path = None, crossling=True, subset=None):
        if crossling:
            ds = datasets.load_from_disk("data/MultiQ_crosslingual")
        else:
            df = pd.read_csv("data/MultiQ.csv")
            ds = datasets.Dataset.from_pandas(df)
        self.dataset = ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_only", action="store_true", help="Use only base model only without applying steering.")
    parser.add_argument("--crosslingual", action="store_true", help="Evaluate on crosslingual dataset. If set to false, evaluates on monolingual dataset.")
    parser.add_argument("--model_name", type=str, help="Model name or path")
    parser.add_argument("--alpha", type=float, default=2.0, help="Steering strength alpha")
    parser.add_argument("--beta", type=float, default=1.0, help="Steering strength beta")
    parser.add_argument("--scaling", type=str, default="norm", help="Scaling method to adapt language vectors (options: 'norm', 'factor', 'relative_norm')")
    parser.add_argument("--restore_norm", action="store_true", help="Whether to restore the original norm of the language vector")
    parser.add_argument("--version", type=str, default="v?", help="String describing the version of the steering method (used for saving).")
    parser.add_argument("--path", type=str, help="Path to load a trained steer module")
    parser.add_argument("--lang", type=str, default=None, help="Language code (e.g., 'deu' for German). If None, use all languages in the dataset.")
    parser.add_argument("--include_lang", type=int, default=1, help="Whether to include the language in the prompt.")
    parser.add_argument("--skip_layers", nargs='+', help="Layers to skip for steering (e.g., '0 1 2')")
    parser.add_argument("--subset", type=str, default=None, help="Subset of the dataset to evaluate on.")
    parser.add_argument("--num_shots", type=int, default=0, help="Number of in-context examples to use. Deafault is 0 (zero-shot).")


    args = parser.parse_args()

    # Display args
    print(args)
    
    print("Generating completions for llama_only: ", str(args.llama_only), "dataset: ", args.crosslingual)
   
    model_name = args.model_name 
    vector_path = "../collect_language_vectors/language_vectors_bucket/flores_plus/{}/full".format(model_name.split("/")[-1])
    
    max_new_tokens = 256

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.eos_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "gemma" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
   
    model.resize_token_embeddings(len(tokenizer))
    if not args.llama_only:
        steering = Steer(
            model, 
            path=vector_path, 
            arithmetic="alpha" if args.path is not None else "naive", 
            remove_content=False if args.path is not None else True, 
            anchor="eng", 
            alpha=args.alpha, 
            beta=args.beta,
            scaling_mode=args.scaling, 
            restore_norm=args.restore_norm,
            adaptive_alpha=True if args.path is not None else False, 
            skip_layers=[int(i) for i in args.skip_layers] if args.skip_layers is not None else None,
        )
        if args.path is not None:
            steering.load_intervention(args.path)
        steering.to("cuda")

        print("Steering loaded")
    
    evaluator = LCBEvaluator(
        steering if not args.llama_only else model,
        tokenizer, 
        dataset_name="multiQ", 
        crosslingual=args.crosslingual, 
        steering=steering, 
        max_new_tokens=max_new_tokens, 
        include_lang=args.include_lang==1, 
        lang=args.lang, 
        subset=args.subset, 
        use_steer=args.llama_only==False,
        version=args.version,
        shots=args.num_shots,
        lang_key="a_lang" if args.crosslingual else "language", 
        source_lang_key="q_lang" if args.crosslingual else "language",
    )
    evaluator.load_dataset(crossling=args.crosslingual, subset=args.subset)
    evaluator.evaluate(max_new_tokens=max_new_tokens,)

if __name__ == "__main__":
    main()