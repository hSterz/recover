# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by the Cambridge Language Technology Lab
from copy import deepcopy
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
from torch import nn

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from steering import Steer
from trainer import SteeringTrainerForCausalLM
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_train_samples_per_lang: Optional[int] = field(
        default=1000,
    )
    max_eval_samples_per_lang: Optional[int] = field(
        default=100,
    )

@dataclass
class RecoverArguments:
    min_steps_per_iteration: Optional[int] = field(
        default=None,
        metadata={
            "help": "Minimum of steps per parameter selection iteration during sparse fine-tuning."
        },
    )
    max_steps_per_iteration: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum of steps per parameter selection iteration during sparse fine-tuning."
        },
    )
    max_epochs_per_iteration: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of epochs per parameter selection iteration during sparse fine-tuning."
        },
    )
    tie_embedding: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to tie the embeddings of the encoder and the decoder."
        },
    )
    train_embedding: Optional[bool] = field(
        default=True,
    )
    rank: Optional[int] = field(
        default=128,
        metadata={
            "help": "The rank of the low-rank approximation."
        },
    )
    intervention_type: Optional[str] = field(   
        default="ad",
        metadata={
            "help": "The type of intervention."
        },
    )
    act_fn: Optional[str] = field(
        default="linear",
        metadata={
            "help": "The activation function."
        },
    )
    layer_wise_AD: Optional[bool] = field(
        default=False
    )
    crosslingual: Optional[bool] = field(
        default=False
    )
    num_buckets: Optional[int] = field(
        default=-1
    )
    skip_layers: Optional[List[int]] = field(
        default_factory=list,
        metadata={
            "help": "The layers to skip."
        },
    )
                                    



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, RecoverArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, reft_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, reft_args, training_args = parser.parse_args_into_dataclasses()



    langs = [
        "en", "es", "fr", "de", "pt", "ru", "zh", "ja",
        "ar", "hi", "id",
        "he", "ta", "fa", "th", "pl", "nl", "bn"
    ]
    lang_map = {
        "en": "eng",
        "es": "spa",
        "fr": "fra",
        "de": "deu",
        "pt": "por",
        "ru": "rus",
        "zh": "cmn",
        "ja": "jpn",
        "ar": "arb",
        "hi": "hin",
        "id": "ind",
        "he": "heb",
        "ta": "tam",
        "fa": "pes",
        "th": "tha",
        "pl": "pol",
        "nl": "nld",
        "bn": "ben",
    }
    lang_map_inv = {v: k for k, v in lang_map.items()}
    id2lang = {i: lang_map[lang] for i, lang in enumerate(langs)}
    lang2id = {lang_map_inv[lang]: i for i, lang in id2lang.items()}
    vector_samples = "full"
    vector_path = "../collect_language_vectors/language_vectors_bucket/flores_plus/{}/{}".format(model_args.model_name_or_path.split("/")[-1], vector_samples)
  

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    dtype = torch.bfloat16
    raw_datasets = datasets.load_from_disk("data/translation_combined")
    
    use_lang_prompt =  [False for i in range(len(raw_datasets))]
    raw_datasets = raw_datasets.add_column("use_lang_prompt", use_lang_prompt)
    raw_datasets = DatasetDict({"train": raw_datasets})

    
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if tokenizer.pad_token == None:
        if tokenizer.unk_token == None and tokenizer.pad_token == None:
            # raw llama3
            print("adding a special padding token...")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            tokenizer.pad_token = tokenizer.unk_token


    if model_args.model_name_or_path:
       
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            torch_dtype=dtype,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    model.train()
    steering = Steer(
        model, 
        path=vector_path, 
        arithmetic="intervene", 
        remove_content=False,
        rank=reft_args.rank,
        beta=0.9,
        scaling_mode="norm", 
        restore_norm=False,
        adaptive_alpha=True,
        layer_wise_AD=reft_args.layer_wise_AD,
        intervention=reft_args.intervention_type,
        crossling=reft_args.crosslingual,
        num_buckets=reft_args.num_buckets,
        skip_layers=reft_args.skip_layers,
    ) 
    steering.id2lang = id2lang
    steering.to("cuda")
    steering.train_steer()

    # Filter out unsupported languages
    raw_datasets = raw_datasets.filter(lambda x: lang_map[x["prompt_lang"]] in steering.vectors)
    raw_datasets = raw_datasets.filter(lambda x: lang_map[x["answer_lang"]] in steering.vectors)
    assert len(raw_datasets["train"]) > 0, "No supported languages found in the dataset."

    print("Steering model is ready.")
    print(steering.crossling)

    
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    
    # We tokenize every text, then concatenate them together before splitting them in smaller parts.
    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.
    
    
    def tokenize_function(examples):
        question = examples["translations"][0]["content"]
        if examples["use_lang_prompt"]:
            question = f"{question} {examples['language_prompt']}"
        answer = examples["translations"][1]["content"]
        
        if "Instruct" in tokenizer.name_or_path or "-it" in tokenizer.name_or_path: 
            result = {}
            system_prompt = "You are a helpful assistant."
            if "gemma" in tokenizer.name_or_path:
                # we remove the BOS, otherwise there will be redundant BOS tokens.
                base_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content":question}],  
                    tokenize=False,
                )# [len("<|begin_of_text|>"):]
                base_input = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question},
                    {"role": "assistant", "content": answer}], 
                    tokenize=False,
                ) + tokenizer.eos_token # [len("<|begin_of_text|>"):] + tokenizer.eos_token
            else:
                # we remove the BOS, otherwise there will be redundant BOS tokens.
                base_prompt = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],  
                    tokenize=False,
                )# [len("<|begin_of_text|>"):]
                base_input = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}], 
                    tokenize=False,
                ) + tokenizer.eos_token # [len("<|begin_of_text|>"):] + tokenizer.eos_token
            base_prompt_ids = tokenizer(
                base_prompt, max_length=max_seq_length, truncation=True, return_tensors="pt")["input_ids"][0]
            base_prompt_length = len(base_prompt_ids)
            base_input_ids = tokenizer(
                base_input, max_length=max_seq_length, truncation=True, return_tensors="pt")["input_ids"][0]
            output_ids = deepcopy(base_input_ids)
            output_ids[:base_prompt_length] = IGNORE_INDEX
            
            result["input_ids"] = base_input_ids
            result["labels"] = output_ids
            result["lang"] = [lang2id[examples["answer_lang"]]]
            result["source_lang"] = [lang2id[examples["prompt_lang"]]]
        else:
            result = tokenizer(examples[text_column_name], max_length=max_seq_length, truncation=True,)
            
        return result
       
            
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
    

    
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
      
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest"
    )
    print(train_dataset)


    # Initialize our Trainer
    trainer_class = SteeringTrainerForCausalLM
    trainer = trainer_class(
        model=steering,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # save 
        state_dict = steering.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if "adaptive_alpha" in k}
        if not os.path.isdir("adaptive_alpha/{}".format(model_args.model_name_or_path.split("/")[-1])):
            os.makedirs("adaptive_alpha/{}".format(model_args.model_name_or_path.split("/")[-1]))
        save_path = os.path.join(training_args.output_dir, "adaptive_alpha_final")
        steering.save_intervention(save_path)
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        args_dict = vars(reft_args)
        json_file_name = f"{training_args.output_dir}/args.json"
        with open(json_file_name, 'w') as json_file:
            json.dump(args_dict, json_file, indent=4)

        

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)
        print(metrics)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # generate some text to test the model
    user_input = "What is the capital of France?"
    system_prompt = "You are a helpful assistant."
    if "gemma" in model_args.model_name_or_path:
        base_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_input}], 
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        base_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}], 
            tokenize=False,
            add_generation_prompt=True
        )
    input_data = tokenizer(base_prompt, return_tensors="pt").to("cuda")
    steering.eval()
    model = steering.model
    model.eval()
    steering.set_lang("deu", anchor="eng")
    with torch.no_grad():
        output = model.generate(
            **input_data,
            max_new_tokens=20,
        )
        print(tokenizer.decode(output[0], skip_special_tokens=True))
         


   


if __name__ == "__main__":
    main()