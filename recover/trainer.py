import os
import torch
from transformers import Trainer, Seq2SeqTrainer
from torch.utils.data import DataLoader


class SteeringTrainer(Trainer):

    def _save(self, output_dir = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.model.train_mode == "steer":
            WEIGHTS_NAME = "steering.bin"
            
            os.makedirs(output_dir, exist_ok=True)
            if state_dict is None:
                state_dict = self.model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if "projector" in k or "intervention" in k}
            torch.save(filtered_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        elif self.model.train_mode == "model":
            state_dict = self.model.model.state_dict()
            self.model.model.save_pretrained(
                output_dir, state_dict=state_dict
            )
        elif self.model.train_mode == "adapter":
            self.model.model.save_adapter(
                output_dir, adapter_name=self.model.adapter_name,
            )


class SteeringTrainerForCausalLM(SteeringTrainer):
    
    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self._train_batch_size, sampler=None, collate_fn=self.data_collator)

