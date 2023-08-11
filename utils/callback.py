import os
import os
import shutil

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


# 
class SavePeftModelCallback(TrainerCallback):
    def on_save(self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs, ):
        if args.local_rank == 0 or args.local_rank == -1:
            # 
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            peft_model_dir = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_dir)
            peft_config_path = os.path.join(checkpoint_folder, "adapter_model/adapter_config.json")
            peft_model_path = os.path.join(checkpoint_folder, "adapter_model/adapter_model.bin")
            if not os.path.exists(peft_config_path):
                os.remove(peft_config_path)
            if not os.path.exists(peft_model_path):
                os.remove(peft_model_path)
            if os.path.exists(peft_model_dir):
                shutil.rmtree(peft_model_dir)
            # 
            best_checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-best")
            # 
            if os.path.exists(state.best_model_checkpoint):
                if os.path.exists(best_checkpoint_folder):
                    shutil.rmtree(best_checkpoint_folder)
                shutil.copytree(state.best_model_checkpoint, best_checkpoint_folder)
            print(f"{state.best_model_checkpoint}{state.best_metric}")
        return control
