# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

"""
# This enable TP + megablock kernels + vllm flash attn3 + PEFT
accelerate launch \
    sft_tp.py \
    --config configs/sft_lora.yaml \
    --model_name_or_path openai/gpt-oss-20b \
    --packing true packing_strategy wrapped \
    --run_name 20b-full-kernels \
    --attn_implementation kernels-community/vllm-flash-attn3
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)


def main(script_args, training_args, model_args):
    # ------------------------
    # Load model & tokenizer
    # ------------------------
    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        quantization_config=quantization_config,
        tp_plan="auto",
        use_kernels=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    # --------------
    # Load dataset
    # --------------
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split)
    dataset = dataset.select(range(int(len(dataset) * 0.1)))

    # -------------
    # Train model
    # -------------
    training_args.report_to = "wandb"

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args)
