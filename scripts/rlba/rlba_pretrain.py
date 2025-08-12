from loguru import logger
from lingine.model.llava import Llava
from lingine.reinforce.clip_rm import CLIPRewardModel
from lingine.reinforce.vl_grpo import VLGRPOTrainer

from accelerate import init_empty_weights, PartialState
from transformers import AutoConfig, AutoProcessor, CLIPConfig, CLIPProcessor
from trl import (
    GRPOConfig,
    TrlParser,
)

import torch
import os
from lingine.model.llava import LLaVAConfig, LLaVAPretrainArguments
from datasets import load_dataset
from dataclasses import dataclass


@dataclass
class VLGRPOConfig(GRPOConfig):
    freeze_vision_model: bool = False
    freeze_mm_projector: bool = False
    freeze_language_model: bool = False


@dataclass
class ThisScriptArguments(LLaVAPretrainArguments):
    reward_model: str = ""


PROMPT_TEMPLATE = (
    "{% for message in messages %}"
    "user\n"
    "{{ message.content }}\n"
    "{% endfor %}"
    "assistant\n\n"
)

if __name__ == "__main__":
    parser = TrlParser((ThisScriptArguments, VLGRPOConfig, LLaVAConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model, Tokenizer & Processor
    ################

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=None,
    )

    if model_args.config_file:
        logger.info(f"Loading model from {model_args.config_file}")
        assert os.path.exists(
            model_args.config_file
        ), f"Config file not found: {model_args.config_file}"
        # This is probably a new model initialized from scratch
        with init_empty_weights():
            model = Llava(AutoConfig.from_pretrained(model_args.config_file))
        processor = model.init_for_mmpretrain(
            model.config, mean_resizing=script_args.mean_resizing, **model_kwargs
        )
        model.config.to_json_file(model_args.config_file)
    elif model_args.model_name_or_path:
        logger.info(f"Loading model from {model_args.model_name_or_path}")
        model = Llava.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError("Either a model name or a config file must be provided")

    processor.bos_token_id = processor.tokenizer.bos_token_id
    processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id
    processor.chat_template = PROMPT_TEMPLATE

    if training_args.freeze_vision_model:
        for param in model.vision_tower.parameters():
            param.requires_grad = False
    if training_args.freeze_mm_projector:
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False
    if training_args.freeze_language_model:
        for param in model.language_model.parameters():
            param.requires_grad = False

    ################
    # Dataset
    ################
    dataset = load_dataset("json", data_files=script_args.dataset_name, split="train")

    # Prepare dataset
    logger.info("Preparing dataset...")

    def prepare_dataset(example):
        formatted_example = {
            "prompt": [
                {
                    "role": "user",
                    "content": example["conversations"][0]["value"],
                }
            ],
            "gt": [
                {
                    "role": "assistant",
                    "content": example["conversations"][1]["value"]
                    + processor.tokenizer.eos_token,
                }
            ],
            "image": os.path.join(script_args.image_path, example["image"]),
        }
        return formatted_example

    with PartialState().local_main_process_first():
        dataset = dataset.map(
            prepare_dataset,
            batched=False,
            remove_columns=dataset.column_names,
        )

    ################
    # Reward functions
    ################

    # 1. Bi-modal alignment reward

    if "Long" in script_args.reward_model:
        logger.info("Using a reward model based on the LongCLIP")
        rm_config = CLIPConfig.from_pretrained(
            script_args.reward_model,
            torch_dtype=torch_dtype,
            ignore_mismatched_sizes=True,
        )
        rm_config.text_config.max_positions_embeddings = 248  # hardcoded for now
        reward_model = CLIPRewardModel.from_pretrained(
            script_args.reward_model, config=rm_config
        )
        reward_processor = CLIPProcessor.from_pretrained(
            script_args.reward_model,
            padding="max_length",
            max_length=248,
            truncate=True,
        )
        reward_processor.pad_token_id = reward_processor.tokenizer.pad_token_id
    else:
        logger.info("Using a reward model based on the CLIP")
        reward_model = CLIPRewardModel.from_pretrained(
            script_args.reward_model,
            torch_dtype=torch_dtype,
            ignore_mismatched_sizes=True,
        )
        reward_processor = AutoProcessor.from_pretrained(script_args.reward_model)
        reward_processor.pad_token_id = reward_processor.tokenizer.pad_token_id

    # 2. Other rewards

    def length_reward(completions, **kwargs):
        return [
            (
                float(len(c[0]["content"].split()))
                / 512
                # if processor.tokenizer.eos_token_id in c[0]
                # else 0
            )
            for c in completions
        ]

    trainer = VLGRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[reward_model],
        reward_processing_classes=[reward_processor],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
