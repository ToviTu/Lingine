import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
)
from accelerate import init_empty_weights

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from PIL import Image
import os
from dataclasses import dataclass
from loguru import logger

from lingine.model.llava import (
    Llava,
    LLAVA_PRETRAIN_TEMPLATE,
    LLaVAConfig,
    LLaVAPretrainArguments,
    MMPretrainConfig,
)

if __name__ == "__main__":
    parser = TrlParser((LLaVAPretrainArguments, MMPretrainConfig, LLaVAConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model, Tokenizer & Processor
    ################
    logger.info(f"Loading model from {model_args.config_file}")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    assert os.path.exists(
        model_args.config_file
    ), f"Config file not found: {model_args.config_file}"
    with init_empty_weights():
        model = Llava(AutoConfig.from_pretrained(model_args.config_file))
    processor = model.init_for_mmpretrain(model.config, **model_kwargs)
    model.config.to_json_file(model_args.config_file)

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
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(example["conversations"], tokenize=False)
            for example in examples
        ]

        images = [
            Image.open(os.path.join(script_args.image_path, example["image"]))
            for example in examples
        ]  # First image only

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        image_token_id = processor.tokenizer.convert_tokens_to_ids(
            processor.image_token
        )
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    ################
    # Dataset
    ################
    logger.info(f"Loading dataset from {script_args.dataset_name}")
    dataset = load_dataset(
        "json",
        data_files=script_args.dataset_name,
    )

    ################
    # Training
    ################
    logger.info("Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=processor.tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)
