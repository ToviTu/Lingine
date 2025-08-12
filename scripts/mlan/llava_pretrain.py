import torch
from datasets import load_dataset
from transformers import AutoConfig, LlavaProcessor
from accelerate import init_empty_weights

torch.compile(disable=True)

from trl import (
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from PIL import Image
import os
from loguru import logger

from lingine.model.llava import (
    Llava,
    LlavaTrainer,
    LLAVA_PRETRAIN_TEMPLATE,
    LLAVA_FINETUNE_TEMPLATE,
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

    # Pretrain
    if model_args.config_file:
        logger.info(f"Loading model from {model_args.config_file}")
        assert os.path.exists(
            model_args.config_file
        ), f"Config file not found: {model_args.config_file}"
        with init_empty_weights():
            model = Llava(AutoConfig.from_pretrained(model_args.config_file))
        processor = model.init_for_mmpretrain(
            model.config, mean_resizing=script_args.mean_resizing, **model_kwargs
        )
        model.config.to_json_file(model_args.config_file)
    # Fine-tune
    else:
        logger.info(f"Loading model from {model_args.model_name_or_path}")
        model = Llava.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )
        processor = LlavaProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        processor.chat_template = LLAVA_FINETUNE_TEMPLATE

    # Freeze the vision model
    for param in model.vision_tower.parameters():
            param.requires_grad = False
    if training_args.freeze_vision_model:
        pass
    else:
        # Unfreeze the lower half of the vision model
        num_layers = len(model.vision_tower.vision_model.encoder.layers)
        for i in range(num_layers // 2, num_layers):
            for param in model.vision_tower.vision_model.encoder.layers[i].parameters():
                param.requires_grad = True
        model.vision_tower.vision_model.post_layernorm.requires_grad = True
    if training_args.freeze_mm_projector:
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False
    if training_args.freeze_language_model:
        for param in model.language_model.parameters():
            param.requires_grad = False

    if processor.chat_template == LLAVA_PRETRAIN_TEMPLATE:
        logger.warning(
            "Using the default chat template for pretraining which ignores user instructions."
        )

    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        if script_args.use_prompt_template:
            # Use the chat template to format the conversations
            texts = [
                processor.apply_chat_template(example["conversations"], tokenize=False)
                + processor.tokenizer.eos_token
                for example in examples
            ]
        else:
            # Use the caption only
            texts = [
                processor.image_token
                + example["conversations"][-1]["value"]
                + processor.tokenizer.eos_token
                for example in examples
            ]

        images = [
            (
                Image.open(
                    os.path.join(script_args.image_path, example["image"])
                ).convert("RGB")
                if example["image"] is not None
                else None
            )
            for example in examples
        ]  # First image only

        image_mask = torch.tensor([image is not None for image in images])

        # If image-text pairs and text-only pairs are mixed, discard text-only pairs
        if image_mask.sum() != 0 and image_mask.sum() != len(image_mask):
            logger.warning(
                "Mixed image-text pairs and text-only pairs in the same batch detected. This is not recommended."
            )

        crop_size = {"height": 500, "width": 500}
        images = [
            (image if mask else torch.zeros(3, crop_size["height"], crop_size["width"]))
            for image, mask in zip(images, image_mask)
        ]

        batch = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            do_rescale=False if isinstance(images[0], torch.Tensor) else True,
        )

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        # labels[labels == processor.tokenizer.bos_token_id] = -100

        # If there are any images
        if (image_mask).any():
            # Ignore the image token index in the loss computation (model specific)
            image_token_id = processor.tokenizer.convert_tokens_to_ids(
                processor.image_token
            )
            image_token_mask = labels == image_token_id
            if not (image_token_mask.sum(dim=1) == model.config.image_seq_length).all():
                logger.warning(
                    "Incomplete inputs detected. If this happens frequently, consider increasing the max_length."
                )

                ignore_mask = (
                    image_token_mask.sum(dim=1) != model.config.image_seq_length
                )
                batch["input_ids"] = batch["input_ids"][~ignore_mask]
                batch["pixel_values"] = batch["pixel_values"][~ignore_mask]
                batch["attention_mask"] = batch["attention_mask"][~ignore_mask]
                labels = labels[~ignore_mask]
                image_token_mask = image_token_mask[~ignore_mask]

            labels[image_token_mask] = -100

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
    model.is_deepspeed_training = True
    trainer = LlavaTrainer(
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
    processor.save_pretrained(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)
