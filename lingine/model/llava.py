import torch
from typing import Optional

from transformers import (
    LlavaForConditionalGeneration,
    CLIPVisionModel,
    AutoModelForCausalLM,
    LlavaProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
)

from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
from transformers.models.llava.configuration_llava import LlavaConfig
from dataclasses import dataclass
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
)

DEFAULT_IMAGE_TOKEN = "<image>"

LLAVA_PRETRAIN_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['from'] == 'human' %}"
    "{% generation %}"
    "{{ 'USER: ' + message['value'] + '\n' }}"
    "{% endgeneration %}"
    "{% else %}"
    "{{ 'ASSISTANT: ' + message['value'] }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
)

LLAVA_LLAMA_3_2_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['from'] == 'human' %}"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{{ message['value'] ~ '<|eot_id|>' }}"
    "{% else %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{{ message['value'] ~ '<|eot_id|>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)


model2template = {
    "lmsys/vicuna-7b-v1.5": LLAVA_PRETRAIN_TEMPLATE,
    "meta-llama/Llama-3.2-1B-Instruct": LLAVA_LLAMA_3_2_CHAT_TEMPLATE,
}


class Llava(LlavaForConditionalGeneration):
    """
    A customized hf version of LlavaForConditionalGeneration to support training from scratch.
    """

    def __init__(self, config: LlavaConfig):
        super().__init__(config)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: Optional[bool] = False,
    ) -> torch.nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of, mean_resizing
        )
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def init_for_mmpretrain(self, config: LlavaConfig, **kwargs):
        """
        An alternative intialization method for training from scratch.
        """

        # Initialize vision tower
        if "vision_tower" in self.config:
            self.vision_tower = CLIPVisionModel.from_pretrained(
                self.config.vision_tower,
                **kwargs,
            )
        else:
            print(
                "Vision tower not found in config...defaulting to openai/clip-vit-large-patch14-336"
            )
            self.vision_tower = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-large-patch14-336",
                **kwargs,
            )
            self.config.vision_tower = "openai/clip-vit-large-patch14-336"

        # Update config
        for k in self.config.vision_config:
            setattr(
                self.config.vision_config,
                k,
                getattr(self.vision_tower.config, k, ""),
            )

        # Initialize multi-modal projector
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.multi_modal_projector.to(
            device=self.vision_tower.device, dtype=self.vision_tower.dtype
        )

        # Initialize language model
        self.language_model = AutoModelForCausalLM.from_pretrained(
            getattr(self.config.text_config, "_name_or_path"),
            **kwargs,
        )

        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"language_model.{k}" for k in self.language_model._tied_weights_keys
            ]

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

        # Update config
        for k in self.config.text_config:
            setattr(
                self.config.text_config, k, getattr(self.language_model.config, k, "")
            )

        self.post_init()

        print("The config file is likely to have changed...")

        # Prepare processors

        # Add image token to tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.text_config._name_or_path)

        print(
            "Registering image token...This operation resizes the token embeddings which can take a while."
        )
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN])
        self.resize_token_embeddings(
            new_num_tokens=len(tokenizer),
            pad_to_multiple_of=64,
            mean_resizing=getattr(kwargs, "mean_resizing", False),
        )

        self.config.vocab_size = len(tokenizer)
        self.config.image_token_index = tokenizer.convert_tokens_to_ids(
            DEFAULT_IMAGE_TOKEN
        )
        self.config.image_seq_length = (
            self.config.vision_config.image_size // self.config.vision_config.patch_size
        ) ** 2
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        self.config.pad_token_id = tokenizer.pad_token_id

        # Initialize image processor
        image_processor = CLIPImageProcessor.from_pretrained(self.config.vision_tower)

        return LlavaProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=self.config.vision_config.patch_size,
            chat_template=model2template.get(
                self.config.text_config._name_or_path, LLAVA_PRETRAIN_TEMPLATE
            ),
            vision_feature_select_strategy="default",
            image_token=DEFAULT_IMAGE_TOKEN,
            num_additional_image_tokens=1,
        )


@dataclass
class LLaVAConfig(ModelConfig):
    config_file: str = ""


@dataclass
class MMPretrainConfig(SFTConfig):
    freeze_vision_model: bool = True
    freeze_mm_projector: bool = True
    freeze_language_model: bool = False


@dataclass
class LLaVAPretrainArguments(ScriptArguments):
    dataset_name: str = ""
    image_path: str = ""
