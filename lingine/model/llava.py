import torch
import torch.nn as nn
from typing import (
    Optional,
    Tuple,
    List,
    Union,
)
from transformers import (
    LlavaForConditionalGeneration,
    CLIPVisionModel,
    AutoModelForCausalLM,
    LlavaProcessor,
    AutoTokenizer,
    LlavaImageProcessor,
)
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
from transformers.utils import is_torchdynamo_compiling

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
    "{% for message in messages %}" "{{ message['value'] ~ '\n' }}" "{% endfor %}"
)

LLAVA_FINETUNE_TEMPLATE = (
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

    is_deepspeed_training: bool = False

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

    def init_for_mmpretrain(
        self, config: LlavaConfig, mean_resizing: Optional[bool] = False, **kwargs
    ):
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
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.text_config._name_or_path,
            extra_special_tokens={"image_token": DEFAULT_IMAGE_TOKEN},
        )

        print(
            "Registering image token..."
            + " This can take a while because you set mean_resizing=True"
            if mean_resizing
            else ""
        )
        self.resize_token_embeddings(
            new_num_tokens=len(tokenizer),
            pad_to_multiple_of=64,
            mean_resizing=mean_resizing,
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
        image_processor = LlavaImageProcessor.from_pretrained(
            self.config.vision_tower, do_pad=True
        )

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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: torch.Tensor = None,
        **lm_kwargs,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).


        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(
                -1
            )
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            if (
                not is_torchdynamo_compiling()
                and not self.is_deepspeed_training
                and inputs_embeds[special_image_mask].numel() != image_features.numel()
            ):
                print(type(self))
                n_image_tokens = (input_ids == self.config.image_token_index).sum()
                n_image_features = image_features.shape[0] * image_features.shape[1]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(
                    logits.device
                )
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = labels[..., 1:][
                    shift_attention_mask.to(labels.device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


from trl import SFTTrainer
from torch.utils.data import Sampler
import random


class ModalityGroupedSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = self.get_modality_grouped_indices()

    def get_modality_grouped_indices(self):
        vision_indices = [[]]
        text_indices = [[]]

        for i, example in enumerate(self.dataset):
            if example.get("image", None) is not None:
                vision_indices[-1].append(i)
            else:
                text_indices[-1].append(i)

            if len(vision_indices[-1]) == self.batch_size:
                vision_indices.append([])
            if len(text_indices[-1]) == self.batch_size:
                text_indices.append([])

        if len(vision_indices[-1]) < self.batch_size:
            vision_indices.pop()
        if len(text_indices[-1]) < self.batch_size:
            text_indices.pop()

        # Merge two lists and shuffle
        indices = text_indices + vision_indices

        if self.shuffle:
            random.shuffle(indices)
        return [i for sublist in indices for i in sublist]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class LlavaTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.args.group_by_modality:
            return ModalityGroupedSampler(
                self.train_dataset,
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
            )
        return super()._get_train_sampler()


@dataclass
class LLaVAConfig(ModelConfig):
    config_file: str = ""
    model_max_length: int = 2048


@dataclass
class MMPretrainConfig(SFTConfig):
    freeze_vision_model: bool = True
    freeze_mm_projector: bool = True
    freeze_language_model: bool = False
    group_by_modality: bool = False


@dataclass
class LLaVAPretrainArguments(ScriptArguments):
    dataset_name: str = ""
    image_path: str = ""
    use_prompt_template: bool = True
    mean_resizing: bool = False
