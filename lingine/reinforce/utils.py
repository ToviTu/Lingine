import torch
import transformers
from typing import List


def get_vl_reward(
    model: torch.nn.Module,
    query_images: torch.Tensor,
    responses: List[str],
    sequence_lengths: int,
    processor: transformers.AutoProcessor,
) -> tuple[torch.Tensor, torch.Tensor]:

    # Assume the query_image are preprocessed by the same processor as the reward_model
    inputs = processor(
        text=responses,
        return_tensors="pt",
    ).to(model.device)
    inputs["pixel_values"] = query_images.to(model.device)

    outputs = model(**inputs)
    reward_logits = outputs.logits_per_image.T  # (B, 1)

    return (
        reward_logits,
        reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1),
    )
