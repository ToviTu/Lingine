from transformers import CLIPModel, CLIPConfig


class CLIPRewardModel(CLIPModel):
    """
    This model aims at converting the CLIP model into a reward model that
    predicts the intermediate value of
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)

        # Compute the similarity as the reward
        reward = outputs.text_embeds @ outputs.image_embeds.T
        return reward
