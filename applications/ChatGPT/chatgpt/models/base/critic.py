from typing import Optional

import torch
import torch.nn as nn

from ..lora import LoRAModule
from ..utils import masked_mean


class Critic(LoRAModule):
    """
    Critic model base class.

    Args:
        model (nn.Module): Critic model.
        value_head (nn.Module): Value head to get value.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        model: nn.Module,
        value_head: nn.Module,
        lora_rank: int = 0,
        lora_train_bias: str = 'none',
        use_action_mask: bool = False,
    ) -> None:

        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        self.value_head = value_head
        self.use_action_mask = use_action_mask
        self.convert_to_lora()

    def forward(self,
                sequences: torch.LongTensor,
                action_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']

        values = self.value_head(last_hidden_states).squeeze(-1)
        # 如果此时返回 values[-num_actions:],那么模型就能预测response中每个token的价值了，也就是Q((response+prompt)[0:i])。
        #在trl版采用MDP进行强化学习的RLHF中就是这样实现的。

        if action_mask is not None and self.use_action_mask:
            num_actions = action_mask.size(1)
            prompt_mask = attention_mask[:, :-num_actions]
            values = values[:, :-num_actions] 
            #利用prompt_mask，只计算与prompt相关的部分，也就是value只与prompt有关，与response无关。
            value = masked_mean(values, prompt_mask, dim=1)
            #输出的value是一个形状为[B]的张量，表示prompt对应状态的价值即V(prompt)。在Actor-Critical系算法里为critical(state)=critical(prompt)
            return value

        #不采用action_mask时，那么sequeces只包含prompt而不包含response部分。并且批次中所有的prompt的有效序列长度都一致。
        values = values[:, :-1]
        value = values.mean(dim=1)
        return value
