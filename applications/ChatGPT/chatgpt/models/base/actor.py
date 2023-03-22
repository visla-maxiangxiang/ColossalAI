from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..generation import generate
from ..lora import LoRAModule
from ..utils import log_probs_from_logits


class Actor(LoRAModule):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self, model: nn.Module, lora_rank: int = 0, lora_train_bias: str = 'none') -> None:
        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        self.convert_to_lora()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        return_action_mask: bool = True,
        **kwargs
    ) -> Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]:
        '''
            参数input_ids：是一个形状为[B,L]的张量，表示B个prompt的token构成的张量，L由其中长度最大的prompt决定。input_ids中的有效位采用右对齐，其左半部分用<PAD>填充。
        '''
        #利用完全随机采样进行序列生成。
        #PPO算法要求必须采用完全随机采样，不能使用greedy与beam-search这类确定性的采样策略，也不能用半随机的topn_sample算法，否则会引入新的偏差。
        sequences = generate(self.model, input_ids, **kwargs)
        attention_mask = None
        pad_token_id = kwargs.get('pad_token_id', None)
        if pad_token_id is not None:
            #将token取值不为<pad>的位置上赋予1，表示相应的token位置是有效的，其他位置置为0表示是无效的token位置。
            attention_mask = sequences.not_equal(pad_token_id).to(dtype=torch.long, device=sequences.device)
        if not return_action_mask:
            return sequences, attention_mask, None
        input_len = input_ids.size(1)
        eos_token_id = kwargs.get('eos_token_id', None)
        if eos_token_id is None:
            action_mask = torch.ones_like(sequences, dtype=torch.bool)
        else:
            # left padding may be applied, only mask action
            # 因为response是以<eos>这个特殊token作为结束标记的，其之前的位置True，其它置False.
            action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
            #把prompt对应的token包括prompt的结束标记全部在action_mask中标记为True。
            action_mask = F.pad(action_mask, (1 + input_len, -1), value=True)    # include eos token and input
        #将prompt对应的token位置False
        action_mask[:, :input_len] = False
        #进行一次偏移，跳过<bos>特殊标记。
        action_mask = action_mask[:, 1:]
        #sequences.size(1) - input_len表示的时response序列的最大长度。action_mask做的切片操作实际上是忽略了prompt只保留了response部分的标记。
        return sequences, attention_mask, action_mask[:, -(sequences.size(1) - input_len):]

    def forward(self,
                sequences: torch.LongTensor,
                num_actions: int,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        """
            参数sequences：一个形状[B,L]的张量，B表示批次大小、L表示prompt+response的长度。值是整数，表示序列中每个token位上的取值。
            参数num_actions: response部分的长度，根据此参数可以确定sequence中prompt与response的界线。
            参数attention_mask：形状为[B,L]的张量。其值只能取0和1。0表示无效的token，也就是被用<PAD>填充的token位，1表示有效的token。
            Returns action log probs
        """
        #利用内部的序列生成模型来输出序列上每个token位置上取每个值的对数似然。
        output = self.model(sequences, attention_mask=attention_mask)
        logits = output['logits'] #logits是一个形状为[B,L,T]的张量T表示token词典大小。表示某序列某位置token取某值的对数似然。
        #计算每个token位上取sequences标示的值的对数似然。
        log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
        #输出response中每个动作的对数似然
        return log_probs[:, -num_actions:]
