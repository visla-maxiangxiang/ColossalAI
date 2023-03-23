from typing import Optional, Union

import loralib as lora
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_approx_kl(log_probs: torch.Tensor,
                      log_probs_base: torch.Tensor,
                      action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html
    我来做一个简单的解释：
    KL(p,q) = \sum_i{p_i*log(p_i/q_i)} = EXPa~p【-log(q_i/p_i)】
    我们发现KL有很多项构成，每一项有正有负，但总体上是正的。
    虽然上面是一个期望表达式，理论上可以利用蒙塔卡罗法进行估算，但是如果我们采样出的样本导致q_i>p_i时，估算出的KL竟然是负的。
    显然-log(q_i/p_i)这个表达式的方差很大，需要采样出大量的样本才能利用蒙塔卡罗法进行估算。
    但是怎么衡量每一项对于整体的贡献呢? KL(p,q) = sum_i {distance(p_i,q_i)} 保障distance(p_i,q_i)都是正的。
    可以这样改写:
    KL(p,q) = \sum_i{p_i*-log(q_i/p_i)} = \sum_i{p_i*[-log(q_i/p_i) + (q_i/p_i -1)]}
    因为\sum_i{p_i*[q_i/p_i - 1]} = sum_i{q_i - p_i} = sum_i{q_i} - sum_i{p_i} = 1-1 = 0
    令：ratio_i=q_i/p_i ; log_ratio_i = log(q_i/p_i) 
    则：KL(p,q) = \sum_i{p_i*[(ratio_i - 1) - log_ratio_i]} = EXPa~p【ratio_i - 1 - log_ratio_i】。
    我们观测期望中的表达式: ratio_i - 1 - log_ratio_i，我们发现该表达式的值恒为正，并且当 q_i = p_i时值为0。
    并且ratio_i-1-log_ratio_i的方差比 -log_ratio_i小的多，更加适合利用蒙塔卡罗法进行采样估计。
    于是我们利用策略p只进行一次采样，得到了一个动作a_i，带入后算法后得到了 KL ~= ratio_i -1 - log_ratio_i = ratio_i.exp() - 1 - log_ratio_i
    当然只进行一次采样所估算出的kl距离还是存在很大误差的，但至少保障了估算出的KL距离为正了，并且当p、q策略收敛且大致相等时，利用p采样出的动作非常集中kl->0，则估算的KL会非常精确。
    因此上面的算法只能用在策略p大致收敛的场景中，
    下面的代码就是利用上面所描述的算法进行kl估算的。
    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    approx_kl = (log_ratio.exp() - 1) - log_ratio
    if action_mask is not None:
        approx_kl = masked_mean(approx_kl, action_mask, dim=1)
        return approx_kl
    approx_kl = approx_kl.mean(dim=1)
    return approx_kl


def compute_reward(r: Union[torch.Tensor, float],
                   kl_coef: float,
                   log_probs: torch.Tensor,
                   log_probs_base: torch.Tensor,
                   action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if kl_coef <= 0.0:
        return r
    #这里计算的并不是kl，甚至也不是kl的近似，只是kl在token真实取值上的分量。
    #直观上的意义是kl新旧策略在sequence项概率取值距离。表征了新旧策略差异的大小。
    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    #返回的奖励=RM(prompt,response) - 新旧策略的差异大小*正因子
    reward = r - kl_coef * kl
    return reward


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

#有些token是无效的，计算平均时忽略掉这些token。
def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor / (mask_sum + 1e-8)
    return mean


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


def normalize(tensor: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    mean = tensor.mean(dim)
    mean_centered = tensor - mean
    var = (mean_centered**2).mean(dim)
    norm = mean_centered * var.clamp(min=eps).rsqrt()
    return norm


def convert_to_lora(model: nn.Module,
                    input_size: int,
                    output_size: int,
                    lora_rank: int = 16,
                    lora_alpha: int = 1,
                    lora_dropout: float = 0.,
                    fan_in_fan_out: bool = False,
                    merge_weights: bool = True):
    if lora_rank > min(input_size, output_size):
        raise ValueError(f"LoRA rank {lora_rank} must be less or equal than {min(input_size, output_size)}")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._modules[name] = lora.Linear(input_size,
                                                output_size,
                                                r=lora_rank,
                                                lora_alpha=lora_alpha,
                                                lora_dropout=lora_dropout,
                                                fan_in_fan_out=fan_in_fan_out,
                                                merge_weights=merge_weights)
