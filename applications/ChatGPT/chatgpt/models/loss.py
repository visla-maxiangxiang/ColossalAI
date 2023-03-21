from typing import Optional

import torch
import torch.nn as nn

from .utils import masked_mean


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    该项损失用于微调SFT模型。
    SFT模型从一个预训练序列生成模型进行初始化，例如可以是一个GPT模型。
    """

    def __init__(self):
        super().__init__()
        #交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        #参数：logits是一个形状为[B,L,D]的张量。表征批次大小、token序列长度、词典大小，值表示token取值的对数似然。
        #参数：labels是一个形状为[B,L]的张量。表征批次大小、token序列长度，值表示token的真正取值。
        #因为下一个token的概率分布取决于当前token位置的输出的logits，因此二者相差一个位置，因此需要进行一次错位对齐。
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        #求交叉熵。最小化该损失函数可以最大化真实序列的log-likely-hood。
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class PolicyLoss(nn.Module):
    """
    采用PPO算法最大化期望回报
    本项目对于文本生成任务所定义的强化学习问题是一个多臂老虎机问题（MAD)，整个决策过程只做了一个动作。
        状态：s = prompt
        动作：a = response
        奖励：r = RM(prompt, response)
    因为只有一个action，所以奖励r也就是等价于动作价值函数Q(s,a),即：
        Q(s,a) = RM(s,a)=RM(prompt, response)
    因此可以最大化Q(s,a)的期望来进行优化：
        EXP(\theta){Q} = \sum_{s,a}{P(s,a)Q(s,a)} = \sum_{s,a}{P(s)\pi(a|s;\theta)Q(s,a)}
        argmax_\theta {EXP(\theta)}
    但是这个算法是一个on-line算法，必须使用最新的策略模型采样出的数据来优化当前模型。为了能够使用历史数据，因此用到了重要性采样技巧：
        EXP(\theta){Q} = \sum_{s,a}{P(s,a)Q(s,a)} = \sum_{s,a}{P(s)\pi(a|s;\theta_old)Q(s,a)\pi(a|s;\theta)/pi(a|s;\theta_old)} 
      = EXP(\theta_old){Q\pi(a|s;\theta)/pi(a|s;\theta_old)}
    这样就可以使用旧的策略采样的（prompt,response）数据来对新的策略进行优化了。
    为了降低策略梯度的方差，PPO算法会用A(s,a) = Q(s,a) - V(s)替换掉损失函数中的Q(s,a)，可以证明该替换不会引入任何梯度偏差。
        其中V(s) = V(prompt) = critical(prompt)，也就是critical网络的输出。因此critical网络的输出是与response无关的。
    可以在数学上证明V(s) = EXP_{a~pi}[Q(s,a)]时策略梯度的方差是最小的。
        因此利用：(V(s)-Q(s,a))^2 = （V(s) - RM(s,a)）^2来作为损失函数优化critical网络。
        数学上可以证明使用此损失函数，V(s)会收敛到EXP_{a~pi}[Q(s,a)]处。
    采用重要性采样技巧可以重复利用旧策略生成的数据，且不会引入偏差。但是如果新旧策略差异较大时，却会带来很大的梯度方差，这一点可以参考TRPO的论文。
        因此PPO算法又引入了一项策略函数损失项KL(old,new)用来约束新策略输出的概率分布不会过分偏离旧策略对应的概率分布。
        因为KL需要知道每个token取值对应的概率，需要存储的数据量非常大，因此openai又提出了PPO_clip算法。
        PPO_clip算法时PPO_kl算法的近似，但只需要存储真实token取值的概率即可。节省内存也能减少训练量，但因为毕竟只是一种近似，所有引入了新的梯度偏差。
        openai认为PPO_kl以及PPO_clip所引入的梯度偏差是有益的，坚定了Agent的信心，让策略向着一个虽然不太正确（大致还是准确的）但是明确的方向快速地优化。
        
    总结：
        - 本文将P(response|prompt)视作单步决策的多臂老虎机(MAD)问题。
        - PPO算法是由策略梯度算法+重要性采样技巧+有偏估计技巧的综合来的。
        - PPO算法需要两个模型：Actor、critical。
        - Actor模型采用PPO_clip作为损失函数。
        - critical模型采用(V(prompt)-RM(prompt,response))^2作为损失函数。
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    #下面的这个计算PPO_clip的算法是有问题的。
    def forward(self,
                log_probs: torch.Tensor,
                old_log_probs: torch.Tensor,
                advantages: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #参数log_probs：一个形状为[B,L]的序列，表示批次、序列长度。值表示token取真实值时在新策略模型下的对数似然。
        #参数old_log_probs：也是一个形状为[B,L]的序列。值表示token取真实值时在旧策略模型下的对数似然。
        #action_mask是一个形状为[B,A]的序列，B表示批次、A表示response对应的token序列的长度。值表示response中
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        if action_mask is not None:
            loss = masked_mean(loss, action_mask)
        loss = loss.mean()
        return loss


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.4) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(self,
                values: torch.Tensor,
                old_values: torch.Tensor,
                reward: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
        surr1 = (values_clipped - reward)**2
        surr2 = (values - reward)**2
        loss = torch.max(surr1, surr2)
        loss = loss.mean()
        return loss


class PPOPtxActorLoss(nn.Module):
    """
    To Do:

    PPO-ptx Actor Loss
    """

    def __init__(self, policy_clip_eps: float = 0.2, pretrain_coef: float = 0.0, pretrain_loss_fn=GPTLMLoss()) -> None:
        super().__init__()
        self.pretrain_coef = pretrain_coef
        self.policy_loss_fn = PolicyLoss(clip_eps=policy_clip_eps)
        self.pretrain_loss_fn = pretrain_loss_fn

    def forward(self,
                log_probs: torch.Tensor,
                old_log_probs: torch.Tensor,
                advantages: torch.Tensor,
                lm_logits: torch.Tensor,
                lm_input_ids: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        policy_loss = self.policy_loss_fn(log_probs, old_log_probs, advantages, action_mask=action_mask)
        lm_loss = self.pretrain_loss_fn(lm_logits, lm_input_ids)
        return policy_loss + self.pretrain_coef * lm_loss


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward)
        log_probs = torch.log(probs)
        loss = -log_probs.mean()
        return loss
