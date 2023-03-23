import torch
from chatgpt.models.utils import compute_reward, normalize

from .base import Experience, ExperienceMaker


class NaiveExperienceMaker(ExperienceMaker):
    """
    Naive experience maker.
    利用旧的策略网络也就是initial_model生成数据。这些数据会被保存起来，用于训练新策略actor。
    生成的经验数据会被replay_buffer进行缓存，并在强化学习过程中定期更新，保障强化学习用的数据的完全随机性。
    因为一直使用同一批经验数据会引入偏差。PPO算法要求强化训练的数据必须是来自旧策略完全随机采样的，（蒙塔卡罗法的要求）。
    
    """

    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        sequences, attention_mask, action_mask = self.actor.generate(input_ids,
                                                                     return_action_mask=True,
                                                                     **generate_kwargs)
        num_actions = action_mask.size(1)

        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
        value = self.critic(sequences, action_mask, attention_mask)
        r = self.reward_model(sequences, attention_mask)

        reward = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask)

        advantage = reward - value
        # TODO(ver217): maybe normalize adv
        if advantage.ndim == 1:
            advantage = advantage.unsqueeze(-1)

        return Experience(sequences, action_log_probs, value, reward, advantage, attention_mask, action_mask)
