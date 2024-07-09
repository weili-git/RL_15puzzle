import parl
import torch.nn as nn
import torch


class Model(parl.Model):
    def __init__(self, obs_dim, act_dim):   # 通用模型
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # [batch, sz*sz, sz, sz] => [batch, act_dim]
        self.model = nn.Sequential(
            nn.Conv2d(self.obs_dim * self.obs_dim, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self.obs_dim * self.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.act_dim)
        )

    def forward(self, obs):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        x = self.model(obs)
        return x
