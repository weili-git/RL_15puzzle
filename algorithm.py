import copy
import parl
import torch
import torch.nn as nn


class DQN(parl.Algorithm):
    def __init__(self, model, gamma=None, lr=None):
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.model = model
        self.target_model = copy.deepcopy(model)

        self.gamma = gamma
        self.lr = lr

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(lr=lr, params=self.model.parameters())

    def predict(self, obs):
        return self.model(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        # 每个动作的reward
        pred_values = self.model(obs)
        action_dim = pred_values.shape[-1]  # squeeze(axis=-1)?
        action_onehot = nn.functional.one_hot(action, num_classes=action_dim)
        # 动作action对应的reward，遵循e_greed
        pred_value = (pred_values * action_onehot).sum(dim=1, keepdim=True)

        with torch.no_grad():
            max_v, _ = self.target_model(next_obs).max(dim=1, keepdim=True)
            # G_t: 未来收益之和 = reward + gamma * G_t+1
            # 用之前算好的target值近似G_t+1
            target = reward + (1 - terminal.int()) * self.gamma * max_v
        loss = self.mse_loss(pred_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def sync_target(self):
        self.model.sync_weights_to(self.target_model)


