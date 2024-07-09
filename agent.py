import random
import parl
import torch


class Agent(parl.Agent):
    def __init__(self, algorithm, act_dim, e_greed=0.1, e_greed_decrement=0):
        super().__init__(algorithm)
        assert isinstance(act_dim, int)
        self.act_dim = act_dim

        self.global_step = 0
        self.update_target_steps = 200  # 每个200个training steps，拷贝参数到target_model

        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement

    def sample(self, obs):  # process one episode
        sample = random.random()
        if sample < self.e_greed:
            act = random.randint(0, self.act_dim - 1)
        else:
            act = self.predict(obs)
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)
        return int(act)

    def predict(self, obs):
        pred = self.alg.predict(obs)
        act = int(pred.argmax())
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        # obs = torch.tensor(obs, dtype=torch.float16)
        # act = torch.tensor(obs, dtype=torch.float16).unsqueeze(-1)
        # reward = torch.tensor(reward, dtype=torch.float16).unsqueeze(-1)
        # next_obs = torch.tensor(next_obs, dtype=torch.float16)
        # terminal = torch.tensor(terminal, dtype=torch.float16).unsqueeze(-1)

        loss = self.alg.learn(obs, act, reward, next_obs, terminal)

        return float(loss)


