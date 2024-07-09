import os
import torch
import random
import torch.nn.functional as F


class Puzzle:
    def __init__(self, sz=4, col=None):
        self.sz = sz
        self.col = col if col else self.sz
        # reset board
        self.board = torch.tensor([i for i in range(1, self.sz * self.col)] + [0]).view(self.sz, self.col)
        self.x = self.sz - 1
        self.y = self.col - 1
        # set goal for check_solved
        self.goal = torch.tensor([i for i in range(1, self.sz * self.col)] + [0]).view(self.sz, self.col)

        # 如何逐个元素检查，虽然状态数少了，但是obs矩阵太稀疏
        # 另一方面，达成一行试错过程太长
        self.row_solved = 0
        self.shuffle()

    def reset(self):
        self.board = torch.tensor([i for i in range(1, self.sz * self.col)] + [0]).view(self.sz, self.col)
        self.x = self.sz - 1
        self.y = self.col - 1
        self.row_solved = 0
        self.shuffle()
        # obs: [16, 4, 4]
        return self.get_obs()

    def step(self, action):
        is_action_valid = self.move(action)
        next_obs = self.get_obs()
        done = self.check_solved()
        reward = self.get_reward(done, is_action_valid)
        return next_obs, reward, done, {}

    def get_obs(self, use_mask=False):
        if use_mask:
            mask = self.board > (self.row_solved + 1) * self.sz
            obs = F.one_hot(self.board)
            obs[mask] = torch.zeros_like(obs[mask])
        else:
            obs = F.one_hot(self.board)
        return obs.to(torch.float).permute(2, 0, 1)

    def get_reward(self, is_done, is_action_valid):
        # 曼哈顿距离越小，不一定离最优结果越近。
        # [8, 2, 3]
        # [4, 5, 6]
        # [x, x, x]
        if is_done:
            return self.sz * self.col * 30
        elif not is_action_valid:
            return - self.sz * self.col
        else:
            for i in range(self.sz-1, 0, -1):   # assume not done
                first_i_row = self.board[:i, :]
                target_i_row = torch.tensor([i for i in range(1, self.col*i + 1)]).view(-1, self.col)
                if torch.equal(first_i_row, target_i_row):
                    self.row_solved = i
                    return self.sz * i
            return -1

    def render(self):
        # os.system('cls' if os.name == 'nt' else ' clear')
        print(self.board)
        print(self.row_solved)
        # mask = self.board > (self.row_solved + 1) * self.sz
        # obs = F.one_hot(self.board)
        # obs[mask] = torch.zeros_like(obs[mask])
        # print(obs)

    def move(self, direction) -> bool:
        if direction == 0:      # left
            if self.y == 0:
                return False
            self.board[self.x][self.y], self.board[self.x][self.y-1] = self.board[self.x][self.y-1], 0
            self.y -= 1
        elif direction == 1:    # up
            if self.x == self.row_solved: # important
                return False
            self.board[self.x][self.y], self.board[self.x-1][self.y] = self.board[self.x - 1][self.y], 0
            self.x -= 1
        elif direction == 2:    # right
            if self.y == self.col - 1:
                return False
            self.board[self.x][self.y], self.board[self.x][self.y + 1] = self.board[self.x][self.y + 1], 0
            self.y += 1
        elif direction == 3:    # down
            if self.x == self.sz - 1:
                return False
            self.board[self.x][self.y], self.board[self.x + 1][self.y] = self.board[self.x + 1][self.y], 0
            self.x += 1
        return True

    def check_solved(self):
        if (self.board == self.goal).all():
            return True
        return False

    def shuffle(self, num=100):
        while self.check_solved():
            for _ in range(num):
                direction = random.choice([0, 1, 2, 3])
                self.move(direction)


if __name__ == "__main__":
    env = Puzzle(sz=3)
    while True:
        env.render()
        direction = input(">>")
        _, reward, _, _ = env.step(int(direction))
        print(f"reward: {reward}")
        if env.check_solved():
            break

