from collections import defaultdict
import random
import numpy as np
import torch
from copy import deepcopy

from Sprite import *
import torch.nn as nn
import torch.optim as optim


class Agent:
    def __init__(self, s: tuple, epsilon_flag=True) -> None:
        super().__init__()
        self.s = s
        self.Q = defaultdict(lambda: [0.0, 0.0])
        self.alpha = 0.6
        self.gama = 0.8
        self.epsilon = 0.0
        self.epsilon_decay = 1.0
        self.action = None
        self.epsilon_flag = epsilon_flag

    def choose_action(self):
        if self.epsilon_flag:
            if random.random() <= self.epsilon:
                self.action = random.randint(0, 1)
                return self.action
        self.action = np.argmax(self.Q[self.s])
        return self.action

    def update_q(self, s_: tuple, r):
        self.Q[self.s][self.action] += self.alpha * (r + self.gama * np.max(self.Q[s_]) - self.Q[self.s][self.action])
        self.s = s_

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay


def relu(x: np.array):
    x[x < 0] = 0.0
    return x


class BirdBrain:
    def __init__(self, n) -> None:
        super().__init__()
        self.n = n
        self.w1 = np.random.random((3, n))
        self.w2 = np.random.random((n + 1, 2))

    def forward(self, x: np.array):
        x = np.append(x, 1.0)
        tmp_x = np.dot(x, self.w1)
        # 激活
        tmp_x = relu(tmp_x)
        tmp_x = np.append(tmp_x, 1.0)
        return np.argmax(np.dot(tmp_x, self.w2))

    def output_weights(self):
        return np.concatenate((self.w1.reshape(3 * self.n), self.w2.reshape((self.n + 1) * 2)))

    def input_weights(self, x: np.array):
        self.w1 = x[0:3 * self.n].reshape((3, self.n))
        self.w2 = x[3 * self.n:].reshape((self.n + 1, 2))


class BirdBrain_Pytorch(nn.Module):
    def __init__(self, n) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(2, n)
        self.relu = nn.ReLU(inplace=True)
        self.linear_2 = nn.Linear(n, 2)

    def forward(self, x):
        return self.linear_2(self.relu(self.linear_1(x)))


class DeepQNetwork:
    def __init__(
            self,
            n_features=2,
            learning_rate=1e-3,
            reward_decay=0.99,
            e_greedy=0.0,
            replace_target_iter=300,
            memory_size=50000,
            batch_size=512,
            net_width=10,
            warm_up_iteration=1000,
            copy_model_iteration=300,
            epsilon_iteration=10000
    ):
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = e_greedy
        self.epsilon_iteration = epsilon_iteration
        self.memory_counter = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # total learning step
        self.learn_step_counter = 0
        self.warm_up_iteration = warm_up_iteration
        self.copy_model_iteration = copy_model_iteration

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.net = BirdBrain_Pytorch(net_width).to(self.device)
        self.target_net = BirdBrain_Pytorch(net_width).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def save_memory(self, s, a, r, s_):
        tmp = np.hstack((s, [a, r], s_))
        self.memory[self.memory_counter % self.memory_size, :] = tmp
        self.memory_counter += 1

    def choose_action(self, s):
        if np.random.rand() > self.epsilon:
            s = s.astype('float32')
            s = torch.tensor(s, device=self.device, requires_grad=False)
            q = self.net(s).cpu().detach().numpy()
            action = np.argmax(q)
        else:
            action = np.random.randint(0, 1)
        return action

    def learn(self):
        # sample batch memory from all memory
        if self.learn_step_counter % self.copy_model_iteration == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        if self.learn_step_counter < self.warm_up_iteration:
            self.learn_step_counter += 1
            return -1.
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :].astype('float32')
        batch_memory_tensor = torch.tensor(batch_memory, device=self.device, requires_grad=True)

        q_next = self.target_net(batch_memory_tensor[:, -self.n_features:]).cpu().detach().numpy()
        q_eval = self.net(batch_memory_tensor[:, 0:self.n_features])

        q_target = q_eval.cpu().detach().numpy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        q_target = torch.tensor(q_target, device=self.device, requires_grad=True)

        loss = self.criterion(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter > 0 and self.learn_step_counter % self.epsilon_iteration == 0:
            self.epsilon /= np.sqrt(10)

        return loss.item()


def calculate_state(bird: BirdSprite, tubes: list, scale_factor=10):
    for i in range(len(tubes)):
        if tubes[i][0].score_flag is False:
            return (bird.rect.x - tubes[i][1].rect.x) // scale_factor, (
                    bird.rect.y - tubes[i][1].rect.y) // scale_factor


def draw_epoch(images, score, screen, window_width, window_height):
    number_list = [int(x) for x in str(score)]
    selected_images = []
    for i in range(len(number_list)):
        selected_images.append(images[number_list[i]])
    w = 0
    h = 0
    for i in range(len(selected_images)):
        w += selected_images[i].get_width()
        h = max(h, selected_images[i].get_height())
    x = (window_width - w) // 2
    y = window_height - window_height // 10 - h
    for i in range(len(selected_images)):
        if i == 0:
            screen.blit(selected_images[i], pygame.Rect(x, y, selected_images[i].get_width(), h))
        else:
            x += selected_images[i - 1].get_width()
            screen.blit(selected_images[i],
                        pygame.Rect(x, y, selected_images[i].get_width(), h))
