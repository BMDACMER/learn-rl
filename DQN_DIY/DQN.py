import random

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import collections
import argparse
from config import add_dqn_args
import time
import matplotlib.pyplot as plt


class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_width, action_dim):
        super(Qnet, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = self.l2(x)
        return x


class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class DQN(object):
    def __init__(self, args):
        self.args = args

        self.q_net = Qnet(args.state_dim, args.hidden_dim, args.action_dim)
        self.target_q_net = Qnet(args.state_dim, args.hidden_dim, args.action_dim)
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.count = 0  # 计数器,记录更新次数

    def choose_action(self, state):
        if np.random.random() < self.args.epsilon:
            action = np.random.randint(self.args.action_dim)
        else:
            # state = torch.tensor([state], dtype=torch.float)
            # 上面一句会出现警告 所以换成下面这句
            state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
            action = self.q_net(state).argmax().item()
        return action

    def learn(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.args.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.q_net_optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.q_net_optimizer.step()

        if self.count % self.args.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1


if __name__ == '__main__':
    env_name = ['CartPole-v0', 'CartPole-v1']
    env_index = 1
    env = gym.make(env_name[env_index])
    env_evaluate = gym.make(env_name[env_index])  # When evaluating the policy, we need to rebuild an environment
    number = 9
    # Set random seed
    seed = 0
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_episode_steps={}".format(max_episode_steps))

    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser = add_dqn_args(parser)
    args = parser.parse_args()
    args.state_dim = state_dim
    args.action_dim = action_dim

    replay_buffer = ReplayBuffer(args.buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(args)
    writer = SummaryWriter(log_dir='runs/DQN/DQN_env_{}_number_{}_seed_{}'.format(env_name[env_index], number,
                                                                                      seed))  # Build a tensorboard
    total_steps = 0
    evaluate_num = 0
    return_list = []

    while total_steps < args.num_episode:
        s = env.reset()
        done = False
        episode_return = 0
        while not done:
            a = agent.choose_action(s)
            s_, r, done, _ = env.step(a)

            replay_buffer.add(s, a, r, s_, done)
            episode_return += r
            s = s_
            # 当buffer数据的数量超过一定值后,才进行Q网络训练
            if replay_buffer.size() > args.minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(args.batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                agent.learn(transition_dict)
        # return_list.append(episode_return)
        if (total_steps + 1) % args.evaluate_freq == 0:
            evaluate_num += 1
            print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, episode_return))
            writer.add_scalar('step_rewards_{}'.format(env_name[env_index]), episode_return,
                              global_step=total_steps)
        total_steps += 1

