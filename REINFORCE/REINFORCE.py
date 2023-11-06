import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.relu(self.l1(s))
        a_prob = F.softmax(self.l2(s), dim=1)
        return a_prob


class REINFORCE(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_width = 64
        self.lr = 4e-4
        self.GAMMA = 0.99  # discount factor
        self.episode_s, self.episode_a, self.episode_r = [], [], []

        self.policy = Policy(self.state_dim, self.action_dim, self.hidden_width)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        prob_weights = self.policy(s).detach().numpy().flatten()
        if deterministic:   # we use the deterministic policy during the evaluation
            a = np.argmax(prob_weights)
            return a
        else:
            a = np.random.choice(range(self.action_dim), p=prob_weights)  # 从动作空间中根据给定的概率权重prob_weights来随机选择一个动作
            return a

    def store(self, s, a, r):
        self.episode_s.append(s)
        self.episode_a.append(a)
        self.episode_r.append(r)

    def learn(self):
        G = []
        g = 0
        for r in reversed(self.episode_r):   # calculate the return G reversely
            g = self.GAMMA * g + r
            G.insert(0, g)

        for t in range(len(self.episode_r)):
            s = torch.unsqueeze(torch.tensor(self.episode_s[t], dtype=torch.float), 0)
            a = self.episode_a[t]
            g = G[t]

            a_prob = self.policy(s).flatten()
            policy_loss = -pow(self.GAMMA, t) * g * torch.log(a_prob[a])
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        # Clean the buffer
        self.episode_s, self.episode_a, self.episode_r = [], [], []


def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s, deterministic=True)
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)


if __name__ == '__main__':
    env_name = ['CartPole-v0', 'CartPole-v1']
    env_index = 1   # The index of the environments above
    env = gym.make(env_name[env_index])
    env_evaluate = gym.make(env_name[env_index])   # When evaluating the policy, we need to rebuild an environment
    number = 1
    seed = 0
    env.seed(seed)
    env_evaluate.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # max_episode_steps = env._max_episode_steps    # # Maximum number of steps per episode
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    # print("max_episode_steps={}".format(max_episode_steps))

    agent = REINFORCE(state_dim, action_dim)
    writer = SummaryWriter(log_dir='runs/REINFORCE/REINFORCE_env_{}_number_{}_seed_{}'.format(env_name[env_index], number, seed))  # build a tensorboard

    max_train_steps = 1e5   # 100k
    evaluate_freq = 1e3
    evaluate_num = 0
    evaluate_rewards = []
    total_steps = 0

    while total_steps < max_train_steps:
        episode_step = 0
        s = env.reset()
        done = False
        while not done:
            episode_step += 1
            a = agent.choose_action(s, deterministic=False)
            s_, r, done, _ = env.step(a)
            agent.store(s, a, r)
            s = s_

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                # evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(env_name[env_index]), evaluate_reward,
                                  global_step=total_steps)
                # if evaluate_num % 10 == 0:
                #     np.save(
                #         './data_train/REINFORCE_env_{}_number_{}_seed_{}.npy'.format(env_name[env_index], number, seed),
                #         np.array(evaluate_rewards))

            total_steps += 1

        # An episode is over, then update
        agent.learn()

