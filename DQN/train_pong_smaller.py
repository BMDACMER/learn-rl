import numpy as np
import gym
from utils import *
from agent import *
from config import *

def train(env, agent, num_episode, eps_init, eps_decay, eps_min, max_t, num_frame=2, constant=0):
    rewards_log = []
    average_log = []
    eps = eps_init

    for i in range(1, 1 + num_episode):

        episodic_reward = 0
        done = False
        frame = env.reset()
        frame = preprocess(frame, constant)
        state_deque = deque(maxlen=num_frame)
        for _ in range(num_frame):
            state_deque.append(frame)
        state = np.stack(state_deque, axis=0)
        state = np.expand_dims(state, axis=0)
        t = 0

        while not done and t < max_t:

            t += 1
            action = agent.act(state, eps)
            frame, reward, done, _ = env.step(action + 1)
            frame = preprocess(frame, constant)
            state_deque.append(frame)
            next_state = np.stack(state_deque, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            agent.memory.append((state, action, reward, next_state, done))

            if t % 5 == 0 and len(agent.memory) >= agent.bs:
                agent.learn()
                agent.soft_update(agent.tau)

            state = next_state.copy()
            episodic_reward += reward

        rewards_log.append(episodic_reward)
        average_log.append(np.mean(rewards_log[-100:]))
        print('\rEpisode {}, Reward {:.3f}, Average Reward {:.3f}'.format(i, episodic_reward, average_log[-1]), end='')
        if i % 100 == 0:
            print()

        eps = max(eps * eps_decay, eps_min)

    return rewards_log


if __name__ == '__main__':
    env = gym.make(VISUAL_ENV_NAME)
    agent = Agent(NUM_FRAME, 3, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE, True)
    rewards_log = train(env, agent, VISUAL_NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T, NUM_FRAME, CONSTANT)
    np.save('{}_smaller_rewards.npy'.format(VISUAL_ENV_NAME), rewards_log)
    agent.Q_local.to('cpu')
    torch.save(agent.Q_local.state_dict(), '{}_smaller_weights.pth'.format(VISUAL_ENV_NAME))
