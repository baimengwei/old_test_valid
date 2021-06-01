import gym
import random
import matplotlib.pyplot as plt
from gym import envs

from gym import envs
print(envs.registry.all())

env = gym.make('Taxi-v3')
env.render()
alpha = 0.4
gamma = 0.999
epsilon = 0.017
q = {}

"""
https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
"""


def update_q_table(prev_state, action, reward, nextstate, alpha, gamma):
    qa = max([q[(nextstate, a)] for a in range(env.action_space.n)])
    q[(prev_state, action)] += alpha * (reward + gamma * qa - q[(prev_state, action)])


def epsilon_greedy_policy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key=lambda x: q[(state, x)])


q_reward_list = []
cnt = 0
reward_avg = 0
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q[(s, a)] = 0.0
for i in range(500):
    r = 0
    prev_state = env.reset()
    env.render()
    while True:
        action = epsilon_greedy_policy(prev_state, epsilon)
        next_state, reward, done, _ = env.step(action)
        # env.render()
        update_q_table(prev_state, action, reward, next_state, alpha, gamma)
        prev_state = next_state
        r += reward

        if done:
            print(i, '-------------------', r)
            cnt += 1
            reward_avg += r
            if cnt % 10 == 0:
                q_reward_list.append(reward_avg )
                reward_avg = 0
            break
    print('total reward:', r)

# -----------------------------------------------------------

rq_reward_list = []
cnt = 0
reward_avg = 0
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q[(s, a)] = 0.0
for i in range(500):
    r = 0
    prev_state = env.reset()
    env.render()
    while True:
        action = epsilon_greedy_policy(prev_state, epsilon)
        next_state, reward, done, _ = env.step(action)
        # env.render()
        if reward > 0:
            update_q_table(prev_state, action, reward + 10, next_state, alpha+0.5, gamma+0.2)
        else:
            update_q_table(prev_state, action, reward, next_state, alpha, gamma)
        prev_state = next_state
        r += reward

        if done:
            print(i, '-------------------', r)
            cnt += 1
            reward_avg += r
            if cnt % 10 == 0:
                rq_reward_list.append(reward_avg)
                reward_avg = 0
            break
    print('total reward:', r)

env.close()
plt.plot(q_reward_list, label='q-learning', color='g', linestyle='--')
plt.plot(rq_reward_list, label='rq-learning', color='b', linestyle='-')
plt.show()

# plt.plot(x,list1,label='list1',color='g',linewidth=2,linestyle=':')
# plt.plot(x,list2,label='list2',color='b',linewidth=5,linestyle='--')
