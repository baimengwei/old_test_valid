import gym_bandits
import gym
import numpy as np
env=gym.make('BanditTenArmedGaussian-v0')
num_rounds=20000

count=np.zeros(10)
sum_rewards=np.zeros(10)
Q=np.zeros(10)

def epsilon_greedy(epsilon):
    if np.random.random()<epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q)

for i in range(num_rounds):
    arm=epsilon_greedy(0.5)
    observation,reward,done,_ =env.step(arm)
    count[arm]+=1
    sum_rewards[arm]+=reward
    Q[arm]=sum_rewards[arm]/count[arm]
print('the optimal is ',np.argmax(Q))
print(Q)
#maybe it is meaningless




