import gym_bandits
import gym
import numpy as np
env=gym.make('BanditTenArmedGaussian-v0')
num_rounds=20000

count=np.zeros(10)
sum_rewards=np.zeros(10)
Q=np.zeros(10)
alpha=np.ones(10)
beta=np.ones(10)

def thompson_sampleing(alpha,beta):
    samples=[np.random.beta(alpha[i]+1,beta[i]+1) for i in range(10)]
    return np.argmax(samples)



for i in range(num_rounds):
    arm=thompson_sampleing(alpha,beta)
    observation,reward,done,_ =env.step(arm)
    count[arm]+=1
    sum_rewards[arm]+=reward
    Q[arm]=sum_rewards[arm]/count[arm]
    if reward>0:
        alpha[arm]+=1
    else:
        beta[arm]+=1
print('the optimal is ',np.argmax(Q))
print(Q)
#maybe it is meaningless




