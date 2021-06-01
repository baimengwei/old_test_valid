import gym_bandits
import gym
import numpy as np
import math
env=gym.make('BanditTenArmedGaussian-v0')
num_rounds=20000

count=np.zeros(10)
sum_rewards=np.zeros(10)
Q=np.zeros(10)

def softmax(tau):
    total=sum([math.exp(val/tau) for val in Q])
    probs=[math.exp(val/tau)/total for val in Q]
    threshold=np.random.random()
    cummulative_prob=0.0
    for i in range(len(probs)):
        cummulative_prob+=probs[i]
        if cummulative_prob>threshold:
            return i
    return np.argmax(probs)


for i in range(num_rounds):
    arm=softmax(0.5)
    observation,reward,done,_ =env.step(arm)
    count[arm]+=1
    sum_rewards[arm]+=reward
    Q[arm]=sum_rewards[arm]/count[arm]
print('the optimal is ',np.argmax(Q))
print(Q)
#maybe it is meaningless




