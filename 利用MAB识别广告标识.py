import gym_bandits
import gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.DataFrame()
df['Banner_type_0']=np.random.randint(0,2,100000)
df['Banner_type_1']=np.random.randint(0,2,100000)
df['Banner_type_2']=np.random.randint(0,2,100000)
df['Banner_type_3']=np.random.randint(0,2,100000)
df['Banner_type_4']=np.random.randint(0,2,100000)
print(df.head(5))#what do guests choose

num_rounds=100000
num_banner=5
banner_selected=[]
count=np.zeros(num_banner)
sum_rewards=np.zeros(num_banner)
Q=np.zeros(num_banner)

def epsilon_greedy(epsilon):
    if np.random.random()<epsilon:
        return np.random.choice(num_banner)
    else:
        return np.argmax(Q)

for i in range(num_rounds):
    banner=epsilon_greedy(0.5)
    reward=df.values[i,banner]
    count[banner]+=1
    sum_rewards[banner]+=reward
    Q[banner]=sum_rewards[banner]/count[banner]
    banner_selected.append(banner)
sns.distplot(banner_selected)    
print(len(banner_selected))
print('the optimal is ',np.argmax(Q))
print(Q)
plt.show()
#maybe it is meaningless




