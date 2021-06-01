import gym
import numpy
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from functools import partial
from gym.envs.toy_text.blackjack import usable_ace
plt.style.use('ggplot')
env=gym.make('Blackjack-v0')

def sample_policy(observation):
    score,dealer_score,usable_ace=observation
    return 0 if score>=20 else 1# 0 stop , 1 get

def generate_episode(policy,env):#using sample_policy as policy and this parameter doesn't use
    states,actions,rewards=[],[],[]
    observation=env.reset()
    while True:
        states.append(observation)
        
        action=sample_policy(observation)
        actions.append(action)   
        
        observation,reward,done,info=env.step(action)
        rewards.append(reward)
        
        if done:
            break
    return states,actions,rewards


def first_vist_mc_prediction(policy,env,n_episodes):
    value_table=defaultdict(float)
    N=defaultdict(int)
    for _ in range(n_episodes):
        states,_,rewards=generate_episode(policy, env)
        returns=0
        for t in range(len(states)-1,-1,-1):#start end=-1 step=-1
            R=rewards[t]
            S=states[t]
            returns +=R
            if S not in states[:t]:#but why i can't find an instance which doesn't go into it
                N[S]+=1
                value_table[S]+=(returns-value_table[S])/N[S]
    print(N)
    return value_table

def plot_blackjack(V,ax1,ax2):
    player_sum=numpy.arange(12,21+1)
    dealer_show=numpy.arange(1,10+1)
    usable_ace=numpy.array([False,True])
    
    state_values=numpy.zeros((len(player_sum),len(dealer_show),len(usable_ace)))
    for i,player in enumerate(player_sum):# enumerate has been included index and value
        for j,dealer in enumerate(dealer_show):
            for k,ace in enumerate(usable_ace):
                state_values[i,j,k]=V[player,dealer,ace]
    
    X,Y=numpy.meshgrid(player_sum,dealer_show)
    ax1.plot_wireframe(X,Y,state_values[:,:,0])
    ax2.plot_wireframe(X,Y,state_values[:,:,1])
    
    for ax in ax1,ax2:
        ax.set_zlim(-1,1)
        ax.set_ylabel('player sum')
        ax.set_xlabel('dealer showing')
        ax.set_zlabel('state-value')


value=first_vist_mc_prediction(sample_policy, env, n_episodes=50000)

print(value)


fig,axes=pyplot.subplots(nrows=2,figsize=(5,8),subplot_kw={'projection':'3d'})

axes[0].set_title('value function without usable ace')
axes[1].set_title('value function with usable ace')
plot_blackjack(value,axes[0],axes[1])
pyplot.show()







