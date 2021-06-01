import gym
import random
env=gym.make('Taxi-v2')
env.render()
alpha=0.4
gamma=0.999
epsilon=0.017
q={}

"""
https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
"""

for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q[(s,a)]=0.0

    
    
def epsilon_greedy_policy(state,epsilon):
    if random.uniform(0,1)<epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)),key=lambda x: q[(state,x)])

for i in range(800):
    r=0
    state=env.reset()
    env.render()
    action=epsilon_greedy_policy(state, epsilon)
    while True:
        next_state,reward,done,_ =env.step(action)
        next_action=epsilon_greedy_policy(next_state, epsilon)
        env.render()
        q[(state,action)]+=alpha*(reward+gamma*q[(next_state,next_action)]-q[(state,action)])
        state=next_state
        action=next_action
        r+=reward

        if done:
            print('-------------------')
            break
    print('total reward:',r)
    
    
    
    
env.close()


