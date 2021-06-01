
import gym
import numpy as np
env=gym.make('FrozenLake-v0')
print(env.observation_space.n)# the number of state
print(env.action_space.n)
#value_table=np.zeros(env.observation_space.n)
#no_of_iterations=100000
#
#for i in range(no_of_iterations):
#    updated_value_table=np.copy(value_table)


def value_iteration(env,gamma=1.0):
    value_table=np.zeros(env.observation_space.n)
    
    no_of_iterations=100000
    threshold=1e-20
    for i in range(no_of_iterations):
        updated_value_table=np.copy(value_table)
        for state in range(env.observation_space.n):#for each state, compute the Q_value and find the max to update
            Q_value=[]
            for action in range(env.action_space.n):
                next_states_rewards=[]
                envP=env.P#all message about this environment is in it
                for next_sr in envP[state][action]:#for each state and action, the probably to next state is a probably problem.
                    trans_prob,next_state,reward_prob,_=next_sr# P  S  R  _   next state reward
                    next_states_rewards.append((trans_prob*(reward_prob+gamma*updated_value_table[next_state])))
                    Q_value.append(np.sum(next_states_rewards))
                    value_table[state]=max(Q_value)
                    
        if np.sum(np.fabs(updated_value_table-value_table))<=threshold :# ending condition
            print('converaged at '+str(i+1))
            break
        
    return value_table

def extract_policy(value_table,gamma=1.0):
    policy=np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table=np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob,next_state,reward_prob,_=next_sr
                Q_table[action]+=(trans_prob*(reward_prob+gamma*value_table[next_state]))
        #compute each state action value, find and choose the action which gain max value.
        policy[state]=np.argmax(Q_table)
    return policy


optimal_value_function=value_iteration(env=env,gamma=1.0)
print(optimal_value_function)
optimal_policy=extract_policy(optimal_value_function,gamma=1.0)
print(optimal_policy)



observation=env.reset()
for t in range(100):
    print('-----',observation)
    action=int(optimal_policy[observation])
    print('---action:',action)
    env.render()
    observation,reward,done,info=env.step(action)
    if done:
        print('{} timesteps taken for the episode'.format(t+1))
        env.render()
        print(observation)
        print('----------------------------------------------')
        break
#note that state and action CAN'T decide the next state
    
    
    
    