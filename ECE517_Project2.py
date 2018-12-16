# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 21:45:21 2018

@author: xuyan, huiliu
"""

import numpy as np
import sys
import random

d=int(sys.argv[1])
m=int(sys.argv[2])
n=int(sys.argv[3])
a=float(sys.argv[4])
g=float(sys.argv[5])
e=float(sys.argv[6])
    
r1=0
r2=0
r3=1
    
START = (0, 0)
GOAL = (d - 1, d - 1)

# possible actions
##action 0 indicates north, 1 for south, 2 for east and 3 for west
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_RIGHT = 2
ACTION_LEFT = 3

ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_RIGHT, ACTION_LEFT]

if m==-1 and d==8:
    trap=[(3,1),(6,1),(5,2),(2,3),(4,3),(7,3),(6,4),(3,5),(5,6),(6,6)]
    bestway=15
else:
    trap=[]
    while len(trap)<m:
        t=(random.randint(1,d-1),random.randint(1,d-1))
        if t not in trap:
            trap.append(t)    
        
terminal=trap.copy()
terminal.append((d-1,d-1))

def initialize_qvalue():
    q_value = np.zeros((d*d*4,1))
    q_value = q_value.reshape((d,d,4))
    return q_value

def initialize_policy():
    policy=np.zeros((d*d*4,1))
    policy+=1/4
    policy=policy.reshape((d,d,4))
    
    ##for terminal states, no action is needed
    for (i,j) in terminal:
        policy[i,j,:]=0
    return policy
    
def stimulate_episode(policy):
   i=0
   j=0
   episode=[]
   
   while (i,j) not in terminal:
       
       action=np.random.choice(4, 1, p=policy[i,j,:]).tolist()[0]
       episode.append((i,j,action))
       
       if action==0:
           i-=1
           if i <0:
               i=0
       if action==1:
           i+=1
           if i > 7:
               i=7
       if action==3:
           j-=1
           if j<0:
               j=0
       if action==2:
           j+=1
           if j>7:
               j=7
               
   episode.append((i,j,"No action"))
   return episode

##We use every-visit MC control method
def policy_iteration(epoch):
    q_value=initialize_qvalue()
    policy=initialize_policy()
    for k in range(epoch):
        #print(k)
        episode=stimulate_episode(policy)  
        #while (episode[-1][0],episode[-1][1]) not in terminal:
        #    episode=stimulate_episode(policy)
        G=0
        if (episode[-1][0],episode[-1][1]) in trap:
            r=r2
        else:
            r=r3
        b=0
        for (i,j,act) in episode[:-1][::-1]:
            if b==0:
                G=r
                b+=1
            else:
                G=round(g*G,6)+r1
            q_value[i,j,act]=round(q_value[i,j,act]+a*(G-q_value[i,j,act]),6)
            maxA = q_value[i,j,:].tolist()
            if maxA.count(max(maxA))==1:
                maxA=maxA.index(max(maxA))
                policy[i,j,:]=[1-e+e/4 if c==maxA else e/4 for c in range(4)]
            elif maxA.count(max(maxA))==3:
                maxA=maxA.index(min(maxA))
                policy[i,j,:]=[e/4 if c==maxA else (1-e/4)/3 for c in range(4)]
            
    return q_value, policy

def policy_test(policy):
    num_trap=0
    num_succeed=0
    best_way=0
    for trial in range(1000):
       i=0
       j=0
       episode=[]
       while (i,j) not in terminal:
       
           action=np.random.choice(4, 1, p=policy[i,j,:]).tolist()[0]
           episode.append((i,j))
           if action==0:
               i-=1
               if i <0:
                   i=0
           if action==1:
               i+=1
               if i > 7:
                   i=7
           if action==3:
               j-=1
               if j<0:
                   j=0
           if action==2:
               j+=1
               if j>7:
                   j=7
       episode.append((i,j))
       if (i,j) in trap:
           num_trap+=1
            
       elif len(episode)==bestway:
           best_way+=1
       else:
           num_succeed+=1
   
    return num_trap, num_succeed, best_way

##use n_step Sarsa method
def step(state, action):
    done = False
    reward = 0.0
    i, j = state

    if action == ACTION_UP:
        next_state = (max(i - 1, 0), j)
    elif action == ACTION_DOWN:
        next_state = (min(i + 1, d - 1), j)
    elif action == ACTION_LEFT:
        next_state = (i, max(j - 1, 0))
    elif action == ACTION_RIGHT:
        next_state = (i, min(j + 1, d - 1))
    if next_state in terminal:
        done = True
        if next_state == GOAL:
            reward += r3
        else: reward += r2
    else: reward += r1
    return next_state, reward, done


def make_policy(q_value, state):
    if np.random.binomial(1, e) == 1:
        action = np.random.choice(ACTIONS)
    else:
        values_ = q_value[state]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
    return action


def nStep_Sarsa_policy_iteration(n_step, num_episodes):
    q_value = initialize_qvalue()
    policy = initialize_policy()

    for k in range(num_episodes):
        #print (k)
        state = START
        action = make_policy(q_value, state)
        Rewards_episode = []
        States_episode = []
        Actions_episode = []
        States_episode.append(state)
        Actions_episode.append(action)
        T = float('inf')
        t = 0
        while(1):
            if t < T:
                state, reward, done = step(state, action)
                # next_action = make_policy(q_value, next_state)
                Rewards_episode.append(reward)
                States_episode.append(state)
                if done:
                    T = t + 1
                else:
                    action = make_policy(q_value, state)
                    Actions_episode.append(action)
            Tau = t - n_step + 1

            if Tau > 0 or Tau == 0:
                G = 0.0
                for i in range((Tau + 1), (min(Tau + n_step, T) + 1)):
                    G += g ** (i - Tau - 1) * Rewards_episode[i - 1]
                if (Tau + n_step) < T:
                    G += g**n_step * q_value[States_episode[Tau + n_step]][ Actions_episode[Tau + n_step]]
                q_value[States_episode[Tau]][ Actions_episode[Tau]] += a * (G - q_value[States_episode[Tau]][Actions_episode[Tau]])

            if Tau == T - 1: break
            t += 1

    for i in range(d):
        for j in range(d):
            temp = q_value[(i,j)].tolist()
            max_num = temp.count(max(temp))
            policy[(i,j)] = [(1 - e) / max_num + e / 4 if c == max(temp) else e / 4 for c in temp]

    return q_value, policy

def plot_policy(policy):
    graph_policy=np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            maxP = policy[i,j,:].tolist()
            if maxP.count(max(maxP))==1:
                graph_policy[i,j]=maxP.index(max(maxP))
            else:
                graph_policy[i,j]=4
    return graph_policy

def main():
    
    if n <= 16:
        qvalue,policy = nStep_Sarsa_policy_iteration(n,10000)
    else:
        qvalue,policy = policy_iteration(10000)
        
    t,s,b=policy_test(policy)
    test = [t,s,b]
    graph_policy=plot_policy(policy)
    np.savetxt("policy_test.txt",test,delimiter="\t")
    np.savetxt("policy_graph.txt",graph_policy,delimiter="\t")
if __name__ == '__main__':
    main()
