from agent import Agent
import numpy as np
import random
from collections import deque
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd
import os
import sys
import pickle as pk
from make_inputs import make_input_line

def take_action(agent, state, action, current_p, current_date):
    reward = 0  # sit
    valid = 1

    # hakf n half
    if action == 1:  # buy
        #share_num = (agent.cash / 2) // current_p
        share_num = len(agent.inventory) // 2
        if len(agent.inventory) <= 1:
            share_num = 1
        loop_num = share_num
        total_buy = (share_num * current_p)
        if agent.cash < total_buy or share_num == 0:
            valid = 0
        else:
            while loop_num > 0:  # half n half strategy
                agent.inventory += [current_p]
                agent.cash -= current_p
                loop_num -= 1
            trans_cost = total_buy * 0.0025  # transaction cost
            agent.cash -= trans_cost


    elif action == 2:  # sell
        if len(agent.inventory) == 0:
            valid = 0
        else:
            share_num = len(agent.inventory) // 2
            if len(agent.inventory) == 1:
                share_num = 1
            loop_num = share_num
            while loop_num > 0:  # half n half strategy
                reward += (current_p - agent.inventory[loop_num - 1]) - (current_p * 0.0025)
                agent.cash += current_p * 0.9975
                loop_num -= 1
            reward = reward / sum(agent.inventory[:share_num])
            reward = reward + np.sign(reward)
            agent.inventory = agent.inventory[share_num:]

    if valid == 0:
        agent.action_vec += ["sit"]
    else:
        agent.action_vec += [["sit", "buy", "sell"][action]]

    # keep trade results in agent's memory
    agent.inven_vec += [sum(agent.inventory)]
    agent.cash_vec += [agent.cash]
    agent.time_vec += [current_date]
    agent.profit_vec += [len(agent.inventory) * current_p + agent.cash - agent.principal]
    agent.price_vec += [current_p]
    return (state, action, reward, valid)

def update(agent, state):
    (prev_state, prev_action, prev_reward, prev_valid) = agent.prev
    agent.memory.append((prev_state, prev_action, state, prev_reward, 0))
    if len(agent.memory) > agent.batchLen and prev_valid == 1:
        agent.update()  # refer to agent.py
        if agent.epsilon > agent.minEpsilon:  # epsilon decay
            agent.epsilon *= agent.decay
    if len(agent.memory) < agent.batchLen:
        print("not enough memory to update yet (memory : " + str(len(agent.memory)) + ")")

def reset(agent, principal):
    agent.principal = agent.cash = principal
    agent.inventory = []
    agent.time_vec = []
    agent.action_vec = []
    agent.inven_vec = []
    agent.cash_vec = []
    agent.profit_vec = []
    agent.price_vec = []
    return agent

# data : ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'BollingerB_20',
#         'Bollinger%b_20', 'RSI_14', 'OBV_10']
# Q-Learning
# Here we train a model that can determine an action (buy, sell, hold)
# Decision depends on recent "window" day/hours of technical indicator data
# and trained DNN (previous experience).
def qlearn(data, window, principal, batch_len, train_len, indicators, trial_num, trial_name):

    train_data = data[:train_len]
    test_data = data[train_len:]

    # input name for test trial

    # create agent
    agent = Agent(window, principal, batch_len)

    # train model
    print("training on "+str(len(train_data))+" instances")
    for row in range(window, len(train_data)):
        # makes 1x30 matrix
        line = make_input_line(train_data, row, window, indicators)
        line = line.reshape(1,-1)
        action = agent.act(line)
        current_p = train_data.iloc[row]['Close']
        current_date = train_data.iloc[row]['date']
        (state, action, reward, valid) = take_action(agent, line, action, current_p, current_date)
        print(str(current_date)+" : "+str(["sit", "buy", "sell"][action]))
        if agent.prev != None:
            update(agent, state)
        agent.prev = (state, action, reward, valid)
    print("train has ended")

    # Save train results
    print("Rendering result in csv file")
    d = {'Time': agent.time_vec,
         'Price': agent.price_vec,
         'Action': agent.action_vec,
         'Inventory': agent.inven_vec,
         'Cash': agent.cash_vec,
         'Profit': agent.profit_vec}
    result_df = pd.DataFrame(data=d)
    result_df.to_csv(trial_name+ "_" + trial_num +'_train.csv', index=False)

    # reset agent parameters
    agent = reset(agent, principal)

    # test model
    print("testing on " + str(len(test_data)) + " instances")
    for row in range(len(train_data), len(data)):
        # makes 1x30 matrix
        line = make_input_line(data, row, window, indicators)
        line = line.reshape(1, -1)
        action = agent.act(line)
        current_p = data.iloc[row]['Close']
        current_date = data.iloc[row]['date']
        (state, action, reward, valid) = take_action(agent, line, action, current_p, current_date)
        print(str(current_date) + " : " + str(["sit", "buy", "sell"][action]))
        if agent.prev != None:
            update(agent, state)
        agent.prev = (state, action, reward, valid)
    print("test has ended")

    # Save train results
    print("Rendering result in csv file")
    d = {'Time': agent.time_vec,
         'Price': agent.price_vec,
         'Action': agent.action_vec,
         'Inventory': agent.inven_vec,
         'Cash': agent.cash_vec,
         'Profit': agent.profit_vec}
    result_df = pd.DataFrame(data=d)
    result_df.to_csv(trial_name + "_" + trial_num + '_test.csv', index=False)
    return









