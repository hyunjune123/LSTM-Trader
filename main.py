from environment import Agent
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
import make_inputs

def main(argv, principal):
    # 설정해야할 paramters
    window = 10     # 몇시간 이내 data로 training 할건가
    batch_len = 32

    # argv[1] 가 1 row csv 라고 생각하고
    # na 있으면 ㅈ댐, 형식은 밑에 처럼
    # input : ['Time', 'Price', 'High', 'Low', 'Volume']
    new_line = (pd.read_csv(argv))
#############################################################################################
# 키움금융에서 받아온 위 5가지 data 로 기술적 분석 지표들을 생성해야함
# 20줄은 있어야 지표들을 계산할 수 있음 (예. 20 period이동평균선)
# df_mem 에 20 줄이 채워지면 다음 단계로 이동함

    # row 들을 pandas dataframe 으로 저장해두자, 이름은 df_mem
    if os.path.exists("df_mem.pkl"):
        df_mem = pd.read_pickle("./df_mem.pkl")
        df_mem = df_mem.append(new_line)
        # 너무 커질까바
        if (len(df_mem) > 10000):
            df_mem = df_mem[1:]
    else:     # df_mem 존재 안할경우 (첫 row input)
        df_mem = new_line
    df_mem.reset_index()
    df_mem.columns = ['Time', 'Price', 'High', 'Low', 'Volume']
    df_mem.to_pickle("df_mem.pkl")

    # current_df 가 20줄 이상은 되야 첫번째 input 계산이 가능함 (ex. 20 period 이동평균선)
    if len(df_mem) < 20:
        print("need " + str(20 - len(df_mem)) + " more inputs")
        return ("need "+str(20-len(df_mem))+" more inputs")

#############################################################################################
# df_mem 써서 현재 지표 계산 후 df_inputs에 저장
# 이 지표 df_mem 은 10줄 이상 있어야 다음 단계로 이동 가능 (지난 10 period 의 지표들을 보고 판단)
# 물론 10 period 은 위 window parameter 로 조정가능

    # input 한줄 계산 (현재를 포함한 20시간 이내 데이타들로)
    new_input_line = make_inputs.make_input_line(df_mem)

    # input row 들을 pandas df 로 저장해두자, 이름은 df_inputs
    if os.path.exists("df_inputs.pkl"):
        df_inputs = pd.read_pickle("./df_inputs.pkl")
        df_inputs = df_inputs.append(new_input_line)
        # 얘는 항상 20줄만 있음 댐
        if (len(df_inputs) > 20):
            df_mem = df_mem[1:]
    else:
        df_inputs = new_input_line
    df_inputs.reset_index(drop=True)
    df_inputs.columns = ['Time', 'Price', 'Bolinger', 'Rsi', 'Volume']
    df_inputs.to_pickle("df_inputs.pkl")

    # df_inputs 는 window 만큼 줄이 있어야 training 가능
    if len(df_inputs) < window:
        print("have only " + str(len(df_inputs)) + " input lines, feed more")
        return ("have only "+str(len(df_inputs))+" input lines")


############################################################################################
# Q-Learning
# 현재 시장 data를 기준으로 NN 을 사용해 action 을 결정.
# 다음 timing에 시장의 변화에따라 NN 을 적응 시킴 (update)
# 시장 상황에 더욱 적합한 action 을 predict 하게됨.
# 반복

    # agent 가 없을경우 하나 생성하고 agent 를 저장
    state = make_inputs.make_train_line(df_inputs).reshape(1,-1)
    if (not (os.path.exists("agent.pkl"))):  # 첫 training / agent initialize
        agent = Agent(window, principal, batch_len)
        with open('agent.pkl', 'wb') as output:
            pk.dump(agent, output, pk.HIGHEST_PROTOCOL)

    with open('agent.pkl','rb') as input:
        agent = pk.load(input)
    action = agent.act(state)
    current_p = float(new_line["Price"])

    reward = 0     # sit
    valid = 1

    # 현재 전략 : 사야할땐 1 주씩 사고 팔아야할땐 다 팔기
    if action == 1:     # buy
        share_num = len(agent.inventory) // 2
        if len(agent.inventory) <= 1:
            share_num = 1
        loop_num = share_num
        total_buy = (share_num * current_p)
        if agent.cash < total_buy:
            valid = 0
        else:
            while loop_num > 0:    # half n half strategy
                agent.inventory += [current_p]
                agent.cash -= current_p
                loop_num -= 1
            trans_cost = total_buy * 0.00015  # 매수 수수료
            agent.cash -= trans_cost


    elif action == 2:    # sell
        if len(agent.inventory) == 0:
            valid = 0
        else:
            share_num = len(agent.inventory)//2
            if len(agent.inventory) == 1:
                share_num = 1
            loop_num = share_num
            while loop_num > 0:     # half n half strategy
                reward += (current_p - agent.inventory[loop_num-1]) - (current_p * 0.00265)
                agent.cash += current_p * 0.99735    # 수수료 / 매도료 0.265% 땐만큼만 cash 추가
                loop_num -= 1
            reward = reward / sum(agent.inventory[:share_num])
            reward = reward + np.sign(reward)
            agent.inventory = agent.inventory[share_num:]

    agent.prev = (state, action, reward)

    if len(df_inputs) == 10:     # 가장 첫번째 input은 학습이 불가능
        print("Learning will start from now on")
        return "sit"
    else:     # 강화학습
        (prev_state, prev_action, prev_reward) = agent.prev
        agent.memory.append((prev_state, prev_action, state, prev_reward, 0))
        if len(agent.memory) > agent.batchLen and valid == 1:
            agent.update()     # environment.py 참고
            print("agent is updated")
            if agent.epsilon > agent.minEpsilon:     # epsilon decay
                agent.epsilon *= agent.decay
        if len(agent.memory) < agent.batchLen:
            print("not enough memory to update yet (memory : "+str(len(agent.memory))+")")

    with open('agent.pkl', 'wb') as output:     # agent 저장
        pk.dump(agent, output, pk.HIGHEST_PROTOCOL)
    if valid == 0: action = 0
    return action

if __name__ == "__main__":
    #main(sys.argv[1], sys.argv[2])

    train_index = [1500, ]
    test_index = [1400, 0]

    # train
    principal = 5000000
    print("start training")
    sk_train = pd.read_csv("sk_train.csv")
    for row in range(train_index[0],train_index[1],-1):
        print("current train number is "+str(row))
        sk_train_line = sk_train.iloc[[row]]
        sk_train_line.to_csv('sk_train_line.csv', index= False)
        main('sk_train_line.csv', principal)

    # test
    print("start testing")
    sk_test = sk_train.iloc[test_index[1]:test_index[0]]
    #sk_test = pd.read_csv("sk_test.csv")
    price_vec = sk_test["Price"].tolist()
    list.reverse(price_vec)
    time_vec = sk_test["Time"].tolist()
    list.reverse(time_vec)
    action_vec = []
    inven_vec = []
    cash_vec = []
    profit_vec = []

    with open('agent.pkl','rb') as input:
        agent = pk.load(input)
    agent.cash = 5000000
    agent.inventory = []
    with open('agent.pkl', 'wb') as output:     # agent 저장
        pk.dump(agent, output, pk.HIGHEST_PROTOCOL)

    for row in range(len(sk_test)-1,-1,-1):
        print("current test number is " + str(row))
        sk_test_line = sk_test.iloc[[row]]
        sk_test_line.to_csv('sk_test_line.csv', index=False)
        action_vec += [main('sk_test_line.csv',principal)]
        with open('agent.pkl', 'rb') as input:
            agent = pk.load(input)
        inven_vec += [sum(agent.inventory)]
        cash_vec += [agent.cash]
        price = sk_test_line["Price"]
        profit_vec += [(price * len(agent.inventory)) + agent.cash - 5000000]
        with open('agent.pkl', 'wb') as output:  # agent 저장
            pk.dump(agent, output, pk.HIGHEST_PROTOCOL)

    d = {'Time': time_vec,
         'Price': price_vec,
         'Action': action_vec,
         'Inventory': inven_vec,
         'Cash': cash_vec,
         'Profit' : profit_vec}

    result_df = pd.DataFrame(data = d)

    result_df.to_csv('test_result.csv', index=False)








