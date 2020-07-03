import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque


def make_model(in_dim,l1,l2,act_size,learn):
    model = Sequential()
    model.add(Dense(units=l1, input_dim=in_dim, activation="relu"))
    model.add(Dense(units=l2, activation="relu"))
    model.add(Dense(act_size, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=learn))

    return model

class Agent:
    def __init__(self,window,principal,batchLen):
        if window > 30: raise Error("Invalid size of window (must be under 30)")
        # {sit = 0, buy = (10 ~ 100%), sell = (10 ~ 100%)}
        
        # input parameter
        self.stateSpace = window
        self.cash = principal
        self.evalCash = principal
        self.batchLen = batchLen
        self.prev = ()
        
        # hyper parameter
        self.g = 0.95 # discount rate
        self.epsilon = 1 # greedy-epsilon
        self.minEpsilon = 0.01
        self.decay = 0.995 # epsilon will decay overtime
        self.memory = deque(maxlen=1000) # for exp replay
        self.model =  make_model(30,16,16,3,0.01) # Qvalue NN approximator
        
        # constant parameter
        self.actionSpace = 3 # {sit = 0, buy = 1, sell = 2} for now...
        self.inventory = []


    def act(self,state):
        # exploration
        if random.random() <= self.epsilon:
            act = random.randrange(self.actionSpace)
        # exploitation
        else:
            act = np.argmax(self.model.predict(state))

        assert act == 0 or act == 1 or act == 2

        return act
    
    # experience replay based update in Q-value NN approximator
    def update(self):
        memoryLen = len(self.memory)
        batchLen = self.batchLen
        batch = [self.memory[i] for i in range(memoryLen-batchLen,memoryLen)]
        # SGD performed here
        for (state,action,nextState,reward,done) in batch:
            # next data exists so expected future discounted reward is added
            if done == 0:
                Qprime = reward+self.g*np.amax(self.model.predict(nextState)[0])
            # last data so only current reward
            else: Qprime = reward
            currentQvec = (self.model.predict(state))
            (currentQvec[0])[action] = Qprime
            # fit X = state, Y = discounted future rewards for each act
            self.model.fit(state,currentQvec, epochs = 1, verbose = 0)
            
    
        