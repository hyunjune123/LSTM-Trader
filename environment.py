from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize 
import numpy as np
import random
from collections import deque 

class Agent:
    def __init__(self,window,principal,batchLen):
        if window > 30: raise Error("Invalid size of window (must be under 30)")
        # {sit = 0, buy = (10 ~ 100%), sell = (10 ~ 100%)}
        
        # input parameter
        self.stateSpace = window
        self.cash = principal
        self.evalCash = principal
        self.batchLen = batchLen
        
        # hyper parameter
        self.g = 0.95 # discount rate
        self.epsilon = 1 # greedy-epsilon
        self.minEpsilon = 0.01
        self.decay = 0.995 # epsilon will decay overtime
        self.memory = deque(maxlen=1000) # for exp replay
        self.model = MLPRegressor(hidden_layer_sizes = (64,32,8),
                                  activation = 'identity',
                                  solver = 'adam') # Qvalue NN approximator
        
        # constant parameter
        self.actionSpace = 3 # {sit = 0, buy = 1, sell = 2} for now...
        self.inventory = []
        self.evalInventory = []



        
    # optimatl action determined by state & currently trained model
    def optimalAct(self,state):
        return np.argmax(self.model.predict(state))
        
    def act(self,state):
        # exploration
        if random.random() <= self.epsilon:
            act = random.randrange(self.actionSpace)
            
        # exploitation
        else:
            act = self.optimalAct(state)
            
        assert act == 0 or act == 1 or act == 2
        
        # Need to discretize more action options here... fk me
        #if act == 1:
        #elif act == 2:
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
                Qprime = reward+self.g*np.amax(self.model.predict(nextState))
            # last data so only current reward
            else: Qprime = reward
            currentQvec = (self.model.predict(state))[0]
            currentQvec[action] = Qprime
            # fit X = state, Y = discounted future rewards for each act
            self.model.partial_fit(state,[currentQvec])
    
    def eval(self,test,date):
        nrow = np.shape(test)[0]
        ncol = 3 # date, closed price, action
        output = np.zeros([nrow,ncol])
        principal = self.evalCash
        state = [test[0]]
        for row in range(0,len(test)-1):
            print("test line %d" %(row))
            action = self.act(state)
            nextState = [test[row+1]]
            if row+1 == len(test)-1: done = 1
            else: done = 0
            # sit
            reward = 0
            currentPrice = test[row][0]
            # buy : add 1 share at current price to inventory
            if action == 1 and currentPrice <= self.evalCash:
                self.evalInventory += [currentPrice]
                self.evalCash -= currentPrice
                print("buy")
            # sell : sell 1 share from inventory at current price
            elif action == 2 and (len(self.evalInventory) != 0):
                pastPrice = self.evalInventory.pop(0)
                reward = max(currentPrice-pastPrice,0)
                self.evalCash += currentPrice
                print("sell")
            self.memory.append((state,action,nextState,reward,done))
            state = nextState
            # update only when enough memory present
            if len(self.memory) > self.batchLen:
                self.update()
                # epsilon decay
                if self.epsilon > self.minEpsilon:
                    self.epsilon *= self.decay
            if len(self.evalInventory) == 0: action = 0
            output[row,0] = date[row][1]
            output[row,1] = test[row][0]
            output[row,2] = action
        print("Total evaluation profit : "+str(self.evalCash-principal))
        return output
            
            
    
        