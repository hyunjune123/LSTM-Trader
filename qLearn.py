from sklearn.preprocessing import normalize 
from environment import Agent
import numpy as np
import random
#import matplotlib.pyplot as plt


def main():
    # takes in only tidy csv file with no headers in the following format
    # index; Date; var1; var2; var3; ...
    rawData = np.genfromtxt("processedSPY.csv",delimiter=",")
    testLength = 300
    trainData = rawData[1:np.shape(rawData)[0]-testLength,2:]
    testData = rawData[np.shape(rawData)[0]-testLength-1:,2:]
    dateIndex = rawData[np.shape(rawData)[0]-testLength-1:,:2]
    
    maxEpisode = 3
    
    # Initialize agent (
    agent = Agent(10,1000,32)
    agent.model.fit([trainData[0]],[[0,0,0]])
    
    # Q Learning
    episode = 1
    
    # loop until maxEpisode
    while episode <= maxEpisode:
        print("Episode "+str(episode))
        state = [trainData[0]]
        principal = agent.cash
        
        # Loop through training data
        for row in range(0,len(trainData)-1):
            print("training line %d" %(row))
            action = agent.act(state)
            nextState = [trainData[row+1]]
            if row+1 == len(trainData)-1: done = 1
            else: done = 0
            # sit
            reward = 0
            currentPrice = trainData[row][0]
            # buy : add 1 share at current price to inventory
            if action == 1:
                agent.inventory += [currentPrice]
                agent.cash -= currentPrice
            # sell : sell 1 share from inventory at current price
            elif action == 2 and (len(agent.inventory) != 0):
                pastPrice = agent.inventory.pop(0)
                reward = max(currentPrice-pastPrice,0)
                agent.cash += currentPrice
            agent.memory.append((state,action,nextState,reward,done))
            state = nextState
            # update only when enough memory present
            if len(agent.memory) > agent.batchLen:
                agent.update()
                # epsilon decay
                if agent.epsilon > agent.minEpsilon:
                    agent.epsilon *= agent.decay
            
        # episode ended, profit computed  / printed
        profit = agent.cash - principal
        print("Episode %d done with profit %f" % (episode, profit))
        episode += 1
    
    
    
    output = agent.eval(testData,dateIndex)
    np.savetxt("test.csv",output,delimiter = ",")
        
    
if __name__ == "__main__":
    main()