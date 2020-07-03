# Implementation of Q learning in stock markets

The following research is done by Woo June Cha under supervision of professor Chad Shafer from Carnegie Mellon University.

## Abstract
Many investors fail because they often don’t have the ability to appropriately time the market. In most cases, they have biased and subjective beliefs on their investments, which leads to erroneous decisions. For a more stable and profitable trade, some quantitative analysists rely on technical analysis based on historical price data. There are also various profitable algorithmic trading techniques, but even in those cases, it is up to the analysists to make the final call to buy or sell.

A decisive objective trading algorithm with guaranteed high return would be ideal. However, such decisive algorithmic trading models are still in early development. Although there is no empirical model that could completely harnesses the unpredictive nature of markets, but there is an ongoing effort to implement Q learning in market decisions. The goal of this research is to find the appropriate parameters for a Q learning based trading model that could possibly yield high and steady rate of return.

## Strategy
The trading strategy used for the model is based on the Bollinger Band Squeeze Strategy. This popular strategy is based on the belief that price variation occurs when a “Squeeze” phase ends. The “Squeeze” phase is the time period when the Bollinger band is within the Keltner band. In the image below, the yellow shaded region is the “Squeeze” phase. After the squeeze is over, traders sell when price fall below the Bollinger band, and buys if price goes above the band.

## Q learning
Just as the name implies, we will train the model so it can learn how to make an optimal decision given some states or characteristics of the market. The imaginary agent (or trader) will observe the market to get useful states of the market. The expected future profits of taking each action (buy, sell, or hold) will be computed by a neural network (DNN), to determine an action. Once the agent takes the action and gets rewarded (profit or loss) from that action, the neural network will be updated so that next time it computes more accurate expected future profits. Each components of the model of interest would be the following:

- State : Moving averages, Bollinger band on price and volume, RSI, etc
- Action : {buy, sit, sell} (or could be discretized to buy and sell at 0% ~ 100%)
- Environment : Stock’s historical data
- Reward : Sell : (sold $ - bought $), Buy : some constant x, Hold : 0
- Q value approximator : Default DNN, but LSTM or RNN could be an option (input is time dependent)
