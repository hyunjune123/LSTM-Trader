![maxresdefault](https://user-images.githubusercontent.com/59113962/86430444-4d02e700-bcc0-11ea-99af-09189e1ccbfd.jpg)

# LSTM Price Trend Estimator

Research project by Woo June Cha under supervision of professor Chad Shafer from Carnegie Mellon University. <br /> 
View "Report.pdf" for the full project report.

## Abstract
For non-financial experts, it is difficult to determine when to enter a position in a stock market. The objective of this project is to build a machine learning model, that could estimate the price trend and identify price trend shifts. We can exploit the model output to make trade decisions that would yield profit.

## Method
Use an LSTM binary classification model to estimate either we are at an up trend of a down trend, and take actions (enter, exit) according to trend shifts. In this project, I used S&P500 index to train the LSTM model for trend prediction and used the model to backtest on Microsoft, Apple, and Amazon stock.


## Result
Backtest results of Microsoft, Amazon and Apple stock on 40 different time periods over the past 16 years. We observe that all distribution of backtest profits are heavily right skewed with positive mean (16% ~ 35%). The first quantiles of each distribution is around -1.7%, which indicates that we expect the model the yield profit greater than -1.7% 75% of the time.

![](distribution.PNG)
