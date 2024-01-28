# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 21:03:52 2022

@author: deban
"""

#Importing necessary libraries
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import yfinance as yf
from tabulate import tabulate

#Defining the inputs of the function
alpha=0.95
w = [.5, .5]
Tickers= ['CORN', 'WEAT']
start="2020-12-01"
end="2022-11-30"

#Creating the function to calculate Historical & Parametric VaR and TVaR
def VaR_TVaR(Tickers, alpha, w, start, end):
    
    #IF Condition to assume alpha acurately, even if wrong alpha is given
    if alpha> .5:
        alpha=1-alpha
        
    #Creating a dataframe to save closing price of stocks
    df=pd.DataFrame()
    
    #looping throught the two tickers to get closing stock price
    for tic in Tickers:
        df[tic]=yf.Ticker(tic).history(start=start, end=end).Close
        
    #Creating a stock return DataFrame
    ret=pd.DataFrame()
    for tic in Tickers:
        ret[tic]=df[tic].dropna().pct_change()
    
    #Creating a dataframe to calculate historical stats
    Portfolio_df=pd.DataFrame()
    
    #Manually weighting the stock returns to get portfolio returns and saving it
    Portfolio_df['Portfolio_Returns']=ret[Tickers[0]]*w[0]+ret[Tickers[1]]*w[1]
    Returns=Portfolio_df['Portfolio_Returns']
    var_H = Returns.quantile(q=alpha, interpolation="higher") #Formula for historical VaR
    tvar_H= Returns[Returns<var_H].mean() #Formula for historical TVaR
    
    #Making a matrix and transposing of Weight list to prepare it for multiplication
    W=np.matrix(w).T

    #Finding mean of portfolio returns
    mu = np.matmul(ret,W).mean()
    
    #Creating a Covariance Matrix of returns
    CovMtrx = ret.cov()
        
    #Calculating the portfolio variance
    portvar= (W.T @ CovMtrx @ W)[0][0]
    
    #Portfolio Standard Deviation
    portSTD=np.sqrt(portvar)
    
    var_P=(mu+stats.norm.ppf(alpha, 0,1)*portSTD)[0] #Formula for Parametric VaR 
    
    #Formula for Parametric TVaR
    tvar_P=(-mu -portSTD * stats.norm.pdf(stats.norm.ppf(alpha, 0, 1),0,1) / (alpha))[0]
    
    #Creating an array from return dataframe
    ret=ret.dropna() 
    ret_array=ret.to_numpy() 
    
    #Creating a description dataframe of the stock returns
    sum_stats = stats.describe(ret_array, axis = 0)._asdict()
    sum_stats = pd.DataFrame(sum_stats, columns=sum_stats.keys()).transpose()
    sum_stats.columns = Tickers
    
    #printing the description dataframe in a table
    print(tabulate(sum_stats, headers='keys'))
    print('\n')
    
    #printing the VaR and TVaR in a table
    VaR_report = pd.DataFrame([['VaR_Parametric', var_P], ['TVaR_Parametric', tvar_P], ['VaR_HS', var_H], 
                               ['TVAR_HS', tvar_H]], columns=['Metric', 'Value'])
    print(tabulate(VaR_report))


#The function run with inputs
VaR_TVaR(Tickers, alpha, w, start, end)