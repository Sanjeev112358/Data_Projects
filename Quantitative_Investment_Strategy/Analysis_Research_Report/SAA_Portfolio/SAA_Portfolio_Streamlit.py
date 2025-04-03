import streamlit as st

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from plotnine import *
import random

import seaborn as sn
import matplotlib.pyplot as plt
from IPython.display import display
import plotly.express as px
import plotly.graph_objects as go

import yfinance as yf

from datetime import date
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
from scipy.stats import linregress
import itertools
from scipy.optimize import minimize


st.title("Sanjeev Narasimhan SAA Portfolio Analytics")


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Index & ETF Performances',
                                                    'Pure S&P',
                                                    'Rebalancing 60/40',
                                                    'Rebalancing Equal Weighted',
                                                    'Rebalancing Ray Dalio',
                                                    'Rebalancing SAA',
                                                    'Comparing Performances'])



# ------------------ LOADING DATA ------------------ #

data = pd.read_csv('Index_Data.csv')
data2 = pd.read_csv('Ray_Dalio_ETFs_Data.csv')


# ------------------ INTERMEDIATE FUNCTIONS ------------------ #

def get_risk_free_rate():
    # download 10-year us treasury bills rates
    annualized = yf.download("^TNX")["Close"]
    DF = pd.DataFrame(annualized)
    DF.columns = ['annualized']
    return DF

if __name__ == "__main__":
    rates = get_risk_free_rate()
    Risk_Free_Return_Percentage = np.round((rates["annualized"].iloc[-1]), 4)

Risk_Free_Return = (Risk_Free_Return_Percentage / 100)

Risk_Free_Return = 0.0
risk_free_rate = 0.0

def calculate_sharpe_ratio(daily_returns, risk_free_rate):
    annualized_std = np.std(daily_returns) * np.sqrt(252)  # Assuming 252 trading days in a year
    portfolio_return = np.mean(daily_returns) * 252
    sharpe_ratio = (portfolio_return - risk_free_rate) / annualized_std
    return sharpe_ratio

def calculate_beta(df, benchmark_column, portfolio_column):
    # Calculate the covariance between the portfolio and the benchmark returns
    covariance = df[portfolio_column].cov(df[benchmark_column])

    # Calculate the variance of the benchmark returns
    benchmark_variance = df[benchmark_column].var()

    # Ensure that both covariance and benchmark_variance are not series
    if isinstance(covariance, pd.Series) or isinstance(benchmark_variance, pd.Series):
        raise TypeError("Covariance or Variance calculation did not return a single value as expected.")

    # Calculate beta as the ratio of covariance to variance
    beta = covariance / benchmark_variance
    return beta

def calculate_volatility(daily_returns):
    annualized_std = np.std(daily_returns) * np.sqrt(252)  # Assuming 252 trading days in a year
    return annualized_std

def calculate_max_drawdown(portfolio_returns):
    # Calculate the cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    # Calculate the previous peaks
    previous_peaks = cumulative_returns.cummax()
    # Calculate the drawdowns
    drawdowns = (cumulative_returns - previous_peaks) / previous_peaks
    # Find the maximum drawdown
    max_drawdown = drawdowns.min()
    return max_drawdown






with tab1:
    # ------------------ INDEX & ETF PERFORMANCE ------------------ #
    
    st.markdown("# Universe of Indexes")
        
    My_Portfolio_X = data[['Dates', 'SPX Index', 'LBUSTRUU Index', 'SPGSCI Index', 'RMSG Index',
                        'LP01TREU Index', 'TWSE Index', 'IBOV Index', 'SASEIDX Index',
                        'NKY Index', 'MXDK Index', 'SMI Index', 'VNINDEX Index']]

    My_Portfolio_X.fillna(method='ffill', inplace=True)

    My_Portfolio_X = My_Portfolio_X.drop_duplicates()

    myIndexes = ['SPX Index', 'LBUSTRUU Index', 'SPGSCI Index', 'RMSG Index',
                'LP01TREU Index', 'TWSE Index', 'IBOV Index', 'SASEIDX Index',
                'NKY Index', 'MXDK Index', 'SMI Index', 'VNINDEX Index']

    # Make sure 'Date' column is in datetime format
    My_Portfolio_X['Dates'] = pd.to_datetime(My_Portfolio_X['Dates'])

    # Sort dataframe by date in descending order
    My_Portfolio_X.sort_values(by='Dates', ascending=True, inplace=True)

    My_Portfolio_X_Optimization_Data = My_Portfolio_X.loc[(My_Portfolio_X['Dates'] >= '2010-01-01')]
    My_Portfolio_X = My_Portfolio_X.loc[(My_Portfolio_X['Dates'] >= '2001-05-30')]

    # Set initial investment to 100 for each index
    initial_investment = 100

    for index in myIndexes:
        My_Portfolio_X[index] = My_Portfolio_X[index] / My_Portfolio_X.iloc[0][index]  * initial_investment

    # Check the updated dataframe
    My_Portfolio_X2 = My_Portfolio_X

    # Calculate daily returns
    for index in myIndexes:
        My_Portfolio_X[index + " Daily Returns"] = My_Portfolio_X[index].pct_change()
        My_Portfolio_X_Optimization_Data[index + " Daily Returns"] = My_Portfolio_X_Optimization_Data[index].pct_change()


    st.markdown("### Index Data")
    st.dataframe(My_Portfolio_X)


    # Define the x-axis (dates)
    dates = My_Portfolio_X['Dates']
    
    # Plotting
    fig = go.Figure()
    
    for index in myIndexes:
        fig.add_trace(go.Scatter(
            x = dates,
            y = My_Portfolio_X[index],
            mode = 'lines',
            name = index,
        ))
    
    fig.update_layout(
        title = 'Performance of Indexes',
        xaxis_title = 'Date',
        yaxis_title = 'Value',
        showlegend = True,
        xaxis = dict(tickformat = '%m-%y'),
        width = 1500,
        height = 600
    )


    st.markdown("### Index Performance")
    st.plotly_chart(fig)


    correlation_matrix = My_Portfolio_X[['SPX Index Daily Returns', 'LBUSTRUU Index Daily Returns', 'SPGSCI Index Daily Returns', 
                                         'RMSG Index Daily Returns', 'LP01TREU Index Daily Returns', 'TWSE Index Daily Returns', 
                                         'IBOV Index Daily Returns', 'SASEIDX Index Daily Returns', 'NKY Index Daily Returns', 
                                         'MXDK Index Daily Returns', 'SMI Index Daily Returns', 'VNINDEX Index Daily Returns']].corr()

    correlation_matrix = correlation_matrix.round(2)

    index_labels = [col.split(' ')[0] for col in correlation_matrix.columns]

    annotations = []
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix.columns)):
            annotations.append(
                go.layout.Annotation(
                    text = str(correlation_matrix.iloc[i, j]),
                    x = index_labels[j],
                    y = index_labels[i],
                    xref = 'x1',
                    yref = 'y1',
                    showarrow = False,
                    font = dict(color = 'white')
                )
            )

    fig = go.Figure(data = go.Heatmap(
        z = correlation_matrix.values,
        x = index_labels,
        y = index_labels,
        colorscale = 'BuPu',
        zmin = -1,
        zmax = 1,
        text = correlation_matrix.values,
        hovertemplate = '%{x}<br>%{y}<br>value: %{text}<extra></extra>'
    ))

    fig.update_layout(
        title = 'Correlation Matrix of Index Returns',
        xaxis_nticks = 36,
        xaxis = dict(tickangle=45),
        yaxis = dict(tickangle=0),
        width = 800,
        height = 600,
        annotations = annotations
    )


    
    st.markdown("### Index Correlations")
    st.plotly_chart(fig)


    st.divider()
    


    My_Portfolio_Z = data2

    My_Portfolio_Z.fillna(method='ffill', inplace=True)

    My_Portfolio_Z = My_Portfolio_Z.drop_duplicates()

    My_Portfolio_Z = My_Portfolio_Z.rename(columns = {"Date" : "Dates"})

    myIndexes = ['VTI US Equity', 'TLT US Equity', 'IEI US Equity',
                'DBC US Equity', 'GLD US Equity']

    # Make sure 'Date' column is in datetime format
    My_Portfolio_Z['Dates'] = pd.to_datetime(My_Portfolio_Z['Dates'])

    # Sort dataframe by date in descending order
    My_Portfolio_Z.sort_values(by='Dates', ascending=True, inplace=True)

    # Set initial investment to 100 for each index
    initial_investment = 100

    for index in myIndexes:
        My_Portfolio_Z[index] = My_Portfolio_Z[index] / My_Portfolio_Z.iloc[0][index]  * initial_investment


    # Calculate daily returns
    for index in myIndexes:
        My_Portfolio_Z[index + " Daily Returns"] = My_Portfolio_Z[index].pct_change()


    st.markdown("### ETF Data")
    st.dataframe(My_Portfolio_Z)
        
        
    # Define the x-axis (dates)
    dates = My_Portfolio_Z['Dates']

    # Plotting
    fig = go.Figure()
    
    for index in myIndexes:
        fig.add_trace(go.Scatter(
            x = dates,
            y = My_Portfolio_Z[index],
            mode = 'lines',
            name = index,
        ))
    
    fig.update_layout(
        title = 'Performance of ETFs',
        xaxis_title = 'Date',
        yaxis_title = 'Value',
        showlegend = True,
        xaxis = dict(tickformat = '%m-%y'),
        width = 1500,
        height = 600
    )
    

    st.markdown("### ETF Performance")
    st.plotly_chart(fig)
    
    
    correlation_matrix = My_Portfolio_Z[['VTI US Equity Daily Returns', 'TLT US Equity Daily Returns', 'IEI US Equity Daily Returns',
                                        'DBC US Equity Daily Returns', 'GLD US Equity Daily Returns']].corr()

    correlation_matrix = correlation_matrix.round(2)

    index_labels = [col.split(' ')[0] for col in correlation_matrix.columns]

    annotations = []
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix.columns)):
            annotations.append(
                go.layout.Annotation(
                    text = str(correlation_matrix.iloc[i, j]),
                    x = index_labels[j],
                    y = index_labels[i],
                    xref = 'x1',
                    yref = 'y1',
                    showarrow = False,
                    font = dict(color = 'white')
                )
            )

    fig = go.Figure(data = go.Heatmap(
        z = correlation_matrix.values,
        x = index_labels,
        y = index_labels,
        colorscale = 'BuPu',
        zmin = -1,
        zmax = 1,
        text = correlation_matrix.values,
        hovertemplate = '%{x}<br>%{y}<br>value: %{text}<extra></extra>'
    ))

    fig.update_layout(
        title = 'Correlation Matrix of ETF Returns',
        xaxis_nticks = 36,
        xaxis = dict(tickangle=45),
        yaxis = dict(tickangle=0),
        width = 800,
        height = 600,
        annotations = annotations
    )
    
    
    st.markdown("### ETF Correlations")
    st.plotly_chart(fig)






with tab2:
    # ------------------ PURE S&P ------------------ #
    
    My_Portfolio_X.fillna(method='ffill', inplace=True)
    My_Portfolio_X = My_Portfolio_X.dropna()

    # Initial investment in stocks
    spx_investment = 100

    # Initialize lists to store values
    dates = []
    spx_values = []
    portfolio_values = []

    # Iterate through each row of the DataFrame
    for index, row in My_Portfolio_X.iterrows():
        # Calculate the value of the stock investment
        spx_value = spx_investment * (1 + row['SPX Index Daily Returns'])

        # Calculate the total portfolio value
        portfolio_value = spx_value

        # Update the initial investment for the next iteration
        spx_investment = spx_value

        # Append the values to the lists
        dates.append(row['Dates'])
        spx_values.append(spx_value)
        portfolio_values.append(portfolio_value)

    # Create a new DataFrame with the calculated values
    portfolio_performance_0 = pd.DataFrame({
        'Date': dates,
        'SPX Value': spx_values,
        'Total Portfolio Value': portfolio_values
    })

    portfolio_performance_0['Portfolio Daily Returns'] = portfolio_performance_0['Total Portfolio Value'].pct_change()
    
    
    st.markdown('# Pure S&P Performance')
    
    st.markdown('### Pure S&P Data')
    st.dataframe(portfolio_performance_0)
    

    # Define the x-axis (dates)
    dates = portfolio_performance_0['Date']

    # Plotting
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x = dates,
        y = portfolio_performance_0['Total Portfolio Value'],
        mode = 'lines',
        name = 'SPX',
        line = dict(color = 'firebrick')
    ))
    
    fig.update_layout(
        title = 'Pure SPX Portfolio Performance',
        xaxis_title = 'Date',
        yaxis_title = 'Value',
        xaxis = dict(tickformat = '%m-%y'),
        width = 1200,
        height = 600
    )
    
    st.plotly_chart(fig)
    





with tab3:
    # ------------------ REBALANCING 60/40 ------------------ #   
     
    spx_investment = 60  # Initial investment in stocks
    lbustruu_investment = 40   # Initial investment in bonds

    # Initialize lists to store values
    dates = []
    spx_values = []
    lbustruu_values = []
    portfolio_values = []

    # Iterate through each row of the DataFrame
    for index, row in My_Portfolio_X.iterrows():
        # Check if the current date is a rebalancing date
        if row['Dates'].month == 1 or (row['Dates'].month in [3, 6, 9] and row['Dates'].day == 1):
            # Check if the rebalancing date is a trading day
            if row['Dates'].weekday() < 5:
                rebalancing_date = row['Dates']
            else:
                # Find the next business day for rebalancing
                rebalancing_date = row['Dates'] + BDay(1)

            # Calculate the total portfolio value
            total_portfolio_value = spx_investment + lbustruu_investment

            # Rebalance the portfolio
            spx_investment = total_portfolio_value * 0.6
            lbustruu_investment = total_portfolio_value * 0.4

        # Calculate the value of the stock investment
        spx_value = spx_investment * (1 + row['SPX Index Daily Returns'])

        # Calculate the value of the bond investment
        lbustruu_value = lbustruu_investment * (1 + row['LBUSTRUU Index Daily Returns'])

        # Calculate the total portfolio value
        portfolio_value = spx_value + lbustruu_value

        # Update the initial investment for the next iteration
        spx_investment = spx_value
        lbustruu_investment = lbustruu_value

        # Append the values to the lists
        dates.append(row['Dates'])
        spx_values.append(spx_value)
        lbustruu_values.append(lbustruu_value)
        portfolio_values.append(portfolio_value)

    # Create a new DataFrame with the calculated values
    portfolio_performance_2 = pd.DataFrame({
        'Date': dates,
        'SPX Value': spx_values,
        'LBUSTRUU Value': lbustruu_values,
        'Total Portfolio Value': portfolio_values
    })

    # Rebalancing 60/40 Final Portfolio DF
    portfolio_performance_2['Portfolio Daily Returns'] = portfolio_performance_2['Total Portfolio Value'].pct_change()    

    st.markdown('# Rebalancing 60/40 Performance')
    
    st.markdown('### Rebalancing 60/40 Data')
    st.dataframe(portfolio_performance_2)
    
    
    # Define the x-axis (dates)
    dates = portfolio_performance_2['Date']

    # Define the y-axis (portfolio values)
    rebalanced_60_40 = portfolio_performance_2['Total Portfolio Value']

    # Plotting
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x = dates,
        y = portfolio_performance_2['Total Portfolio Value'],
        mode = 'lines',
        name = '60/40',
        line = dict(color = 'cornflowerblue')
    ))
    
    fig.update_layout(
        title = 'Rebalancing 60/40 Portfolio Performance',
        xaxis_title = 'Date',
        yaxis_title = 'Value',
        xaxis = dict(tickformat = '%m-%y'),
        width = 1200,
        height = 600
    )
    
    st.plotly_chart(fig)
    

    
    
    
    
    
with tab4:
    # ------------------ EQUAL WEIGHTED ------------------ #    
    
    myIndexes = ['SPX Index', 'LBUSTRUU Index', 'SPGSCI Index', 'RMSG Index',
                'LP01TREU Index', 'TWSE Index', 'IBOV Index', 'SASEIDX Index',
                'NKY Index', 'MXDK Index', 'SMI Index', 'VNINDEX Index']

    # Initialize the Baseline_SAA_Weights DataFrame
    numIndexes = len(myIndexes)
    Baseline_SAA_Weights = pd.DataFrame(index=myIndexes, columns=['Weight'])

    # Set the weights for each index
    Baseline_SAA_Weights['Weight'] = 0.0833333

    # Calculate the initial investments
    Baseline_SAA_Weights['Initial Investment'] = 100 * Baseline_SAA_Weights['Weight']
    investments = {index: Baseline_SAA_Weights.loc[index, 'Initial Investment'] for index in myIndexes}

    # Initialize lists to store values
    dates = []
    index_values = {index: [] for index in myIndexes}  # Initial value for each index
    portfolio_values = []  # Total portfolio value

    # Iterate through each row of the DataFrame
    for _, row in My_Portfolio_X.iterrows():
        # Check if the current date is a rebalancing date
        if row['Dates'].month == 1 or (row['Dates'].month in [3, 6, 9] and row['Dates'].day == 1):
            # Check if the rebalancing date is a trading day
            if row['Dates'].weekday() < 5:
                rebalancing_date = row['Dates']
            else:
                # Find the next business day for rebalancing
                rebalancing_date = row['Dates'] + BDay(1)

            # Rebalance the portfolio
            total_portfolio_value = sum(investments.values())
            for index in myIndexes:
                investments[index] = total_portfolio_value * Baseline_SAA_Weights.loc[index, 'Weight']

        # Calculate the value of each index
        for index in myIndexes:
            index_values[index].append(investments[index] * (1 + (row[index] / 100)))

        # Calculate the total portfolio value
        portfolio_value = sum(investments[index] * (1 + (row[index] / 100)) for index in myIndexes)
        portfolio_values.append(portfolio_value)

        # Append the values to the lists
        dates.append(row['Dates'])

    # Create a new DataFrame with the calculated values
    portfolio_performance_w = pd.DataFrame({
        'Date': dates,
        **{index + ' Value': values for index, values in index_values.items()},
        'Total Portfolio Value': portfolio_values
    })

    # Divide all index values and the total portfolio value by 2
    portfolio_performance_w[[index + ' Value' for index in myIndexes] + ['Total Portfolio Value']] /= 2

    # Calculate the daily returns
    portfolio_performance_w['Portfolio Daily Returns'] = portfolio_performance_w['Total Portfolio Value'].pct_change()
    
    
    st.markdown('# Rebalancing Equal Weighted Performance')
    
    st.markdown('### Equal Weighted Data')
    st.dataframe(portfolio_performance_w)
    
    
    # Define the x-axis (dates)
    dates = portfolio_performance_w['Date']

    # Define the y-axis (portfolio values)
    rebalanced_ew_saa = portfolio_performance_w['Total Portfolio Value']

    # Plotting
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x = dates,
        y = portfolio_performance_w['Total Portfolio Value'],
        mode = 'lines',
        name = 'EW',
        line = dict(color = 'mediumpurple')
    ))
    
    fig.update_layout(
        title = 'Rebalancing Equal Weighted Portfolio Performance',
        xaxis_title = 'Date',
        yaxis_title = 'Value',
        xaxis = dict(tickformat = '%m-%y'),
        width = 1200,
        height = 600
    )
    
    st.plotly_chart(fig)





with tab5:
    # ------------------ RAY DALIO ------------------ #   
    
    myIndexes = ['VTI US Equity', 'TLT US Equity', 'IEI US Equity',
                'DBC US Equity', 'GLD US Equity']
    
    Ray_Dalio_Portfolio_Allocation = {'Index': ['VTI', 'TLT', 'IEI', 'DBC', 'GLD'],
                                    'Allocation': [0.3, 0.4, 0.15, 0.075, 0.075]}

    Ray_Dalio_Portfolio_Allocation = pd.DataFrame(Ray_Dalio_Portfolio_Allocation)

    # Initialize lists to store values
    dates = []
    portfolio_values = []

    # Initial investments based on optimal weights
    investments = {index: initial_investment * weight for index, weight in zip(myIndexes, Ray_Dalio_Portfolio_Allocation["Allocation"])}

    # Lists to store the values of each index's allocation
    index_values = {index: [] for index in myIndexes}

    # Iterate through each row of the DataFrame
    for _, row in My_Portfolio_Z.iterrows():
        # Check if the current date is a rebalancing date
        if row['Dates'].month == 1 or (row['Dates'].month in [3, 6, 9] and row['Dates'].day == 1):
            # Check if the rebalancing date is a trading day
            if row['Dates'].weekday() < 5:
                rebalancing_date = row['Dates']
            else:
                # Find the next business day for rebalancing
                rebalancing_date = row['Dates'] + BDay(1)

            # Calculate the total portfolio value
            total_portfolio_value = sum(investments.values())

            # Rebalance the portfolio
            for index in myIndexes:
                investments[index] = total_portfolio_value * Ray_Dalio_Portfolio_Allocation["Allocation"][myIndexes.index(index)]

        # Calculate the value of each index
        for index in myIndexes:
            index_values[index].append(investments[index] * (1 + (row[index] / 100)))

        # Calculate the value of the portfolio
        portfolio_value = sum(index_values[index][-1] for index in myIndexes)

        # Append the values to the lists
        dates.append(row['Dates'])
        portfolio_values.append(portfolio_value)

    # Create a new DataFrame with the calculated values
    portfolio_performance_y = pd.DataFrame({
        'Date': dates,
        **{index + ' Value': values for index, values in index_values.items()},
        'Total Portfolio Value': portfolio_values
    })

    # Divide all index values and the total portfolio value by 2
    portfolio_performance_y[[index + ' Value' for index in myIndexes] + ['Total Portfolio Value']] /= 2

    # Calculate the daily returns
    portfolio_performance_y['Portfolio Daily Returns'] = portfolio_performance_y['Total Portfolio Value'].pct_change()
    
    
    st.markdown('# Rebalancing Ray Dalio Performance')
    
    st.markdown('### Ray Dalio Data')
    st.dataframe(portfolio_performance_y)
    
    
    # Define the x-axis (dates)
    dates = portfolio_performance_y['Date']

    # Define the y-axis (portfolio values)
    rebalanced_ray_dalio_portfolio = portfolio_performance_y['Total Portfolio Value']

    # Plotting
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x = dates,
        y = portfolio_performance_y['Total Portfolio Value'],
        mode = 'lines',
        name = 'Ray Dalio',
        line = dict(color = 'forestgreen')
    ))
    
    fig.update_layout(
        title = 'Rebalancing Ray Dalio Portfolio Performance',
        xaxis_title = 'Date',
        yaxis_title = 'Value',
        xaxis = dict(tickformat = '%m-%y'),
        width = 1200,
        height = 600
    )
    
    st.plotly_chart(fig)
    
    
    
    
    
    
with tab6:
    # ------------------ REBALANCING SAA ------------------ #  

    # Function to calculate portfolio metrics
    def calculate_metrics(weights, daily_returns, risk_free_rate=0):
        portfolio_return = np.dot(daily_returns.mean(), weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        return portfolio_return, portfolio_std, sharpe_ratio

    # Objective function to maximize Sharpe ratio
    def objective_function(weights, daily_returns, risk_free_rate=0):
        portfolio_return, portfolio_std, sharpe_ratio = calculate_metrics(weights, daily_returns, risk_free_rate)
        return -sharpe_ratio  # Minimize negative Sharpe ratio to maximize

    # Define the indexes and their respective daily return columns
    myIndexes = ['SPX Index', 'LBUSTRUU Index', 'SPGSCI Index', 'RMSG Index',
                'LP01TREU Index', 'TWSE Index', 'IBOV Index', 'SASEIDX Index',
                'NKY Index', 'MXDK Index', 'SMI Index', 'VNINDEX Index']

    # Targets
    min_target_return = 0.13
    max_target_volatility = 0.12

    # Setup initial weights
    num_indexes = len(myIndexes)
    init_weights = [1 / num_indexes] * num_indexes

    # Constraints
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda x: calculate_metrics(x, My_Portfolio_X_Optimization_Data[myIndexes].pct_change().dropna())[0] * 252 - min_target_return},
        {'type': 'ineq', 'fun': lambda x: calculate_metrics(x, My_Portfolio_X_Optimization_Data[myIndexes].pct_change().dropna())[1] - max_target_volatility},
        {'type': 'ineq', 'fun': lambda x: 0.25 - x}
    )

    # Bounds
    bounds = tuple((0, 1) for _ in range(num_indexes))

    # Optimize
    optimal_weights = minimize(objective_function, init_weights, args=(My_Portfolio_X_Optimization_Data[myIndexes].pct_change().dropna(), risk_free_rate),
                            method='SLSQP', bounds=bounds, constraints=constraints)

    # Create DataFrame for results
    optimal_allocation = pd.DataFrame(index=myIndexes)
    optimal_allocation['Optimal Weights'] = optimal_weights.x
    optimal_allocation['Optimal Weights'] = optimal_allocation['Optimal Weights'].apply(lambda x: round(x, 6))
    
    optimal_allocation = optimal_allocation[optimal_allocation['Optimal Weights'] > 0]

    # Calculate portfolio metrics
    portfolio_return, portfolio_std, sharpe_ratio = calculate_metrics(optimal_weights.x, My_Portfolio_X_Optimization_Data[myIndexes].pct_change().dropna(), risk_free_rate)


    SAA_Portfolio_Allocation = {'Index': ["SPX", "RMSG", "LP01TREU", "NKY", "MXDK"],
                                'Allocation': [0.25, 0.112693, 0.25, 0.137307, 0.25]}

    np.sum(SAA_Portfolio_Allocation['Allocation'])

    SAA_Portfolio_Allocation = pd.DataFrame(SAA_Portfolio_Allocation)
        
    
    st.markdown('# Rebalancing SAA Performance')
    st.markdown('## Determining Optimal Allocation')
    
    st.markdown('### Optimal Index Weights')
    st.dataframe(optimal_allocation)
    st.write("Expected Portfolio Return: {:.2f}%".format(portfolio_return * 100))
    st.write("Expected Portfolio Volatility: {:.2f}%".format(portfolio_std * 100))
    st.write("Expected Sharpe Ratio: {:.4f}".format(sharpe_ratio))
    
    
    correlation_matrix = My_Portfolio_X[['SPX Index Daily Returns', 'RMSG Index Daily Returns', 'LP01TREU Index Daily Returns',
                                        'NKY Index Daily Returns', 'MXDK Index Daily Returns']].corr()

    correlation_matrix = correlation_matrix.round(2)

    index_labels = [col.split(' ')[0] for col in correlation_matrix.columns]

    annotations = []
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix.columns)):
            annotations.append(
                go.layout.Annotation(
                    text = str(correlation_matrix.iloc[i, j]),
                    x = index_labels[j],
                    y = index_labels[i],
                    xref = 'x1',
                    yref = 'y1',
                    showarrow = False,
                    font = dict(color = 'white')
                )
            )

    fig = go.Figure(data = go.Heatmap(
        z = correlation_matrix.values,
        x = index_labels,
        y = index_labels,
        colorscale = 'BuPu',
        zmin = -1,
        zmax = 1,
        text = correlation_matrix.values,
        hovertemplate = '%{x}<br>%{y}<br>value: %{text}<extra></extra>'
    ))

    fig.update_layout(
        title = 'Correlation Matrix of Index Returns',
        xaxis_nticks = 36,
        xaxis = dict(tickangle=45),
        yaxis = dict(tickangle=0),
        width = 800,
        height = 600,
        annotations = annotations
    )


    st.markdown("### Index Correlations")
    st.plotly_chart(fig)
    
    
    st.divider()


    # Initialize lists to store values
    dates = []
    portfolio_values = []

    # Initial investments based on optimal weights
    initial_investment = 100  # initial investment amount
    investments = {index: initial_investment * weight for index, weight in zip(myIndexes, optimal_weights.x)}

    # Lists to store the values of each index's allocation
    index_values = {index: [] for index in myIndexes}

    # Iterate through each row of the DataFrame
    for _, row in My_Portfolio_X.iterrows():
        # Check if the current date is a rebalancing date
        if row['Dates'].month == 1 or (row['Dates'].month in [3, 6, 9] and row['Dates'].day == 1):
            # Check if the rebalancing date is a trading day
            if row['Dates'].weekday() < 5:
                rebalancing_date = row['Dates']
            else:
                # Find the next business day for rebalancing
                rebalancing_date = row['Dates'] + BDay(1)

            # Calculate the total portfolio value
            total_portfolio_value = sum(investments.values())

            # Rebalance the portfolio
            for index in myIndexes:
                investments[index] = total_portfolio_value * optimal_weights.x[myIndexes.index(index)]

        # Calculate the value of each index
        for index in myIndexes:
            index_values[index].append(investments[index] * (1 + (row[index] / 100)))

        # Calculate the value of the portfolio
        portfolio_value = sum(index_values[index][-1] for index in myIndexes)

        # Append the values to the lists
        dates.append(row['Dates'])
        portfolio_values.append(portfolio_value)

    # Create a new DataFrame with the calculated values
    portfolio_performance_z = pd.DataFrame({
        'Date': dates,
        **{index + ' Value': values for index, values in index_values.items()},
        'Total Portfolio Value': portfolio_values
    })

    # # Divide all index values and the total portfolio value by 2
    portfolio_performance_z[[index + ' Value' for index in myIndexes] + ['Total Portfolio Value']] /= 2

    # Calculate the daily returns
    portfolio_performance_z['Portfolio Daily Returns'] = portfolio_performance_z['Total Portfolio Value'].pct_change()


    st.markdown('## Creating Portfolio')
    st.markdown('### SAA Data')
    st.dataframe(portfolio_performance_z)
    
    
    # Define the x-axis (dates)
    dates = portfolio_performance_z['Date']

    # Define the y-axis (portfolio values)
    rebalanced_saa = portfolio_performance_z['Total Portfolio Value']

    # Plotting
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x = dates,
        y = portfolio_performance_z['Total Portfolio Value'],
        mode = 'lines',
        name = 'EW',
        line = dict(color = 'yellowgreen')
    ))
    
    fig.update_layout(
        title = 'Rebalancing SAA Portfolio Performance',
        xaxis_title = 'Date',
        yaxis_title = 'Value',
        xaxis = dict(tickformat = '%m-%y'),
        width = 1200,
        height = 600
    )
    
    st.plotly_chart(fig)
    
    
    
    
    
    
with tab7:
    # ------------------ COMPARING PERFORMANCE ------------------ # 
    
    st.markdown('# Comparing Portfolio Performances')
    st.markdown('## Performance Overview')
    
    
    Complete_Portfolio_Performance = pd.DataFrame()
    Complete_Portfolio_Performance['Date'] = portfolio_performance_z['Date']

    Complete_Portfolio_Performance['Pure SPX Portfolio Value'] = portfolio_performance_0['Total Portfolio Value']
    Complete_Portfolio_Performance['Pure SPX Daily Return'] = portfolio_performance_0['Portfolio Daily Returns']

    Complete_Portfolio_Performance['Rebalanced 60/40 Portfolio Value'] = portfolio_performance_2['Total Portfolio Value']
    Complete_Portfolio_Performance['Rebalanced 60/40 Daily Return'] = portfolio_performance_2['Portfolio Daily Returns']

    Complete_Portfolio_Performance['Rebalanced Equal Weighted SAA Portfolio Value'] = portfolio_performance_w['Total Portfolio Value']
    Complete_Portfolio_Performance['Rebalanced Equal Weighted SAA Daily Return'] = portfolio_performance_w['Portfolio Daily Returns']

    Complete_Portfolio_Performance['Rebalanced SAA Portfolio Value'] = portfolio_performance_z['Total Portfolio Value']
    Complete_Portfolio_Performance['Rebalanced SAA Daily Return'] = portfolio_performance_z['Portfolio Daily Returns']

    portfolio_performance_y2 = portfolio_performance_y[["Date", "Total Portfolio Value", "Portfolio Daily Returns"]]
    portfolio_performance_y2 = portfolio_performance_y2.rename(columns = {"Total Portfolio Value" : "Rebalanced Ray Dalio Portfolio Value",
                                                                        "Portfolio Daily Returns" : "Rebalanced Ray Dalio Portfolio Daily Return"})

    Complete_Portfolio_Performance = Complete_Portfolio_Performance.loc[(Complete_Portfolio_Performance['Date'] >= '2001-05-31')]

    Complete_Portfolio_Performance = pd.merge(Complete_Portfolio_Performance, portfolio_performance_y2, on='Date', how='left')

    Complete_Portfolio_Performance.fillna(method='ffill', inplace=True)
    

    st.markdown('### Strategy Data')
    st.dataframe(Complete_Portfolio_Performance)

    Complete_Portfolio_Performance.to_csv("Complete_Portfolio_Performance.csv")

    # Define the x-axis (dates)
    dates = Complete_Portfolio_Performance['Date']

    # Define the y-axis (portfolio values)
    pure_spx = Complete_Portfolio_Performance['Pure SPX Portfolio Value']
    rebalanced_60_40 = Complete_Portfolio_Performance['Rebalanced 60/40 Portfolio Value']
    rebalanced_ew_saa = Complete_Portfolio_Performance['Rebalanced Equal Weighted SAA Portfolio Value']
    rebalanced_ray_dalio_portfolio = Complete_Portfolio_Performance['Rebalanced Ray Dalio Portfolio Value']
    rebalanced_saa = Complete_Portfolio_Performance['Rebalanced SAA Portfolio Value']


   # Plotting
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x = dates,
        y = Complete_Portfolio_Performance['Pure SPX Portfolio Value'],
        mode = 'lines',
        name = 'Pure SPX',
        line = dict(color = 'firebrick')
    ))
    
    fig.add_trace(go.Scatter(
        x = dates,
        y = Complete_Portfolio_Performance['Rebalanced 60/40 Portfolio Value'],
        mode = 'lines',
        name = 'Rebalanced 60/40',
        line = dict(color = 'cornflowerblue')
    ))
    
    fig.add_trace(go.Scatter(
        x = dates,
        y = Complete_Portfolio_Performance['Rebalanced Equal Weighted SAA Portfolio Value'],
        mode = 'lines',
        name = 'Rebalanced Equal Weighted',
        line = dict(color = 'mediumpurple')
    ))
    
    fig.add_trace(go.Scatter(
        x = dates,
        y = Complete_Portfolio_Performance['Rebalanced Ray Dalio Portfolio Value'],
        mode = 'lines',
        name = 'Rebalanced Ray Dalio',
        line = dict(color = 'forestgreen')
    ))
    
    fig.add_trace(go.Scatter(
        x = dates,
        y = Complete_Portfolio_Performance['Rebalanced SAA Portfolio Value'],
        mode = 'lines',
        name = 'Rebalanced SAA',
        line = dict(color = 'yellowgreen')
    ))
    
    fig.update_layout(
        title = 'Performance of Different Portfolio Strategies',
        xaxis_title = 'Date',
        yaxis_title = 'Value',
        xaxis = dict(tickformat = '%m-%y'),
        width = 1500,
        height = 600,
        showlegend = True
    )
    
    
    st.markdown('### Strategy Performances')
    st.plotly_chart(fig)
    
    
    st.divider()
    
    
    st.markdown('## Comparing Performance Metrics')

    # Ensure the 'Date' column is in datetime format
    Complete_Portfolio_Performance['Date'] = pd.to_datetime(Complete_Portfolio_Performance['Date'])

    # Find the last date in the 'Date' column
    last_date = Complete_Portfolio_Performance['Date'].iloc[-1]

    # Define the periods
    periods = {
        "Current": last_date,
        '1yr': last_date - relativedelta(years=1),
        '3yr': last_date - relativedelta(years=3),
        '5yr': last_date - relativedelta(years=5),
        '10yr': last_date - relativedelta(years=10),
        '20yr': last_date - relativedelta(years=20)
    }

    def find_next_trading_day(df, target_date):
        future_dates = df[df['Date'] >= target_date]
        if not future_dates.empty:
            return future_dates.iloc[0].name
        else:
            return None

    # Extract indices for 1yr, 3yr, and 5yr periods
    indices = []
    for period in periods.values():
        # Try to get the index for the exact date
        exact_date_index = Complete_Portfolio_Performance.index[Complete_Portfolio_Performance['Date'] == period]
        if not exact_date_index.empty:
            indices.append(exact_date_index[0])
        else:
            # If the exact date is not present, find the next trading day
            next_date_index = find_next_trading_day(Complete_Portfolio_Performance, period)
            if next_date_index is not None:
                indices.append(next_date_index)

    # Use the indices to select the rows from the original DataFrame
    selected_rows = Complete_Portfolio_Performance.loc[indices]

    # This is your new DataFrame with the selected rows
    portfolio_values = pd.DataFrame(selected_rows)
    portfolio_values['Period'] = ["Current",
                                "1 Year",
                                "3 Year",
                                "5 Year",
                                "10 Year",
                                "20 Year"]

    def get_risk_free_rate():
        # download 10-year us treasury bills rates
        annualized = yf.download("^TNX")["Close"]
        DF = pd.DataFrame(annualized)
        DF.columns = ['annualized']
        return DF

    if __name__ == "__main__":
        rates = get_risk_free_rate()
        Risk_Free_Return_Percentage = np.round((rates["annualized"].iloc[-1]), 4)

    Risk_Free_Return = (Risk_Free_Return_Percentage / 100)

    Risk_Free_Return = 0.0

    def calculate_sharpe_ratio(daily_returns, risk_free_rate):
        annualized_std = np.std(daily_returns) * np.sqrt(252)  # Assuming 252 trading days in a year
        portfolio_return = np.mean(daily_returns) * 252
        sharpe_ratio = (portfolio_return - risk_free_rate) / annualized_std
        return sharpe_ratio, annualized_std

    def calculate_beta(df, benchmark_column, portfolio_column):
        # Calculate the covariance between the portfolio and the benchmark returns
        covariance = df[portfolio_column].cov(df[benchmark_column])

        # Calculate the variance of the benchmark returns
        benchmark_variance = df[benchmark_column].var()

        # Ensure that both covariance and benchmark_variance are not series
        if isinstance(covariance, pd.Series) or isinstance(benchmark_variance, pd.Series):
            raise TypeError("Covariance or Variance calculation did not return a single value as expected.")

        # Calculate beta as the ratio of covariance to variance
        beta = covariance / benchmark_variance
        return beta

    def calculate_volatility(daily_returns):
        annualized_std = np.std(daily_returns) * np.sqrt(252)  # Assuming 252 trading days in a year
        return annualized_std

    def calculate_max_drawdown(portfolio_returns):
        # Calculate the cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        # Calculate the previous peaks
        previous_peaks = cumulative_returns.cummax()
        # Calculate the drawdowns
        drawdowns = (cumulative_returns - previous_peaks) / previous_peaks
        # Find the maximum drawdown
        max_drawdown = drawdowns.min()
        return max_drawdown

    # Function to calculate annualized return, beta, and Sharpe ratio
    def calculate_metrics(portfolioDF, risk_free_rate):
        metrics = {}
        for portfolio_type in ['Pure SPX', 'Rebalanced 60/40', 'Rebalanced Equal Weighted SAA', 'Rebalanced Ray Dalio Portfolio', 'Rebalanced SAA']:
            portfolio_returns = portfolioDF[portfolio_type + ' Daily Return']
            portfolio_return = np.mean(portfolio_returns) * 252
            portfolio_beta = calculate_beta(portfolioDF, 'Rebalanced 60/40 Daily Return', portfolio_type + ' Daily Return')
            sharpe_ratio, _ = calculate_sharpe_ratio(portfolio_returns, risk_free_rate)
            portfolio_volatility = calculate_volatility(portfolio_returns)
            max_drawdown = calculate_max_drawdown(portfolio_returns)
            metrics[portfolio_type] = {'Annualized Return': portfolio_return,
                                    'Portfolio Beta': portfolio_beta,
                                    'Sharpe Ratio': sharpe_ratio,
                                    'Volatility': portfolio_volatility,
                                    'Max Drawdown': max_drawdown}
        return metrics

    # Define periods for 1 year, 3 years, 5 years, and 10 years
    periods = {
        '1 year': last_date - pd.DateOffset(years=1),
        '3 years': last_date - pd.DateOffset(years=3),
        '5 years': last_date - pd.DateOffset(years=5),
        '10 years': last_date - pd.DateOffset(years=10),
        '20 years': last_date - pd.DateOffset(years=20)
    }

    # Define risk-free rate (you may need to update this)
    risk_free_rate = Risk_Free_Return

    # Create a dictionary to store dataframes for each period
    period_dfs = {}

    # Iterate over periods
    for period, end_date in periods.items():
        # Filter rows based on the period
        period_df = Complete_Portfolio_Performance[Complete_Portfolio_Performance['Date'] >= end_date]
        # Calculate metrics for the period
        metrics = calculate_metrics(period_df, risk_free_rate)
        # Create dataframe from metrics dictionary
        period_dfs[period] = pd.DataFrame(metrics).transpose()

    # Access the dataframes for each period
    for period, df in period_dfs.items():
        period_dfs[period] = df

    st.markdown("#### 1 Year Portfolio Performances")
    st.dataframe(period_dfs['1 year'])  # Access the dataframe for 1 year period

    # Define the portfolio names
    portfolios = ['Pure SPX', 'Rebalanced 60/40', 'Rebalanced Equal Weighted SAA', 'Rebalanced Ray Dalio Portfolio', 'Rebalanced SAA']

    # Extract data for each portfolio
    annualized_return = period_dfs['1 year']['Annualized Return']
    portfolio_beta = period_dfs['1 year']['Portfolio Beta']
    sharpe_ratio = period_dfs['1 year']['Sharpe Ratio']
    volatility = period_dfs['1 year']['Volatility']
    max_drawdown = period_dfs['1 year']['Max Drawdown']

    # Define the categories
    categories = ['Annualized Return', 'Portfolio Beta', 'Sharpe Ratio', 'Volatility', 'Max Drawdown']
    
    # Define the colors for the bars
    colors = ['firebrick', 'cornflowerblue', 'mediumpurple', 'forestgreen', 'yellowgreen']

    # Create the bar traces for each portfolio
    traces = []
    for i, portfolio in enumerate(portfolios):
        traces.append(go.Bar(
            x = categories,
            y = np.round(period_dfs['1 year'].iloc[i], 4),
            name = portfolio,
            marker = dict(color = colors[i],
                          line = dict(color = 'black',
                                      width = 1)),
        ))

    # Create the figure
    fig = go.Figure(data = traces)

    # Customize the layout
    fig.update_layout(
        title = 'Comparison of Portfolio Strategies',
        xaxis = dict(title = 'Metrics', tickangle = 45),
        yaxis = dict(title = 'Values'),
        barmode = 'group',
        width = 1200,
        height = 600
    )

    st.plotly_chart(fig)
    
    
    
    st.markdown("#### 3 Year Portfolio Performances:")
    st.dataframe(period_dfs['3 years'])  # Access the dataframe for 3 year period

    # Define the portfolio names
    portfolios = ['Pure SPX', 'Rebalanced 60/40', 'Rebalanced Equal Weighted SAA', 'Rebalanced Ray Dalio Portfolio', 'Rebalanced SAA']

    # Extract data for each portfolio
    annualized_return = period_dfs['3 years']['Annualized Return']
    portfolio_beta = period_dfs['3 years']['Portfolio Beta']
    sharpe_ratio = period_dfs['3 years']['Sharpe Ratio']
    volatility = period_dfs['3 years']['Volatility']
    max_drawdown = period_dfs['3 years']['Max Drawdown']

    # Define the categories
    categories = ['Annualized Return', 'Portfolio Beta', 'Sharpe Ratio', 'Volatility', 'Max Drawdown']

    # Define the colors for the bars
    colors = ['firebrick', 'cornflowerblue', 'mediumpurple', 'forestgreen', 'yellowgreen']

    # Create the bar traces for each portfolio
    traces = []
    for i, portfolio in enumerate(portfolios):
        traces.append(go.Bar(
            x = categories,
            y = np.round(period_dfs['3 years'].iloc[i], 4),
            name = portfolio,
            marker = dict(color = colors[i],
                          line = dict(color = 'black',
                                      width = 1)),
        ))

    # Create the figure
    fig = go.Figure(data = traces)

    # Customize the layout
    fig.update_layout(
        title = 'Comparison of Portfolio Strategies',
        xaxis = dict(title = 'Metrics', tickangle = 45),
        yaxis = dict(title = 'Values'),
        barmode = 'group',
        width = 1200,
        height = 600
    )

    st.plotly_chart(fig)
    
    
    
    st.markdown("#### 5 Year Portfolio Performances:")
    st.dataframe(period_dfs['5 years'])  # Access the dataframe for 5 year period

    # Define the portfolio names
    portfolios = ['Pure SPX', 'Rebalanced 60/40', 'Rebalanced Equal Weighted SAA', 'Rebalanced Ray Dalio Portfolio', 'Rebalanced SAA']

    # Extract data for each portfolio
    annualized_return = period_dfs['5 years']['Annualized Return']
    portfolio_beta = period_dfs['5 years']['Portfolio Beta']
    sharpe_ratio = period_dfs['5 years']['Sharpe Ratio']
    volatility = period_dfs['5 years']['Volatility']
    max_drawdown = period_dfs['5 years']['Max Drawdown']

    # Define the categories
    categories = ['Annualized Return', 'Portfolio Beta', 'Sharpe Ratio', 'Volatility', 'Max Drawdown']

    # Define the colors for the bars
    colors = ['firebrick', 'cornflowerblue', 'mediumpurple', 'forestgreen', 'yellowgreen']

    # Create the bar traces for each portfolio
    traces = []
    for i, portfolio in enumerate(portfolios):
        traces.append(go.Bar(
            x = categories,
            y = np.round(period_dfs['5 years'].iloc[i], 4),
            name = portfolio,
            marker = dict(color = colors[i],
                          line = dict(color = 'black',
                                      width = 1)),
        ))

    # Create the figure
    fig = go.Figure(data = traces)

    # Customize the layout
    fig.update_layout(
        title = 'Comparison of Portfolio Strategies',
        xaxis = dict(title = 'Metrics', tickangle = 45),
        yaxis = dict(title = 'Values'),
        barmode = 'group',
        width = 1200,
        height = 600
    )

    st.plotly_chart(fig)
    
    
    
    st.markdown("#### 10 Year Portfolio Performances:")
    st.dataframe(period_dfs['10 years'])  # Access the dataframe for 10 year period

    # Define the portfolio names
    portfolios = ['Pure SPX', 'Rebalanced 60/40', 'Rebalanced Equal Weighted SAA', 'Rebalanced Ray Dalio Portfolio', 'Rebalanced SAA']

    # Extract data for each portfolio
    annualized_return = period_dfs['10 years']['Annualized Return']
    portfolio_beta = period_dfs['10 years']['Portfolio Beta']
    sharpe_ratio = period_dfs['10 years']['Sharpe Ratio']
    volatility = period_dfs['10 years']['Volatility']
    max_drawdown = period_dfs['10 years']['Max Drawdown']

    # Define the categories
    categories = ['Annualized Return', 'Portfolio Beta', 'Sharpe Ratio', 'Volatility', 'Max Drawdown']

    # Define the colors for the bars
    colors = ['firebrick', 'cornflowerblue', 'mediumpurple', 'forestgreen', 'yellowgreen']

    # Create the bar traces for each portfolio
    traces = []
    for i, portfolio in enumerate(portfolios):
        traces.append(go.Bar(
            x = categories,
            y = np.round(period_dfs['10 years'].iloc[i], 4),
            name = portfolio,
            marker = dict(color = colors[i],
                          line = dict(color = 'black',
                                      width = 1)),
        ))

    # Create the figure
    fig = go.Figure(data = traces)

    # Customize the layout
    fig.update_layout(
        title = 'Comparison of Portfolio Strategies',
        xaxis = dict(title = 'Metrics', tickangle = 45),
        yaxis = dict(title = 'Values'),
        barmode = 'group',
        width = 1200,
        height = 600
    )

    st.plotly_chart(fig)
    
    
    
    st.markdown("#### 20 Year Portfolio Performances:")
    st.dataframe(period_dfs['20 years'])  # Access the dataframe for 20 year period

    # Define the portfolio names
    portfolios = ['Pure SPX', 'Rebalanced 60/40', 'Rebalanced Equal Weighted SAA', 'Rebalanced Ray Dalio Portfolio', 'Rebalanced SAA']

    # Extract data for each portfolio
    annualized_return = period_dfs['20 years']['Annualized Return']
    portfolio_beta = period_dfs['20 years']['Portfolio Beta']
    sharpe_ratio = period_dfs['20 years']['Sharpe Ratio']
    volatility = period_dfs['20 years']['Volatility']
    max_drawdown = period_dfs['20 years']['Max Drawdown']

    # Define the categories
    categories = ['Annualized Return', 'Portfolio Beta', 'Sharpe Ratio', 'Volatility', 'Max Drawdown']

    # Define the colors for the bars
    colors = ['firebrick', 'cornflowerblue', 'mediumpurple', 'forestgreen', 'yellowgreen']

    # Create the bar traces for each portfolio
    traces = []
    for i, portfolio in enumerate(portfolios):
        traces.append(go.Bar(
            x = categories,
            y = np.round(period_dfs['20 years'].iloc[i], 4),
            name = portfolio,
            marker = dict(color = colors[i],
                          line = dict(color = 'black',
                                      width = 1)),
        ))

    # Create the figure
    fig = go.Figure(data = traces)

    # Customize the layout
    fig.update_layout(
        title = 'Comparison of Portfolio Strategies',
        xaxis = dict(title = 'Metrics', tickangle = 45),
        yaxis = dict(title = 'Values'),
        barmode = 'group',
        width = 1200,
        height = 600
    )

    st.plotly_chart(fig)
  