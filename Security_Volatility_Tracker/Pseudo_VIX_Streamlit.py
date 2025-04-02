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

import datetime as dt
from datetime import date
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from dateutil.relativedelta import relativedelta


st.title("Sanjeev Narasimhan Daily Pseudo VIX Tracker")

my_portfolio = ['VOO', 'VOOG', 'UNH', 'XOM', 'MSFT', 'AMZN', 'GOOGL']

tabs = st.tabs(my_portfolio)


# ----------------------------------- DEFINING VIX_f -----------------------------------
class VIX_f:
    
    def __init__(self, Ticker, Days):
        self.Ticker = Ticker
        self.Days = Days
        self._cache = {}
        
        
    def Create_VIXf_DF(self):
        # setting timeframe
        today = pd.to_datetime('now')
        timeframe = self.Days
        period_days_ago = today - BDay(timeframe)

        today = today.strftime("%Y-%m-%d")
        period_days_ago = period_days_ago.strftime("%Y-%m-%d")

        # initializing lookback period as n = 20
        n = 20

        # initializing list to store values
        dates = []
        vix_f_values = []
        close_prices = []

        # pull price data prices
        data = yf.download(self.Ticker, start=period_days_ago, end=today)

        data = data.reset_index().rename(columns={"index":"Date"})
        data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

        for i in range(len(data)):
            vix_f_value = 0
            dates.append(data['Date'][i])
            if i >= n:
                vix_f_value = (((data['Close'][(i-n):i+1]).max()) - (data['Low'][i])) / (((data['Close'][(i-n):i+1]).max()))
                vix_f_value = vix_f_value * 100
                vix_f_values.append(vix_f_value)
            else:
                vix_f_values.append(0)

            close_prices.append(data['Close'][i])

        close_prices_Data = pd.DataFrame(list(zip(dates, close_prices)), columns =['Date', 'Close Price'])
        self._cache['Close_Prices_Data'] = close_prices_Data

        vix_f_Data = pd.DataFrame(list(zip(dates, vix_f_values)), columns =['Date', 'VIX_f'])

        vix_f_Data = vix_f_Data[20:]
        vix_f_Data.reset_index()

        vix_f_Data["3_Day_Mov_Avg"] = vix_f_Data['VIX_f'].rolling(window=3).mean()
        vix_f_Data["5_Day_Mov_Avg"] = vix_f_Data['VIX_f'].rolling(window=5).mean()

        vix_f_Data = vix_f_Data[4:]
        vix_f_Data.reset_index()
        self._cache['VIX_f_Data'] = vix_f_Data
        
        return self._cache['VIX_f_Data']


    def Create_VIXf_Chart(self):
        vix_f_Data = self._cache['VIX_f_Data']
        
        # Define the x-axis (dates)
        dates = vix_f_Data['Date']

        # Define the y-axis (VIX_f values)
        VIX_f = vix_f_Data['VIX_f']
        VIX_f_3_Day_Mov_Avg = vix_f_Data['3_Day_Mov_Avg']
        VIX_f_5_Day_Mov_Avg = vix_f_Data['5_Day_Mov_Avg']
        
        figures = []

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x = dates,
            y = VIX_f,
            mode = 'lines',
            name = 'VIX_f',
            line = dict(color = 'grey')
        ))

        fig.add_trace(go.Scatter(
            x = dates,
            y = VIX_f_3_Day_Mov_Avg,
            mode = 'lines',
            name = 'MOV 3',
            line = dict(color = 'teal')
        ))

        fig.add_trace(go.Scatter(
            x = dates,
            y = VIX_f_5_Day_Mov_Avg,
            mode = 'lines',
            name = 'MOV 5',
            line = dict(color = 'darkturquoise')
        ))

        fig.add_hline(y = 8, line_width = 1, line_color = 'red', line_dash = 'dash')

        fig.update_layout(
            title = "Pseudo VIX_f of " + self.Ticker,
            xaxis_title = 'Date',
            yaxis_title = 'VIX_f Values',
            showlegend = True,
            xaxis = dict(tickformat='%m-%d-%Y'),
            width = 1200,
            height = 600
        )

        figures.append(fig)
        return figures
    
    
    def Create_Close_Chart(self):
        close_prices_Data = self._cache['Close_Prices_Data']

        # Define the x-axis (dates)
        dates = close_prices_Data['Date']

        # Define the y-axis
        Close_Price = close_prices_Data['Close Price']
        
        figures = []

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x = dates,
            y = Close_Price,
            mode = 'lines',
            name = 'Close',
            line = dict(color = 'firebrick')
        ))

        fig.update_layout(
            title = "Close Prices of " + self.Ticker,
            xaxis_title = 'Date',
            yaxis_title = 'Close Price',
            showlegend = True,
            xaxis = dict(tickformat='%m-%d-%Y'),
            width = 1200,
            height = 600
        )

        figures.append(fig)
        return figures


    def Run_VIXf_Analysis(self):
        outputs = {'VIX_f_DF' : self.Create_VIXf_DF(),
                   'VIX_f_Chart' : self.Create_VIXf_Chart(),
                   'Close_Chart' : self.Create_Close_Chart()
                   }
        return outputs
    
    
    
# ----------------------------------- DEFINING SVIX -----------------------------------
class SVIX:
    
    def __init__(self, Ticker, Days):
        self.Ticker = Ticker
        self.Days = Days
        self._cache = {}
        
        
    def Create_SVIX_DF(self):
        # setting timeframe
        today = pd.to_datetime('now')
        timeframe = self.Days
        period_days_ago = today - BDay(timeframe)

        today = today.strftime("%Y-%m-%d")
        period_days_ago = period_days_ago.strftime("%Y-%m-%d")

        # initializing lookback period as n = 20
        n = 20

        # initializing list to store values
        dates = []
        svix_values = []
        close_prices = []

        # pull price data prices
        data = yf.download(self.Ticker, start=period_days_ago, end=today)

        data = data.reset_index().rename(columns={"index":"Date"})
        data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

        for i in range(len(data)):
            svix_value = 0
            dates.append(data['Date'][i])
            if i >= n:
                svix_value = ((data['Close'][i]) - (data['Low'][(i-n):i+1].min())) / ((data['High'][(i-n):i+1].max()) - (data['Low'][(i-n):i+1].min()))
                svix_value = svix_value * 100
                svix_values.append(svix_value)
            else:
                svix_values.append(0)

            close_prices.append(data['Close'][i])

        close_prices_Data = pd.DataFrame(list(zip(dates, close_prices)), columns = ['Date', 'Close Price'])
        self._cache['Close_Prices_Data'] = close_prices_Data

        svix_Data = pd.DataFrame(list(zip(dates, svix_values)), columns =['Date', 'SVIX'])

        svix_Data = svix_Data[20:]
        svix_Data.reset_index()

        svix_Data["3_Day_Mov_Avg"] = svix_Data['SVIX'].rolling(window=3).mean()
        svix_Data["5_Day_Mov_Avg"] = svix_Data['SVIX'].rolling(window=5).mean()

        svix_Data = svix_Data[4:]
        svix_Data.reset_index()
        self._cache['SVIX_Data'] = svix_Data
        
        return self._cache['SVIX_Data']


    def Create_SVIX_Chart(self):
        svix_Data = self._cache['SVIX_Data']
        
        # Define the x-axis (dates)
        dates = svix_Data['Date']

        # Define the y-axis (SVIX values)
        SVIX = svix_Data['SVIX']
        SVIX_3_Day_Mov_Avg = svix_Data['3_Day_Mov_Avg']
        SVIX_5_Day_Mov_Avg = svix_Data['5_Day_Mov_Avg']
        
        figures = []

        # Plotting
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x = dates,
            y = SVIX,
            mode = 'lines',
            name = 'SVIX',
            line = dict(color = 'grey')
        ))

        fig.add_trace(go.Scatter(
            x = dates,
            y = SVIX_3_Day_Mov_Avg,
            mode = 'lines',
            name = 'MOV 3',
            line = dict(color = 'teal')
        ))

        fig.add_trace(go.Scatter(
            x = dates,
            y = SVIX_5_Day_Mov_Avg,
            mode = 'lines',
            name = 'MOV 5',
            line = dict(color = 'darkturquoise')
        ))

        fig.add_hline(y = 80, line_width = 1, line_color = 'green', line_dash = 'dash')
        fig.add_hline(y = 20, line_width = 1, line_color = 'red', line_dash = 'dash')


        fig.update_layout(
            title = "Pseudo SVIX of " + self.Ticker,
            xaxis_title = 'Date',
            yaxis_title = 'SVIX Values',
            showlegend = True,
            xaxis = dict(tickformat='%m-%d-%Y'),
            width = 1200,
            height = 600
        )

        figures.append(fig)
        return figures
    
    
    def Create_Close_Chart(self):
        close_prices_Data = self._cache['Close_Prices_Data']

        # Define the x-axis (dates)
        dates = close_prices_Data['Date']

        # Define the y-axis
        Close_Price = close_prices_Data['Close Price']
        
        figures = []

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x = dates,
            y = Close_Price,
            mode = 'lines',
            name = 'Close',
            line = dict(color = 'firebrick')
        ))

        fig.update_layout(
            title = "Close Prices of " + self.Ticker,
            xaxis_title = 'Date',
            yaxis_title = 'Close Price',
            showlegend = True,
            xaxis = dict(tickformat='%m-%d-%Y'),
            width = 1200,
            height = 600
        )

        figures.append(fig)
        return figures


    def SVIX_20_Day(self):
        today = pd.to_datetime('now')
        twenty_days_ago = today - pd.DateOffset(days=20)

        today = today.strftime("%Y-%m-%d")
        twenty_days_ago = twenty_days_ago.strftime("%Y-%m-%d")

        # pull price data prices
        data = yf.download(self.Ticker, start=twenty_days_ago, end=today)

        data = data.reset_index().rename(columns={"index":"Date"})
        data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        
        self._cache['SVIX_20_Day_DF'] = data
        return self._cache['SVIX_20_Day_DF']
        

    def Run_SVIX_Analysis(self):
        outputs = {'SVIX_DF' : self.Create_SVIX_DF(),
                   'SVIX_Chart' : self.Create_SVIX_Chart(),
                   'Close_Chart' : self.Create_Close_Chart(),
                   'SVIX_20_Day_DF' : self.SVIX_20_Day()
                   }
        return outputs
    
    
    
    
security_counter = 0
 
for tab, security in zip(tabs, my_portfolio):
    
    with tab:
        security_counter +=1
        class_counter = 0
        
        st.header(f'{security} Overview')
        
        st.divider()
        st.markdown("## VIX_f")
        
        # Running Class
        Analysis = VIX_f(security, 101)
        VIX_f_Outputs = Analysis.Run_VIXf_Analysis()
        
        st.markdown("### VIX_f Data")
        st.dataframe(VIX_f_Outputs['VIX_f_DF'])

        st.markdown("### VIX_f Chart")
        charts = VIX_f_Outputs['VIX_f_Chart']
        for fig in charts:
            st.plotly_chart(
                figure_or_data = fig,
                key = f'Key_{security_counter}_{class_counter}')
            
        class_counter += 1

        st.markdown("### Close Price Chart")
        charts = VIX_f_Outputs['Close_Chart']
        for fig in charts:
            st.plotly_chart(
                figure_or_data = fig,
                key = f'Key_{security_counter}_{class_counter}')
            
        class_counter += 1    
                   
        st.markdown("### VIX_f Recommendation")
        vix_f_Data = VIX_f_Outputs['VIX_f_DF']
        if (vix_f_Data['VIX_f'].iloc[-1] < 8) and (vix_f_Data['3_Day_Mov_Avg'].iloc[-1] < 8) and (vix_f_Data['5_Day_Mov_Avg'].iloc[-1] < 8):
            st.write("VIX_F: ", np.round(vix_f_Data['VIX_f'].iloc[-1], 4))
            st.write("VIX_F 3 Day Moving Average: ", np.round(vix_f_Data['3_Day_Mov_Avg'].iloc[-1], 4))
            st.write("VIX_F 5 Day Moving Average: ", np.round(vix_f_Data['5_Day_Mov_Avg'].iloc[-1], 4))
        elif (vix_f_Data['VIX_f'].iloc[-1] > 8) and (vix_f_Data['3_Day_Mov_Avg'].iloc[-1] > 8) and (vix_f_Data['5_Day_Mov_Avg'].iloc[-1] < 8):
            st.write("VIX_F: ", np.round(vix_f_Data['VIX_f'].iloc[-1], 4))
            st.write("VIX_F 3 Day Moving Average: ", np.round(vix_f_Data['3_Day_Mov_Avg'].iloc[-1], 4))
            st.write("VIX_F 5 Day Moving Average: ", np.round(vix_f_Data['5_Day_Mov_Avg'].iloc[-1], 4))
        else:
            st.write("VIX_F: ", np.round(vix_f_Data['VIX_f'].iloc[-1], 4))
            st.write("VIX_F 3 Day Moving Average: ", np.round(vix_f_Data['3_Day_Mov_Avg'].iloc[-1], 4))
            st.write("VIX_F 5 Day Moving Average: ", np.round(vix_f_Data['5_Day_Mov_Avg'].iloc[-1], 4))
            
        
        st.divider()
        st.divider()
            
            
        st.markdown("## SVIX")
        
        # Running Class
        Analysis = SVIX(security, 101)
        SVIX_Outputs = Analysis.Run_SVIX_Analysis()
        
        st.markdown("### SVIX Data")
        st.dataframe(SVIX_Outputs['SVIX_DF'])

        st.markdown("### SVIX Chart")
        charts = SVIX_Outputs['SVIX_Chart']
        for fig in charts:
            st.plotly_chart(
                figure_or_data = fig,
                key = f'Key_{security_counter}_{class_counter}')
            
        class_counter += 1
            
            
        st.markdown("### Close Price Chart")
        charts = SVIX_Outputs['Close_Chart']
        for fig in charts:
            st.plotly_chart(
                figure_or_data = fig,
                key = f'Key_{security_counter}_{class_counter}')
            
        class_counter += 1
            
        st.markdown("### SVIX Recommendation")
        
        data = SVIX_Outputs['SVIX_20_Day_DF']
        
        today = pd.to_datetime('now')
        twenty_days_ago = today - pd.DateOffset(days=20)
        today = today.strftime("%Y-%m-%d")
        twenty_days_ago = twenty_days_ago.strftime("%Y-%m-%d")
        
        st.write("Todays Date: ", today)
        st.write("Date 20 Days Ago: ", twenty_days_ago)

        curr_close = data['Close'].to_list()[-1]
        st.write("Current Close Price: ", np.round(curr_close, 4))

        twenty_day_high = data['High'].max()
        st.write("20 Day High: ", np.round(twenty_day_high, 4))

        twenty_day_low = data['Low'].min()
        st.write("20 Day Low: ", np.round(twenty_day_low, 4))

        svix_value = ((data['Close'].to_list()[-1]) - (data['Low'].min())) / ((data['High'].max()) - (data['Low'].min()))
        svix_value = svix_value * 100

        st.write("SVIX Value: ", np.round(svix_value, 4))






















        
        
