import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pandas_datareader import data as pdr
import scipy.stats as scs
from pandas.tseries.offsets import MonthEnd
import pandas as pd



st.title("Momentum Portfolio Analysis")

multi = '''The following dashboard was created as a way to expierment with returns and basic analysis of a momentum trading strategy while being able to change various important inputs.

If you encounter any errors, please refresh the webpage or reselect a starting date.


Note: See what happens when you add GME!
'''

st.markdown(multi)
assets = st.text_input("Please select listed tickers ","AAPL, MSFT, GOOGL")
benchmark = st.text_input("Please choose a benchmark ", "^GSPC")


# selections

method_selections = ["Quantile"]
method = st.selectbox("Please choose the portfolio weighting method (equal or quantile)", method_selections)
period = st.number_input("Please input number of periods for momentum calculation", 11)
start  = st.date_input("Pick choose a starting date", pd.to_datetime("2007-01-1"))
amount_invested = st.number_input("Please input an intial investment amount", 1000)

class Portfolio():

    def __init__(self,assets,benchmark,start):
        self.start = start

        # get assets + prices
        self.assets = assets
        self.prices = yf.download(self.assets, start = self.start)['Adj Close']
        self.m_rets = self.monthly_rets()

        # get benchmark + prices
        self.benchmark = benchmark
        self.b_prices = yf.download(self.benchmark, start = self.start)['Adj Close']
        self.b_rets = self.b_prices.pct_change().resample("M").agg(lambda x: (x + 1).prod() - 1)

    def monthly_rets(self):

        m_rets = self.prices.pct_change().resample("M").agg(lambda x: (x + 1).prod() - 1)

        return m_rets
    
    def get_mom(self,period, method = "Quantile",quantile = 0.25):

        m_rets_lb = self.m_rets.rolling(period).agg(lambda x: (x + 1).prod() - 1)
        m_rets_lb.dropna(inplace = True)

        rets_dict = {"Momentum" : [], "Win" : [], "Loss" : []}

        for row in range(len(m_rets_lb) - 1):
        
            # get current month label
            curr = m_rets_lb.iloc[row]

            # winners
            if method == "Equal":
                # winners
                win = curr.nlargest(10)
                # losers
                loss = curr.nsmallest(10)

            elif method == "Quantile":
                # winners
                win = curr[curr > np.quantile(curr,1 - quantile)]
                # losers
                loss = curr[curr < np.quantile(curr,quantile)]

            # winner return 
            win_ret = self.m_rets.loc[win.name + MonthEnd(1), win.index]
            # loser return
            loss_ret = self.m_rets.loc[loss.name + MonthEnd(1),loss.index]

            # append returns
            rets_dict["Win"].append(win_ret.mean())
            rets_dict["Loss"].append(loss_ret.mean())
            rets_dict["Momentum"].append(win_ret.mean() - loss_ret.mean())


        # format  dict into series and take cumulative returns
        rets_dict["Win"] = (pd.Series(rets_dict["Win"], index = m_rets_lb[:len(m_rets_lb) - 1].index) + 1)
        rets_dict["Loss"] = (pd.Series(rets_dict["Loss"], index = m_rets_lb[:len(m_rets_lb) - 1].index) + 1)
        rets_dict["Momentum"] = (pd.Series(rets_dict["Momentum"], index = m_rets_lb[:len(m_rets_lb) - 1].index) + 1)
        rets_dict["Benchmark"] = (pd.Series(self.b_rets, index = m_rets_lb[:len(m_rets_lb) - 1].index) + 1)
        # print(rets_dict)
        # return dictionary
        return pd.DataFrame(rets_dict)

    def get_rolling_beta(self, data, rolling_period = 6):

        # Group by rolling window
        groups = data.rolling(rolling_period).cov().groupby(level=0)

        # Calculate beta
        beta = groups.apply(lambda x: x.iloc[1, 0] / x.iloc[0, 0])
        beta = pd.DataFrame(beta).rename(columns = {0: "Beta"})

        return beta
    
    def get_rolling_sr(self, data, rf_rate = 0.02, rolling_period = 6):

        # Calculate excess returns
        data['excess_return'] = data['Momentum'] - rf_rate

        # Calculate rolling Sharpe ratio
        data['Sharpe Ratio'] = data['excess_return'].rolling(window=rolling_period).mean() / data['excess_return'].rolling(window=rolling_period).std()

        return data["Sharpe Ratio"]
        
p = Portfolio(assets, benchmark= benchmark,start = start)

data = p.get_mom(period = period, method = method)

st.subheader("Portfolio Performance")

# portfolio stats
st.caption("Portfolio Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

b_amt = round(amount_invested * data["Benchmark"].cumprod()[-1])
mom_amt = round(amount_invested * data["Momentum"].cumprod()[-1])
win_amt = round(amount_invested * data["Win"].cumprod()[-1])
loss_amt = round(amount_invested * data["Loss"].cumprod()[-1])

col1.metric("Amount Invested",  "$" + str(amount_invested))
col2.metric("Benchmark Return","$" + str(b_amt), str(round((b_amt/amount_invested - 1) * 100)) + "%")
col3.metric("Momentum Return", "$" + str(mom_amt), str(round((mom_amt/amount_invested - 1) * 100)) + "%")
col4.metric("Winner Return", "$" + str(win_amt), str(round((win_amt/amount_invested - 1) * 100)) + "%")
col5.metric("Loser Return", "$" + str(loss_amt), str(round((loss_amt/amount_invested - 1) * 100)) + "%")


st.caption('Portfolio Returns')
st.line_chart(data = data.cumprod())

# distribution of returns 
st.caption("Return Comparison Across Strategies")
chart_data = data - 1
st.bar_chart(chart_data)

# rolling beta chart
st.caption("Rolling Beta")
rolling_period = st.number_input("Select rolling period for beta calculation", 6)
beta_data = p.get_rolling_beta(data, rolling_period = rolling_period)
st.line_chart(beta_data)

# rolling sharpe ratio 
st.caption("Rolling Sharpe Ratio")
rolling_period = st.number_input("Select rolling period for Sharpe Ratio calculation", 6)
sr_data = p.get_rolling_sr(data,rf_rate = 0.02, rolling_period = rolling_period)

st.line_chart(sr_data)


