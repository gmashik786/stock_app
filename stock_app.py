from seaborn.distributions import kdeplot
import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import altair as alt
import random
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
#from my_utils import *
from datetime import datetime
#from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objs as go
#import tensorflow as tf
from plotly import __version__ 
import pickle
import os
import requests

#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.markdown("<h2 style='text-align: center; color: red;'><b>Financial Data analysis and Stock Price Prediction</b></h2>", unsafe_allow_html=True)

option = st.selectbox('Select institution name',('All institutions','Bank of America', 'CitiGroup','Goldman Sachs', 'JPMorgan Chase','Morgan Stanley','Apple','Google'))
today=datetime.now()
y=today.year
syear=st.slider('select year from where you want to start:',2000,y,value=2000)
start = datetime(syear, 1, 1)
end = today
#Bank of America
#BAC = data.DataReader("BAC", 'yahoo', start, end)
d1=yf.Ticker("BAC")
BAC=d1.history(period="1d",start=start,end=end)
# CitiGroup
d2=yf.Ticker("C")
C=d2.history(period="1d",start=start,end=end)
#C = data.DataReader("C", 'yahoo', start, end)
# Goldman Sachs
d3=yf.Ticker("GS")
GS=d3.history(period="1d",start=start,end=end)
#GS = data.DataReader("GS", 'yahoo', start, end)
# JPMorgan Chase
#JPM = data.DataReader("JPM", 'yahoo', start, end)
d4=yf.Ticker("JPM")
JPM=d4.history(period="1d",start=start,end=end)
# Morgan Stanley
#MS = data.DataReader("MS", 'yahoo', start, end)
d5=yf.Ticker("MS")
MS=d5.history(period="1d",start=start,end=end)
# Apple
#APPL = data.DataReader("AAPL", 'yahoo', start, end)
d6=yf.Ticker("AAPL")
APPL=d6.history(period="1d",start=start,end=end)
#google
d7=yf.Ticker("GOOGL")
GOOGL=d7.history(period="1d",start=start,end=end)
keys = ['BAC', 'C', 'GS', 'JPM', 'MS', 'APPLE','GOOGL']
di={'Bank of America':'BAC', 'Goldman Sachs':'GS', 'CitiGroup':'C','JPMorgan Chase':'JPM','Morgan Stanley':'MS','Apple':'APPLE','Google':'GOOGL'}
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, APPL,GOOGL],axis=1,keys=keys)
bank_stocks.columns.names = ['Institute Name','Stock Info']
#st.write(bank_stocks)
#st.write("## What is the max Close price for each bank's stock throughout the time period?")
#df1 = pd.DataFrame(data=bank_stocks.xs(key='Close',axis=1,level='Stock Info').max())
#st.write(df1)
# df1.columns = ['Close price']
# returns=pd.DataFrame()
creturns=pd.DataFrame()
# for name in keys:
#   returns[name+" Return"]=bank_stocks[name]['Close'].pct_change()
for name in keys:
    creturns[name]=bank_stocks[name]['Close']
com_list=['Bank of America', 'CitiGroup','Goldman Sachs', 'JPMorgan Chase','Morgan Stanley','Apple','Google']
if option=="All institutions":
    #st.line_chart(creturns)
    st.header("Stock price history for all institutions from "+str(syear)+" to 2021")
    creturns=creturns.reset_index()
    clr=['red','blue','violet','green','yellow','purple','black']
    
    fig = go.Figure([
    go.Scatter(
    name=com_list[0],
    x=creturns['Date'],
    y=creturns[di[com_list[0]]],
    mode='markers+lines',
    marker=dict(color=clr[0], size=2),
    showlegend=True
    ),
    go.Scatter(
    name=com_list[1],
    x=creturns['Date'],
    y=creturns[di[com_list[1]]],
    mode='markers+lines',
    marker=dict(color=clr[1], size=2),
    showlegend=True
    ),
    go.Scatter(
    name=com_list[2],
    x=creturns['Date'],
    y=creturns[di[com_list[2]]],
    mode='markers+lines',
    marker=dict(color=clr[2], size=2),
    showlegend=True
    ),
    go.Scatter(
    name=com_list[3],
    x=creturns['Date'],
    y=creturns[di[com_list[3]]],
    mode='markers+lines',
    marker=dict(color=clr[3], size=2),
    showlegend=True
    ),
    go.Scatter(
    name=com_list[4],
    x=creturns['Date'],
    y=creturns[di[com_list[4]]],
    mode='markers+lines',
    marker=dict(color=clr[4], size=2),
    showlegend=True
    ),
    go.Scatter(
    name=com_list[5],
    x=creturns['Date'],
    y=creturns[di[com_list[5]]],
    mode='markers+lines',
    marker=dict(color=clr[5], size=2),
    showlegend=True
    ),
    go.Scatter(
    name=com_list[6],
    x=creturns['Date'],
    y=creturns[di[com_list[6]]],
    mode='markers+lines',
    marker=dict(color=clr[6], size=2),
    showlegend=True
    )
    ])
    fig.update_layout(
    yaxis_title='',
    title='Closing price',
    hovermode="x",
    width=800, height=400)
    st.plotly_chart(fig)
    fig11, ax11 = plt.subplots()
    ax11=sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)
    st.header("Correlation between different institutions")
    st.pyplot(fig11)
    l1=["Bank of America", "CitiGroup", "Goldman Shacs", "JPMorgan", "Morgan Stanley","Apple","Google"]
    bank_stocks1 = pd.concat([BAC, C, GS, JPM, MS, APPL,GOOGL],axis=1,keys=l1)
    bank_stocks1.columns.names = ['Institute Name','Stock Info']
    s1=bank_stocks1.xs(key='Close',axis=1,level='Stock Info').max()
    temp1={"Price":s1}
    maxd=pd.DataFrame(temp1)#
    ss=maxd.style.background_gradient(cmap ='copper',axis=0)
    st.write("**The maximum Close price for each institution's stock throughout the time period**")
    st.dataframe(ss)
    s23=bank_stocks1.loc['2021-01-01':today].xs(key='Close',axis=1,level='Stock Info').max()
    temp23={"Price":s23}
    maxd23=pd.DataFrame(temp23).style.background_gradient(cmap = 'copper')
    st.write("**The maximum Close price for each institution's stock for the year**",today.year)
    st.dataframe(maxd23)
    sm1=bank_stocks1.xs(key='Close',axis=1,level='Stock Info').min()
    tempmin1={"Price":sm1}
    mind=pd.DataFrame(tempmin1)#
    ss1=mind.style.background_gradient(cmap ='Blues',axis=0)
    st.write("**The minimum Close price for each institution's stock throughout the time period**")
    st.dataframe(ss1)
    smin23=bank_stocks1.loc['2021-01-01':today].xs(key='Close',axis=1,level='Stock Info').min()
    tempmin23={"Price":smin23}
    mind23=pd.DataFrame(tempmin23).style.background_gradient(cmap = 'Blues')
    st.write("**The minimum Close price for each institution's stock for the year**",today.year)
    st.dataframe(mind23)
    returns=pd.DataFrame()
    sa1=bank_stocks1.xs(key='Close',axis=1,level='Stock Info').mean()
    tempavg1={"Price":sa1}
    avgd=pd.DataFrame(tempavg1)#
    ssa1=avgd.style.background_gradient(cmap ='CMRmap',axis=0)
    st.write("**The average Close price for each institution's stock throughout the time period**")
    st.dataframe(ssa1)
    savg23=bank_stocks1.loc['2021-01-01':today].xs(key='Close',axis=1,level='Stock Info').mean()
    tempavg23={"Price":savg23}
    avgd23=pd.DataFrame(tempavg23).style.background_gradient(cmap = 'CMRmap')
    st.write("**The average Close price for each institution's stock for the year**",today.year)
    st.dataframe(avgd23)
    returns=pd.DataFrame()
    for name in l1:
        returns[name]=bank_stocks1[name]['Close'].pct_change()
    s2=returns.std()
    temp2={"Standard Deviation":s2}
    maxd1=pd.DataFrame(temp2).style.background_gradient(cmap = 'Reds')
    st.write("**Take a look at the standard deviation of the returns, which stock would we can classify as the riskiest over the entire time period? (Dark red is the riskiest one)**")
    st.dataframe(maxd1)
    s3=returns.loc['2021-01-01':today].std()
    temp3={"Standard Deviation":s3}
    maxd2=pd.DataFrame(temp3).style.background_gradient(cmap = 'Reds')
    st.write("**Similar analysis for the year",today.year," (Dark red is the riskiest one)**")
    st.dataframe(maxd2)
    st.header("Corelation plot")
    st.write("               ")
    st.pyplot(sns.pairplot(returns[1:]))
    
else:
    st.header("Stock price history of "+ option)
    ins=di[option]
    maxp=creturns[ins].max()
    minp=creturns[ins].min()
    maxi=creturns[ins].idxmax()
    maxi=datetime(maxi.year,maxi.month,maxi.day)
    mini=creturns[ins].idxmin()
    mini=datetime(mini.year,mini.month,mini.day)
    cymaxp=creturns[ins].loc['2021-01-01':today].max()
    cyminp=creturns[ins].loc['2021-01-01':today].min()
    cymaxi=creturns[ins].loc['2021-01-01':today].idxmax()
    cymaxi=datetime(cymaxi.year,cymaxi.month,cymaxi.day)
    cymini=creturns[ins].loc['2021-01-01':today].idxmin()
    cymini=datetime(cymini.year,cymini.month,cymini.day)
    creturns=creturns.reset_index()
    ins=di[option]
    clr=['red','blue','violet','green']
    fig = go.Figure([
    go.Scatter(
        name=option,
        x=creturns['Date'],
        y=creturns[ins],
        mode='markers+lines',
        marker=dict(color=clr[random.randint(0,len(clr)-1)], size=2),
        showlegend=True
    )

    ])
    fig.update_layout(
        yaxis_title='',
        title='Closing Price',
        hovermode="x",
        width=800, height=400)
    st.plotly_chart(fig)
    st.write("** Highest price recorded on ",maxi,"** which is : ",maxp)
    st.write("** Lowest price recorded on ",mini, "** which is : ",minp)
    st.write("** Highest price this year recorded on ",cymaxi, "** which is : ",cymaxp)
    st.write("** Lowest price this year recorded on ",cymini, "** which is : ",cyminp)
    st.header("**Candlestick Chart **")
    #df445=creturns[ins]
    #fig445=df445.ta_plot(study='sma',periods=[14,28,56],title='Simple Moving Averages',asFigure=True)
    if option=='Apple':
        df4456=APPL.reset_index()
    elif option=='Bank of America':
        df4456=BAC.reset_index()
    elif option=='CitiGroup':
        df4456=C.reset_index()
    elif option=='Goldman Sachs':
        df4456=GS.reset_index()
    elif option=='Morgan stanley':
        df4456=MS.reset_index()
    elif option=='JPMorgan Chase':
        df4456=JPM.reset_index()
    else:
        df4456=GOOGL.reset_index()
    df4456=df4456[['Date','Open', 'High', 'Low', 'Close']]
    fig4456 = go.Figure(data=[go.Candlestick(x=df4456['Date'],
                open=df4456['Open'],
                high=df4456['High'],
                low=df4456['Low'],
                close=df4456['Close'])])
    st.plotly_chart(fig4456)
    st.header("**Moving Average **")
    fig123 = go.Figure([
    go.Scatter(
        name="Original curve",
        x=creturns['Date'],
        y=creturns[ins],
        mode='markers+lines',
        marker=dict(color="black", size=2),
        showlegend=True
    ),
    go.Scatter(
        name="7-days moving average",
        x=creturns['Date'],
        y=creturns[ins].rolling(7).mean(),
        mode='markers+lines',
        marker=dict(color="red", size=2),
        showlegend=True
    ),
    go.Scatter(
        name="14-days moving average",
        x=creturns['Date'],
        y=creturns[ins].rolling(14).mean(),
        mode='markers+lines',
        marker=dict(color="green", size=2),
        showlegend=True
    ),
    go.Scatter(
        name="28-days moving average",
        x=creturns['Date'],
        y=creturns[ins].rolling(28).mean(),
        mode='markers+lines',
        marker=dict(color="blue", size=2),
        showlegend=True
    ),
    go.Scatter(
        name="56-days moving average",
        x=creturns['Date'],
        y=creturns[ins].rolling(56).mean(),
        mode='markers+lines',
        marker=dict(color="brown", size=2),
        showlegend=True
    )

    ])
    fig123.update_layout(
        yaxis_title='',
        title='Moving Average',
        hovermode="x",
        width=800, height=400)
    st.plotly_chart(fig123)
if st.button('Show Data'):
    st.header('Price in Tabular Form')
    if option=='All institutions':
        st.dataframe(creturns)
    else:
        st.dataframe(creturns[['Date',di[option]]])
if option != "All institutions":
    returns=pd.DataFrame()
    for name in keys:
        returns[name]=bank_stocks[name]['Close'].pct_change()
    fig1101, ax = plt.subplots()
    st.header("**Stock price return distribution over the whole time period**")
    ax=sns.distplot(returns.loc[start:today][di[option]],color=clr[random.randint(0,len(clr)-1)],bins=70)
    st.pyplot(fig1101)
    fig1102, ax3 = plt.subplots()
    st.header("**Stock price return distribution for the current year**")
    ax3=sns.distplot(returns.loc['2021-01-01':today][di[option]],color=clr[random.randint(0,len(clr)-1)],bins=70)
    st.pyplot(fig1102)
st.markdown("<h2 style='text-align: center; color: red;'><b>Stock Price Prediction</b></h2>", unsafe_allow_html=True)
pr_option= st.selectbox('Select institution for which you want to predict the stock price:',('Select an institution','Bank of America', 'Apple', 'Google','CitiGroup','Goldman Sachs', 'JPMorgan Chase','Morgan Stanley'))
preiod=st.slider('How many days you want to predict:',365,545,value=365)
if pr_option=='Select an institution':
    st.write('**Please select the rediction length from slider and then select an institution to see rediction**')


if pr_option!="Select an institution":
    if pr_option=="Bank of America":
        data=BAC.drop(['High', 'Low', 'Open', 'Volume', 'Dividends','Stock Splits'], axis=1)
    if pr_option=="CitiGroup":
        data=C.drop(['High', 'Low', 'Open', 'Volume', 'Dividends','Stock Splits'], axis=1)
    if pr_option=="Goldman Sachs":
        data=GS.drop(['High', 'Low', 'Open', 'Volume', 'Dividends','Stock Splits'], axis=1)
    if pr_option=="JPMorgan Chase":
        data=JPM.drop(['High', 'Low', 'Open', 'Volume', 'Dividends','Stock Splits'], axis=1)
    if pr_option=="Morgan Stanley":
        data=MS.drop(['High', 'Low', 'Open', 'Volume', 'Dividends','Stock Splits'], axis=1)
    if pr_option=="Apple":
        data=APPL.drop(['High', 'Low', 'Open', 'Volume', 'Dividends','Stock Splits'], axis=1)
    if pr_option=="Google":
        data=GOOGL.drop(['High', 'Low', 'Open', 'Volume', 'Dividends','Stock Splits'], axis=1)

    data.reset_index(inplace=True)
    data['ds']=data['Date']
    data['y']=data['Close']
    data.drop(['Close','Date'],axis=1,inplace=True)
    m = Prophet(interval_width=0.95)
    m.fit(data)
    future = m.make_future_dataframe(periods=preiod)
    forecast = m.predict(future)
    fig_fin= plot_plotly(m, forecast)
    st.plotly_chart(fig_fin)
    st.write("** General, Yearly and Weekly Trend**")
    fig_fin1=plot_components_plotly(m, forecast)
    st.plotly_chart(fig_fin1)
#     t11=[i for i in range(len(data))]
#     ser11=list(data["Close"])
#     series = np.array(ser11)
#     time = np.array(t11)
#     #K.set_session(session)
#     @st.cache
#     def forecast(steps,window_size=18):
#     #forecast
#         _based_series = series[-window_size:]
#         results = []
#         for i in range(steps):
#             _based_series = _based_series[-window_size:]
#             _r = os_model_forecast(model, _based_series.reshape(-1,1),window_size)[-1,-1]
#             results.append(_r[0])
#             _based_series = np.append(_based_series,_r)
#     #convert back to normal scale
#         results = np.array(results)
#         results = results.reshape(-1,1).reshape(-1,)
#         return results
#     res=forecast(preiod,18)
#     res=list(res)
#     plt.figure(figsize=(16, 6))
    
#     x1=[i for i in range(len(time),len(time)+7)]
#     fig_fin = go.Figure([
#         go.Scatter(
#             name="Real data",
#             x=time[len(time)-20:len(time)],
#             y=series[len(time)-20:len(time)],
#             mode='markers+lines',
#             marker=dict(color="yellow", size=2),
#             showlegend=True
#         ),
#         go.Scatter(
#             name="Predicted data",
#             x=x1,
#             y=res,
#             mode='markers+lines',
#             marker=dict(color="blue", size=2),
#             showlegend=True
#         )

#         ])
#     fig_fin.update_layout(
#             yaxis_title='',
#             title='Closing Price',
#             hovermode="x",
#             width=800, height=400)
#     st.plotly_chart(fig_fin)
# fig = go.Figure([
#     go.Scatter(
#         name='City group',
#         x=creturns['Date']
#         y=creturns['C'],
#         mode='markers+lines',
#         marker=dict(color='red', size=2),
#         showlegend=True
#     ),
#     go.Scatter(
#         name='Morganstanley',
#         x=creturns['Date'],
#         y=creturns['MS'],
#         mode='lines',
#         marker=dict(color="#444"),
#         line=dict(width=1),
#         showlegend=True
#     )
# ])
# fig.update_layout(
#     yaxis_title='',
#     title='Closing price',
#     hovermode="x",
#     width=800, height=400
# )
# 
#st.write(creturns.columns)
# st.write("Corelation plot")
# st.write("               ")
# st.pyplot(sns.pairplot(returns[1:]))
# st.write(" **On the dates below each institue stock had the worst single day returns. We can see that 3 of the banks share the same day for the worst drop, did anything significant happen that day?** ")
# st.write(returns.idxmax())
# st.write("** Take a look at the standard deviation of the returns, which stock would we can classify as the riskiest over the entire time period?**")
# st.write(returns.std())
# st.write(" **we can classify Citigroup as the riskiest over the entire time period.** ")
# st.write(" **Similar Analysis fo this year: **")
# st.write(returns.loc[l:today].std())
# #fig11=bank_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()
# st.write(""" ## Different comapies stock price (closing price)
# """)
# st.line_chart(creturns)
# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(returns['2020-01-01':today], bins=100,density=True)[0]
# st.bar_chart(hist_values)
# fig1101, ax = plt.subplots()
# ax=sns.distplot(returns.loc['2020-01-01':today]['MS Return'],color='green',bins=70)
# st.pyplot(fig1101)
# newdf=bank_stocks.xs(key='Close',axis=1,level='Stock Info')
# newdf.head()
# fig = go.Figure([
    
#     go.Scatter(
#         name='City group',
#         y=newdf['C'],
#         mode='markers+lines',
#         marker=dict(color='red', size=2),
#         showlegend=True
#     ),
#     go.Scatter(
#         name='Morganstanley',
#         y=newdf['MS'],
#         mode='lines',
#         marker=dict(color="#444"),
#         line=dict(width=1),
#         showlegend=True
#     )
# ])
# fig.update_layout(
#     yaxis_title='',
#     title='Closing price',
#     hovermode="x"
# )
# st.plotly_chart(fig)
