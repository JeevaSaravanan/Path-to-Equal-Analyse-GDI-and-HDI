import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
from plotly.express import choropleth
import altair as alt
import plotly.graph_objects as go
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from statsmodels.tsa.arima.model import ARIMA


st.set_page_config(page_title="Prediction and Classification of HDI", page_icon="👾📊📈", layout="wide")

st.markdown("<h1 style='text-align: center;'>The Path to Equal</h1>",unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col3:
    st.caption("~ Made by Jeeva Saravana Bhavanandam")
st.subheader("", divider='rainbow')

shdi = pd.read_csv('SHDI-SGDI-Total 7.0.csv')

columns_to_drop = ['iso_code', 'GDLCODE','level','sgdi','shdif','shdim','healthindex','healthindexf','healthindexm','incindex', 'incindexf', 'incindexm','edindexf', 'edindexm','eschf', 'eschm','mschf', 'mschm','lifexpf', 'lifexpm','gnicf','gnicm','lgnic', 'lgnicf', 'lgnicm', 'pop']
shdi.drop(columns=columns_to_drop, inplace=True)

shdi['shdi'] = pd.to_numeric(shdi['shdi'], errors='coerce')
shdi['edindex'] = pd.to_numeric(shdi['edindex'], errors='coerce')
shdi['esch'] = pd.to_numeric(shdi['esch'], errors='coerce')
shdi['msch'] = pd.to_numeric(shdi['msch'], errors='coerce')
shdi['lifexp'] = pd.to_numeric(shdi['lifexp'], errors='coerce')
shdi['gnic'] = pd.to_numeric(shdi['gnic'], errors='coerce')

def categorise_hdi(row):
    shdi_value = row['shdi']
    if (shdi_value >= 0.8):
        return "Very High"
    elif(shdi_value>=0.7):
        return "High"
    elif(shdi_value>=0.550):
        return "Medium"
    else:
        return "Low"
shdi['hdi_category'] = shdi.apply(categorise_hdi, axis=1)

country_list = shdi["country"].unique()

st.subheader("Prediction with RandomForest")

col1, col2,col3 = st.columns(3)
with col1:
    selected_country= st.selectbox('Country', country_list)
    region_list = shdi[(shdi["country"]==selected_country)]["region"].unique()
with col2:
    selected_region= st.selectbox('Region',region_list)

df = shdi[(shdi["country"]==selected_country) & (shdi['region']==selected_region)]


st.write('features to be considered for prediction')



col1, col2 = st.columns(2)
# Create Figure
box=[]
feature_list=[ 'edindex','esch','msch','lifexp','gnic']
with col2:
    st.write("")
    st.write("")
    st.write("")
    for featue in feature_list:
        box.append(st.checkbox(featue,value=True))
    selected_values = [value for value, is_true in zip(feature_list, box) if is_true]

with col1:
    features = df[selected_values+['year']]
   
    X_train, X_test, y_train, y_test = features[features["year"]<2017], features[features["year"]>=2010] , df[df['year']<2017]['shdi'],df[df['year']>=2010]['shdi']
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)

    dict_predictions = {"year":list(range(2010,2022)),"shdi":rf_predictions}

    df_predictions = pd.DataFrame(dict_predictions)

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    print(f'Random Forest RMSE: {rf_rmse}')



    # Create Line Plot 1
    trace1 = go.Scatter(x=df['year'], y=df['shdi'], mode='lines', name='HDI - Actual', line=dict(color='lightblue'))

    # Create Line Plot 2
    trace2 = go.Scatter(x=df_predictions['year'], y=df_predictions['shdi'], mode='lines', line_dash='dash',name='HDI - Predicted', line=dict(color='blue'))

    # Create Layout
    layout = go.Layout(title='HDI - Actual vs Predicted', xaxis=dict(title='Year'), yaxis=dict(title='HDI'))
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


st.subheader("Arima Forecast")


train=df.copy()



arima_model = ARIMA(train['shdi'], order=(5,1,0))
arima_fit = arima_model.fit()

arima_predictions = arima_fit.forecast(steps=8)
dict_preds = {"year":list(range(2021,2029)),"shdi":arima_predictions}

df_preds = pd.DataFrame(dict_preds)

# Create Line Plot 1
trace1 = go.Scatter(x=train['year'], y=train['shdi'], mode='lines', name='HDI - Actual', line=dict(color='lightblue'))

# Create Line Plot 2
trace2 = go.Scatter(x=df_preds['year'], y=df_preds['shdi'], mode='lines', line_dash='dash',name='HDI - Predicted', line=dict(color='blue'))

# Create Layout
layout = go.Layout(title='HDI - Actual vs Predicted', xaxis=dict(title='Year'), yaxis=dict(title='HDI'))

# Create Figure
fig = go.Figure(data=[trace1, trace2], layout=layout)

# Display the Plotly chart in Streamlit
st.plotly_chart(fig, use_container_width=True)
st.write(arima_predictions)
