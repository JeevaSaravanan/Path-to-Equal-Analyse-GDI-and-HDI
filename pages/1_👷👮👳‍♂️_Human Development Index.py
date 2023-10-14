import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px


st.set_page_config(page_title="Human Development Index", page_icon="👷👮👳‍♂️")


st.sidebar.header("Human Development Index")

st.subheader('Human Development Index')
st.write('The Human Development Index (HDI) is a summary measure of average achievement in key dimensions of human development: a long and healthy life, being knowledgeable and having a decent standard of living. The HDI is the geometric mean of normalized indices for each of the three dimensions.')
hdi_road_map_img = Image.open('Images/hdiRoadMap.png')
st.image(hdi_road_map_img,"HDI Dimensions & Indicator")


hdr_data = pd.read_csv("HDR_Data.csv")
st.subheader("Explore HDI")
st.markdown("*ADD COUNTRY TO COMPARE (UP TO 3)*")


fig = px.bar(hdr_data, x="hdi", y="year", color='continent', orientation='h',
                 hover_data=["gnipc", "hdi_rank_2021"],
                 title='Restaurant bills')

st.plotly_chart(fig, theme="streamlit")

col1, col2 = st.columns(2)
with col1:
   selected_year = st.selectbox('Year', list(reversed(range(1990,2021))))





