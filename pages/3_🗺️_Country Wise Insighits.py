import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
from plotly.express import choropleth
import altair as alt

st.set_page_config(page_title="Country Wise Insighits", page_icon="🗺️")
hdr_data = pd.read_csv("HDR_Data.csv")

st.subheader("HUMAN DEVELOPMENT INSIGHTS - HDI Rank(2021)")
st.write("Access and explore human development data for 191 countries and territories worldwide.")

map_fig=choropleth(data_frame=hdr_data, locations='iso3', color='country', hover_name='hdi_rank_2021')
map_fig.update_geos(fitbounds="locations", visible=False)
map_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(map_fig,use_container_width=True)
