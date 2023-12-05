import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import altair as alt

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)




st.markdown("<h1 style='text-align: center;'>The Path to Equal</h1>",unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col3:
    st.caption("~Made by Jeeva Saravana Bhavanandam")
st.subheader("", divider='rainbow')

tab1, tab2, tab3 = st.tabs(["Origins: Unveiling the Background and Purpose", "The Dataset","The App's Significance: Why Pursuing It is Worthwhile"])

with tab1:
    st.subheader('Origins: Unveiling the Background and Purpose')
    st.markdown("""
    ***Goal:***
    The goal of this app is to analyze the Gender Development Index (GDI) dataset to understand the extent of gender disparities in human development across different countries. we can analyse how do gender disparities in health and education impact overall human development and improvement over these years
    """)
    st.write('In today\'s world, we grapple with an unprecedented "uncertainty complex" characterized by the destabilizing forces of the Anthropocene, the pursuit of transformative societal changes, and increasing polarization. The ongoing COVID-19 pandemic, global environmental crises, and rising inequalities have fueled this uncertainty, impeding human development and creating a sense of insecurity. For the first time, the global Human Development Index (HDI) has declined for two consecutive years, while democratic backsliding has worsened. Despite the perils of these new uncertainties, there is promise in reimagining our future, adapting our institutions, and crafting new narratives. This presents an opportunity to thrive in a rapidly changing world.')
    video_file = open('hdr21-22_animation.mp4 (1080p).mp4', 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)
    st.caption("Reference : Video Referred from HDRO video for HDR21-22")

with tab2:
    st.subheader("Dataset:")
    st.markdown("""
    - ***Kaggle Dataset:*** https://www.kaggle.com/datasets/iamsouravbanerjee/gender-development-index-dataset/data

    **Description:**
    This dataset provides comprehensive historical data on gender development indicators at a global level. It includes essential columns such as ISO3 (the ISO3 code for each country/territory), Country (the name of the country or territory), Continent (the continent where the country is located), Hemisphere (the hemisphere in which the country is situated), Human Development Groups, UNDP Developing Regions, HDI Rank (2021) representing the Human Development Index Rank for the year 2021, and Gender Development Index spanning from 1990 to 2021.

    - ***United Nations Development Program HDI Reports:*** https://hdr.undp.org/data-center/documentation-and-downloads

    **Description:**

    - Life expectancy at birth: UNDESA (2022a).
    - Expected years of schooling: CEDLAS and World Bank (2022), ICF Macro Demographic and Health Surveys (various years), UNESCO Institute for Statistics (2022) and United Nations Children’sFund (UNICEF) Multiple Indicator Cluster Surveys(various years).
    - Mean years of schooling for adults ages 25 and older: Barro and Lee (2018), ICF Macro Demographic and Health Surveys (various years), OECD (2022),UNESCO Institute for Statistics (2022) and UNICEF Multiple Indicator Cluster Surveys (various years).
    - Estimated earned income: Human Development Report Office estimates based on female and male shares of the economically active population, the ratio of the female to male wage in all sectors and gross national income in 2017 purchasing power parity (PPP) terms, and female and male shares of population from ILO (2022), IMF (2022), UNDESA(2022a), United Nations Statistics Division (2022) and World Bank (2022).

    """)

with tab3:
    st.subheader("")
    st.markdown("""
    - This app is worthy of completion because it addresses a critical issue – gender disparities in human development – which has real-world implications for policymaking and social progress. By analyzing the GDI dataset and creating an accessible web app, we can contribute to the ongoing efforts to promote gender equality and improve human development outcomes globally.

    - **THE PATH TO GENDER EQUALITY:** This expands the measures for women and girls to exercise their potential, their opportunities and the choices available to them. Policies that seek to further empower women and girls and achieve gender parity require robust data and measures that are comparable across countries and based on a sound methodology. While some such measures are available, the picture has been incomplete.
    """)