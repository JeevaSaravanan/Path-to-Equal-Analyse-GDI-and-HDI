import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)


st.subheader('Human Development Index')
st.write('The Human Development Index (HDI) is a summary measure of average achievement in key dimensions of human development: a long and healthy life, being knowledgeable and having a decent standard of living. The HDI is the geometric mean of normalized indices for each of the three dimensions.')
hdi_road_map_img = Image.open('Images/hdiRoadMap.png')
st.image(hdi_road_map_img,"HDI Dimensions & Indicator")

st.divider()

st.subheader('Gender Development Index')
st.write('The Gender Development Index (GDI) is a composite measure designed to assess gender disparities and inequalities in a society by considering factors related to human development. It is an extension of the Human Development Index (HDI) and focuses on three key dimensions: health, education, and income. In the GDI, these dimensions are assessed separately for males and females, allowing for a comparison of gender-based development gaps. Health indicators typically include life expectancy at birth for both genders. Education indicators encompass literacy rates and enrollment in primary, secondary, and tertiary education for both males and females. The income component typically examines income levels and workforce participation for both genders.')
gdi_img = Image.open('Images/gdi.png')
st.image(gdi_img,"Calculating the GDI—graphical presentation")