import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt
import time

with st.spinner('In Progress..'):
    time.sleep(3)

st.subheader("Evaluating Gender Development: Insights from the Gender Development Index")
st.divider()

st.subheader('Overview')
st.write('The Gender Development Index (GDI) is a composite measure designed to assess gender disparities and inequalities in a society by considering factors related to human development. It is an extension of the Human Development Index (HDI) and focuses on three key dimensions: health, education, and income. In the GDI, these dimensions are assessed separately for males and females, allowing for a comparison of gender-based development gaps. Health indicators typically include life expectancy at birth for both genders. Education indicators encompass literacy rates and enrollment in primary, secondary, and tertiary education for both males and females. The income component typically examines income levels and workforce participation for both genders.')

st.divider()

df = pd.read_csv('Gender Development Index.csv')

scale = alt.Scale(
    domain=["Asia", "Africa", "Europe", "America", "Oceania"],
    range=["#e7ba52", "#a7a7a7", "#aec7e8", "#1f77b4", "#9467bd"],
)
color = alt.Color("Continent:N", scale=scale)

brush = alt.selection_interval(encodings=["x"])
click = alt.selection_multi(encodings=["color"])

points = (
    alt.Chart()
    .mark_point()
    .encode(
        alt.X("HDI Rank (2021)", title="HDI Rank (2021)"),
        alt.Y("Human Development Groups",title="Human Development Groups"),
        color=alt.condition(brush, color, alt.value("lightgray")),
        size=alt.Size("HDI Rank (2021):Q", scale=alt.Scale(range=[5, 200])),
    )
    .properties(width=550, height=300)
    .add_selection(brush)
    .transform_filter(click)
)

bars = (
    alt.Chart()
    .mark_bar()
    .encode(
        x="count()",
        y="Continent",
        color=alt.condition(click, color, alt.value("lightgray")),
)
    .transform_filter(brush)
    .properties(
        width=550,
    )
    .add_selection(click)
)


chart = alt.vconcat(points, bars, data=df, title="HDI Rank (2021) based on Human Development Groups")

year_X = []
gdi_Y = []
for i in range(2015,2022):
    ylabel = "Gender Development Index ("+str(i)+")"
    year_X.append(i)
    gdi_Y.append(ylabel)

tab1, tab2, tab3 = st.tabs(["HDI Rank (2021) based on Human Development Groups", "HDI Rank (2021) based vs Gender Development Index (2021)","Gender Development Index over Last 6 years"])
with tab1:
    st.altair_chart(chart, theme="streamlit", use_container_width=True)
    with st.expander("See explanation"):
        st.write("The above chart defines HDI Rank (2021) based on Human Development Groups")
with tab2:
    st.area_chart(df, x="HDI Rank (2021)", y="Gender Development Index (2021)",color="Continent")
    with st.expander("See explanation"):
        st.write("The above chart shows HDI Rank (2021) based vs Gender Development Index (2021)")
with tab3:
    st.line_chart(df,x="HDI Rank (2021)",y=gdi_Y,color="Country")
    with st.expander("See explanation"):
        st.write("Gender Development Index over Last 6 years")

