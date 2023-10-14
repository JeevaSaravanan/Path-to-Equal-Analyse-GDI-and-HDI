import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import altair as alt

st.set_page_config(page_title="Gender Development Index", page_icon="👩‍🎓🧕👩‍⚕️")


def data_with_filtered(deathRateData,array):
    result=deathRateData
    for column in array:
        if(len(column)==2):
            name,compare=column[0],column[1]
            if(compare=='All'):
                continue
            if(type(compare) == list):
                for _ in compare:
                    result =result[result[name]==_]
            else:
                result =result[result[name]==compare]
    return result

st.sidebar.header("Gender Development Index")

st.subheader('Gender Development Index')
st.write('The Gender Development Index (GDI) is a composite measure designed to assess gender disparities and inequalities in a society by considering factors related to human development. It is an extension of the Human Development Index (HDI) and focuses on three key dimensions: health, education, and income. In the GDI, these dimensions are assessed separately for males and females, allowing for a comparison of gender-based development gaps. Health indicators typically include life expectancy at birth for both genders. Education indicators encompass literacy rates and enrollment in primary, secondary, and tertiary education for both males and females. The income component typically examines income levels and workforce participation for both genders.')
gdi_img = Image.open('Images/gdi.png')
st.image(gdi_img,"Calculating the GDI—graphical presentation")

st.divider()

df = pd.read_csv('Gender Development Index.csv')
hdr_data = pd.read_csv("HDR_Data.csv")

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
    .interactive()
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
    .interactive()
)


chart = alt.vconcat(points, bars, data=df, title="HDI Rank (2021) based on Human Development Groups").interactive()

year_X = []
gdi_Y = []

for i in range(2015,2022):
    ylabel = "Gender Development Index ("+str(i)+")"
    year_X.append(i)
    gdi_Y.append(ylabel)


st.subheader("HDI Rank (2021) based on Human Development Groups")
with st.expander("See explanation"):
    st.write("The above chart defines HDI Rank (2021) based on Human Development Groups")
st.altair_chart(chart, theme="streamlit", use_container_width=True)


st.subheader("HDI Rank (2021) based vs Gender Development Index (2021)")
with st.expander("See explanation"):
    st.write("The above chart shows HDI Rank (2021) based vs Gender Development Index (2021)")
col1, col2 = st.columns(2)
with col1:
   selected_year = st.selectbox('Year', list(reversed(range(1990,2021))))
with col2:
   selected_continent = st.selectbox('Continent', list(hdr_data['continent'].unique()))
selected_data = data_with_filtered(hdr_data,[['continent',selected_continent]])
st.write(selected_data[selected_data['year']==selected_year])
fig = px.line(selected_data[selected_data['year']==selected_year],  y="gdi",color="country")
st.plotly_chart(fig, theme="streamlit")



st.subheader("Gender Development Index over Last 6 years")
with st.expander("See explanation"):
    st.write("Gender Development Index over Last 6 years")
st.line_chart(df,x="HDI Rank (2021)",y=gdi_Y,color="Country")


