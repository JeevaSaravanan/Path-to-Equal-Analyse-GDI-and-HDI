import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
from plotly.express import choropleth
import altair as alt



st.set_page_config(page_title="Human Development Index", page_icon="👷👮👳‍♂️", layout="wide")

st.markdown("<h1 style='text-align: center;'>The Path to Equal</h1>",unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col3:
    st.caption("~A Project done by Jeeva Saravana Bhavanandam")
st.subheader("", divider='rainbow')

st.sidebar.header("Human Development Index")

st.subheader('Human Development Index')
st.write('The Human Development Index (HDI) is a summary measure of average achievement in key dimensions of human development: a long and healthy life, being knowledgeable and having a decent standard of living. The HDI is the geometric mean of normalized indices for each of the three dimensions.')
hdi_road_map_img = Image.open('Images/hdiRoadMap.png')
st.image(hdi_road_map_img,"HDI Dimensions & Indicator")

st.divider()

def data_with_filtered(data,array):
    result=data
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

hdr_data = pd.read_csv("HDR_Data.csv")
st.subheader("Explore HDI")
countries_list=list(hdr_data["country"].unique())
line_charts=[]

col1, col2 = st.columns(2)

with col1:
   st.subheader("HDI - Country wise comparison")
   st.markdown("""
               - User can compare Human Development Index of different countries(up to 3).
               - On Hover information like Gross National Income per Capita(GNIPC) ,HDI value for that year will be shown.
               - From the graph we can understand How the HDI is distributed over the year for different countries.

               **What is GNIPC?**
               Gross National Income per capita (GNI per capita) calculates the average income earned by a country's citizens over a year, indicating economic well-being and living standards. It's frequently used alongside metrics such as the Human Development Index (HDI) to evaluate a nation's overall development and quality of life.

               **More on GNIPC and HDI Relation:**
               - **Economic Well-being:** GNI per capita is a key factor in HDI, representing a nation's ability to improve living standards, health, education, and overall well-being.

               - **Resource Allocation:** Higher GNI per capita provides countries with more resources for crucial services like healthcare and education, improving human development and boosting HDI scores.

               - **Standard of Living:** GNI per capita reflects citizens' average income, impacting their standard of living, and a higher GNI per capita typically leads to an improved standard of living, a factor considered in the HDI.

               - **Development Trajectory:** While a high GNI per capita doesn't guarantee a high HDI score, it can foster human development through investments in healthcare, education, and infrastructure, enhancing key HDI components like life expectancy, education, and income
               
               """)
with col2:

   options = st.multiselect('Add Country to compare (UP TO 3)',countries_list,[countries_list[4], countries_list[24],countries_list[69]],max_selections=3)
   st.write("")
   st.write("")
   st.write("")
   st.write("")
   st.write("")
   st.write("")

   for country in options:
      temp=hdr_data[hdr_data["country"]==country]
      fig = alt.Chart(temp).mark_line( point=alt.OverlayMarkDef(filled=False, fill="white")).encode(
      x='year:N',
      y='hdi',
      color='country:N',
      tooltip=["gnipc","country","year","hdi"]
      ).properties(
      title='HDI - Country wise comparison'
      ).interactive()
      line_charts.append(fig)
      
   if(len(line_charts)>0):
      fin_fig=line_charts[0]
      for i in range(1,len(line_charts)):
         fin_fig = fin_fig+line_charts[i]

      st.altair_chart(fin_fig, theme="streamlit")


st.divider()

st.subheader("HDI distribution across Continents")

col1, col2,col3 = st.columns(3)

with col1:
   option = st.selectbox(
    'Select a decade you want to compare',
    ('2010-2020','2000-2009','1990-1999'))
   decade_dict = {'1990-1999':list(range(1990,2000)),'2000-2009':list(range(2000,2010)),'2010-2020':list(range(2010,2020))}
   
   select_data = hdr_data[hdr_data["year"].isin(decade_dict[option])]
   fig = px.bar(select_data, x="hdi", y="year", color='continent', orientation='h',
                  hover_data=["gnipc","country"],
                  title='HDI distribution across the Continent')

   st.plotly_chart(fig, theme="streamlit")

with col3:
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.write("")
      st.subheader("Overview:")
      st.markdown("""
               - Users can explore the distribution of the Human Development Index (HDI) across various continents.
               - When hovering over data points, users can access additional information, including Gross National Income per Capita (GNIPC) and the specific HDI value for that year.
               - This graphical representation offers insights into how HDI evolves over the years for different continents, enabling a better understanding of its trends and variations. """)

st.divider()

st.subheader("Top 10 Countries from a Continent with High HDI")


col21,col22,col23 = st.columns(3)

with col23:
      continent_list = list(hdr_data['continent'].unique())
      continent_list.insert(0,"All")
      selected_continent = st.selectbox('Filter by Continent', continent_list)

if selected_continent!="All":
   selected_conti_data = hdr_data[hdr_data["continent"]==selected_continent]
else:
   selected_conti_data = hdr_data
ten_largest = selected_conti_data[selected_conti_data["year"]==2021].sort_values(by="hdi",ascending=False)

fig1 = alt.Chart(ten_largest.head(10)).mark_bar().encode(
   x = alt.X("country:N").sort("-y"),
   y = "hdi:Q",
   color='country:N',
   tooltip=["country","hdi_rank_2021","hdi"]
).properties(
   title='Top 10 Countries from a Continent with High HDI'
).interactive()
text = fig1.mark_text(
    align='center',
    dy=-3
).encode(
    text='hdi:Q'
)

st.altair_chart(fig1+text, theme="streamlit",use_container_width=True)

st.divider()


col_left, col_mid, col_right = st.columns(3)


df = pd.read_csv('Gender Development Index.csv')

with col_left:
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



   st.subheader("HDI Rank (2021) based on Human Development Groups")
   st.altair_chart(chart, theme="streamlit", use_container_width=True)

with col_right:
      st.subheader("Overview:")
      st.markdown("""
               - Users can explore the distribution of the Human Development Index (HDI) Rank across various continents based on Human Development Groups.
               - In this context, countries are categorized into Human Development Groups  `Low(<0.550)`, `Medium (0.550-0.699)`, `High (0.700-0.799)`, `Very high (≥ 0.800)` based on their Human Development Index (HDI) scores.
               - HDI Rank (2021) represents the ranking assigned to each country based on their Human Development Index (HDI) scores in the year 2021.
               - The bar charts below display the record count for each continent.
               - Users have the option to select specific data segments of interest and access detailed information about them.
               """)
st.divider()



st.subheader("HUMAN DEVELOPMENT INSIGHTS - HDI Rank(2021)")
st.write("Explore human development data for 191 countries and territories worldwide.")

map_fig=choropleth(data_frame=hdr_data, locations='iso3', color='continent', hover_data=['hdi_rank_2021','hdi','country'],featureidkey="properties.district")
map_fig.update_geos(fitbounds="locations", visible=False)
map_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(map_fig,use_container_width=True)

st.divider()
