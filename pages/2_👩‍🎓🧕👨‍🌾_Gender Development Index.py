import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
from plotly.express import choropleth
import altair as alt
import math


st.set_page_config(page_title="Gender Development Index", page_icon="👩‍🎓🧕👨‍🌾", layout="wide")

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

df = pd.read_csv('Gender Development Index.csv')
hdr_data = pd.read_csv("HDR_Data.csv")

st.markdown("<h1 style='text-align: center;'>The Path to Equal</h1>",unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col3:
    st.caption("~A Project done by Jeeva Saravana Bhavanandam")
st.subheader("", divider='rainbow')



st.sidebar.header("Gender Development Index")

st.subheader('Gender Development Index')
st.write('The Gender Development Index (GDI) is a composite measure designed to assess gender disparities and inequalities in a society by considering factors related to human development. It is an extension of the Human Development Index (HDI) and focuses on three key dimensions: health, education, and income. In the GDI, these dimensions are assessed separately for males and females, allowing for a comparison of gender-based development gaps. Health indicators typically include life expectancy at birth for both genders. Education indicators encompass literacy rates and enrollment in primary, secondary, and tertiary education for both males and females. The income component typically examines income levels and workforce participation for both genders.')
st.markdown("""
**How to Read the values?** 
- Values `below 1` indicate `higher human development` for `men than women`, while `values above 1` indicate `the opposite`.
""")
gdi_img = Image.open('Images/gdi.png')
st.image(gdi_img,"Calculating the GDI—graphical presentation")

st.divider()

st.subheader("Countries and their respective Gender Development Index (GDI) values ")
col1, col2 = st.columns(2)


country_list=list(hdr_data.country.unique())

continent_list=list(hdr_data.continent.unique())
continent_list.insert(0,"All")

year_list=list(reversed(range(1990,2021)))


with col1:
   selected_continent= st.selectbox('Continent', continent_list)
with col2:
    selected_year = st.slider('Year',1990,2021)


heading="Countries in "+ selected_continent +" and their resprective GDI in " + str(selected_year)

st.markdown("#### "+heading)


selcted_data = data_with_filtered(hdr_data,[['continent',selected_continent],['year',selected_year]]) if selected_continent!="All" else data_with_filtered(hdr_data,[['year',selected_year]])
selcted_data = selcted_data.sort_values(by='gdi', ascending=False)


map_fig=choropleth(data_frame=selcted_data, locations='iso3', color='gdi', hover_name='country',color_continuous_scale="sunset")
map_fig.update_geos(fitbounds="locations", visible=False)
map_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(map_fig,use_container_width=True)



st.divider()

#---------------------------------------------------

st.subheader("A comparison between the Gender Development Index (GDI) and the Human Development Index (HDI) for a country.")

col_left, col_mid, col_right = st.columns(3)

with col_left:

    countries_list=list(hdr_data["country"].unique())
    line_charts=[]

    options = st.multiselect('Select 1 countries',countries_list,[countries_list[4]],max_selections=1)
    for country in options:
        temp=hdr_data[hdr_data["country"]==country]
        fig = alt.Chart(temp).mark_line().encode(
            x='year:N',
            y='hdi',
            color='country:N',
            strokeDash=alt.condition(
                alt.datum.symbol == country,
                alt.value([5, 5]),  # dashed line: 5 pixels  dash + 5 pixels space
                alt.value([5, 5]),  # solid line
            )).interactive()
    
        fig2 = alt.Chart(temp).mark_line(point=True).encode(
            x='year:N',
            y='gdi',
            color='country:N',
        ).interactive()
    
        f_fig=fig+fig2
    
    
        line_charts.append(f_fig)
    
    if(len(line_charts)>0):
        fin_fig=line_charts[0]
        for i in range(1,len(line_charts)):
            fin_fig = fin_fig+line_charts[i]

        st.altair_chart(fin_fig, theme="streamlit")

with col_right:
    st.write("")
    st.write("")
    st.subheader("Overview:")
    st.markdown("""
               - This line chart provides a direct comparison between a country's Gender Development Index (GDI) and its Human Development Index (HDI) over a specific period.
               - Users can visually assess how the GDI and HDI scores have changed or remained consistent for the chosen country.
               - Trends and patterns in gender development and overall human development can be easily identified by examining the slopes and intersections of the two lines.
               - The chart offers a clear perspective on the relationship between gender-specific development and the broader human development context in a single country.
               """)

st.divider()


#------------------------------------------------------------

st.subheader("The top 10 countries from a specific continent, ranked by their Gender Development Index (GDI).")
col21,col22,col23 = st.columns(3)

with col23:
      continent_list = list(hdr_data['continent'].unique())
      continent_list.insert(0,"All")
      selected_continent = st.selectbox('Filter by Continent', continent_list)

if selected_continent!="All":
   selected_conti_data = hdr_data[hdr_data["continent"]==selected_continent]
else:
   selected_conti_data = hdr_data
ten_largest = selected_conti_data[selected_conti_data["year"]==2021].sort_values(by="gdi",ascending=False)
fig1 = alt.Chart(ten_largest.head(10)).mark_bar().encode(
   x =  alt.X("country:N").sort("-y"),
   y = "gdi:Q",
   color='country:N',
   tooltip=["country","gdi","gii"]
).properties(
   title='Top 10 Countries ranked by GDI  '
).interactive()
text = fig1.mark_text(
    align='center',
    dy=-3
).encode(
    text='gdi:Q'
)
st.altair_chart(fig1+text, theme="streamlit",use_container_width=True)

st.divider()

#---------------------------------

st.subheader("Gender Development Index - Report(2021_2022)")

def build_table(selcted_data2):
    s=""
    start=f"""<table>
    <tr>
    <th>Hdi Rank</th>
    <th>Flag</th>
    <th>Country</th>
    <th>GDI 2021</th>
    <th>Change From 2020</th></tr>
    """ 
    mid_all=""
    for index, row in selcted_data2.iterrows():

        
        style=f"style='color: red;'"
        val=row['difference']
        gdi=row['Gender Development Index (2021)']
        rank=row['HDI Rank (2021)']

        if(math.isnan(gdi)):
            gdi="-"
        if(math.isnan(rank)):
            rank="-"
        
        if(math.isnan(val)):
            style=f"style='color: gray;'"
            val="-"
            arrow=f"<text>  </text>  "
        elif(val>0):
            style=f"style='color: green;'"
            arrow=f"<text> ↑ </text> "
        elif(val==0.0):
            style=f"style='color: gray;'"
            arrow=f"<text> = </text> "
        else:
            arrow=f"<text> ↓ </text>  "

        
        
        

        mid=f"<tr>  <td>{rank}</td>  <td> <img src='{row['URL']}' alt='drawing' width='20'/> </td> <td>{row['Country_x']}</td>  <td  >{gdi}</td> <td {style} >{val} {arrow}</td> </tr>  "
        mid_all=mid_all+mid

    
    
    end="""</table>"""

    s=start+mid_all+end


    return s

def data_with_filtered(hdr_data,array):
    result=hdr_data
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

flags = pd.read_csv("flags_iso.csv")

hdr_data = hdr_data.merge(flags, left_on='iso3', right_on='Alpha-3 code', how='left')
df=df.merge(flags, left_on='ISO3', right_on='Alpha-3 code', how='left')

cont_list = list(df.Continent.unique())
cont_list.insert(0,"All")

contc1= st.selectbox('Continent', cont_list)
df["difference"]=(df["Gender Development Index (2021)"]-df["Gender Development Index (2020)"]).round(4)

srtby= st.selectbox('Sort by', ["Country A-Z", "Country Z-A","HDI Rank Asc" ,"GDI Change Asc","GDI Change Dsc"] )

st.write("")
st.write("")

selcted_data2 = data_with_filtered(df,[['Continent',contc1]])

if(srtby=="Country A-Z"):
    selcted_data2=selcted_data2.sort_values(by='Country_x', ascending=True)
elif(srtby=="Country Z-A"):
    selcted_data2=selcted_data2.sort_values(by='Country_x', ascending=False)
elif(srtby=="HDI Rank Asc"):
    selcted_data2=selcted_data2.sort_values(by='HDI Rank (2021)', ascending=True)
elif(srtby=="HDI Rank Dsc"):
    selcted_data2=selcted_data2.sort_values(by='HDI Rank (2021)', ascending=False)
elif(srtby=="GDI Change Asc"):
    selcted_data2=selcted_data2.sort_values(by='difference', ascending=True)
elif(srtby=="GDI Change Dsc"):
    selcted_data2=selcted_data2.sort_values(by='difference', ascending=False)

n = (selcted_data2.shape[0])//3
table1=build_table(selcted_data2.iloc[:n,:])
table2=build_table(selcted_data2.iloc[n:2*n,:])
table3=build_table(selcted_data2.iloc[2*n:,:])
col_table1, col_table2, col_table3 = st.columns(3)
with col_table1:
    st.markdown(table1,True)
with col_table2:
    st.markdown(table2,True)
with col_table3:
    st.markdown(table3,True)





