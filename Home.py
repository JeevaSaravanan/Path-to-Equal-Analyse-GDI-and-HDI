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

st.sidebar.header("Home")

tab1, tab2, tab3, tab4 = st.tabs(["Origins: Unveiling the Background and Purpose", "The Dataset","The App's Significance: Why Pursuing It is Worthwhile","Conclusion and Future Work"])

with tab1:
    st.subheader('Origins: Unveiling the Background and Purpose')
    coli, colk = st.columns(2)
    with coli:
        st.image("https://mammothmemory.net/images/user/base/geography/changing%20economic%20world/remember-human-development-index-in-changing-economic-world-geography-2.01867df.jpg")
    with colk:
        st.markdown("""
        ***Goal:***
        The goal of this app is to analyze the Gender Development Index (GDI) and Human Development Index(HDI) dataset to understand the extent of gender disparities in human development across different countries. we can analyse how do gender disparities in health and education impact overall human development and improvement over these years
        """)
        st.write('In today\'s world, we grapple with an unprecedented "uncertainty complex" characterized by the destabilizing forces of the Anthropocene, the pursuit of transformative societal changes, and increasing polarization. The ongoing COVID-19 pandemic, global environmental crises, and rising inequalities have fueled this uncertainty, impeding human development and creating a sense of insecurity. For the first time, the global Human Development Index (HDI) has declined for two consecutive years, while democratic backsliding has worsened. Despite the perils of these new uncertainties, there is promise in reimagining our future, adapting our institutions, and crafting new narratives. This presents an opportunity to thrive in a rapidly changing world.')
    
    st.markdown("""
The Human Development Index (HDI) and Gender Development Index (GDI) are both composite indices developed by the United Nations Development Programme (UNDP) to assess and compare the overall development and gender-related development of countries.

### Human Development Index (HDI):

The Human Development Index is a summary measure of a country's average achievements in three key dimensions of human development:

1. **Health:** Measured by life expectancy at birth.
2. **Education:** Measured by mean years of schooling for adults and expected years of schooling for children entering school.
3. **Standard of Living:** Measured by gross national income (GNI) per capita, adjusted for purchasing power parity.

HDI is designed to provide a comprehensive overview of a country's overall development by considering health, education, and living standards. It ranks countries into categories such as low, medium, high, or very high human development.

### Gender Development Index (GDI):

The Gender Development Index is an extension of the HDI that takes into account gender disparities in the same three dimensions:

1. **Health (GDI-H):** Compares female and male life expectancies.
2. **Education (GDI-E):** Compares female and male mean years of schooling and expected years of schooling.
3. **Standard of Living (GDI-S):** Compares female and male per capita income.

The GDI is calculated as the ratio of the female HDI to the male HDI. A GDI value of 1 indicates gender parity, while values less than 1 signify gender disparities, with lower values indicating greater disparities.

### Relationship Between HDI and GDI:

- **Positive Correlation:** In general, countries with higher HDI values tend to have higher GDI values as well. This suggests that improvements in overall human development are often associated with reductions in gender disparities.

- **Focus on Gender Inequality:** While the HDI provides an overall assessment of development, the GDI specifically highlights gender inequalities. It draws attention to disparities in health, education, and living standards between females and males within a country.

- **Policy Implications:** The GDI can inform policymakers about areas where gender disparities persist, helping them design targeted interventions to promote gender equality. It complements the HDI by offering a gender-sensitive perspective.

Both the HDI and GDI contribute to a more nuanced understanding of a country's development landscape, emphasizing the importance of not only overall progress but also gender-inclusive development. These indices play a crucial role in shaping global development agendas and policies aimed at achieving sustainable and equitable human development.
""")
    
    video_file = open('hdr21-22_animation.mp4 (1080p).mp4', 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)
    st.caption("Reference : Video Referred from HDRO video for HDR21-22")

with tab2:

    st.subheader("Dataset:")

    colkd, colka, colkb = st.columns(3)
    with colka:
        st.image("https://www.shutterstock.com/image-vector/measures-human-development-rectangle-infographic-260nw-2157969053.jpg")
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
    st.subheader("The App's Significance: Why Pursuing It is Worthwhile")

    colkj, colkk = st.columns(2)

    with colkj:
        st.image("https://i.ytimg.com/vi/w5wORaWcWPY/maxresdefault.jpg",use_column_width=True)
    with colkk:
        st.markdown("""
        - This app is worthy of completion because it addresses a critical issue – gender disparities in human development – which has real-world implications for policymaking and social progress. By analyzing the GDI dataset and creating an accessible web app, we can contribute to the ongoing efforts to promote gender equality and improve human development outcomes globally.

        - **THE PATH TO GENDER EQUALITY:** This expands the measures for women and girls to exercise their potential, their opportunities and the choices available to them. Policies that seek to further empower women and girls and achieve gender parity require robust data and measures that are comparable across countries and based on a sound methodology. While some such measures are available, the picture has been incomplete.
        """)

with tab4:
    st.subheader("Conclusion")

    st.image("https://earimediaprodweb.azurewebsites.net/Api/v1/Multimedia/ce656816-a566-4317-97a2-1879b0d3fd88/Rendition/low-res/Content/Public")
    st.markdown("""
### Conclusion:

The Gender Development Index (GDI) app serves as a valuable tool for analyzing and understanding gender disparities in human development across various countries. Through the exploration of indicators related to health and education, the app provides insights into the extent of gender inequality and its impact on overall human development.

The analysis revealed patterns and variations in gender disparities, shedding light on regions or countries where improvements are needed. The incorporation of historical data allows for tracking changes over the years, providing a dynamic perspective on gender development trends.

Key findings include identifying areas where gender disparities persist, understanding the correlation between gender disparities in health and education, and recognizing the importance of addressing these disparities for comprehensive human development.

### Future Work:

1. **Enhanced Data Visualization:**
   - Implement more interactive and informative data visualizations, such as dynamic maps, trend charts, and comparative dashboards, to enhance the user experience and facilitate a deeper understanding of the data.

2. **Predictive Modeling:**
   - Integrate predictive modeling techniques to forecast future trends in gender development. Machine learning algorithms can help identify factors influencing gender disparities and predict potential changes over time.

3. **Incorporate Socioeconomic Factors:**
   - Expand the analysis by including additional socioeconomic factors that contribute to gender disparities, such as income inequality, employment opportunities, and social norms. This can provide a more comprehensive view of the challenges faced by different communities.

4. **User Feedback and Iterative Improvements:**
   - Gather user feedback to understand the needs and preferences of the app's audience. Use this feedback to make iterative improvements, add new features, and address any limitations or usability issues.

5. **Policy Recommendations:**
   - Develop a section of the app dedicated to policy recommendations based on the analysis. Provide insights for policymakers, NGOs, and advocates to design targeted interventions aimed at reducing gender disparities in health and education.

6. **Global Collaboration:**
   - Foster collaboration with international organizations, researchers, and data contributors to continually update and expand the dataset. A global collaborative effort can enhance the accuracy and relevance of the information presented in the app.

7. **Educational Outreach:**
   - Incorporate educational resources within the app to raise awareness about gender disparities and promote gender equality. This could include informative articles, case studies, and success stories from regions that have successfully addressed gender inequalities.

By incorporating these future work recommendations, the GDI app can evolve into a powerful tool for not only analyzing gender disparities but also driving positive change through informed decision-making and targeted interventions.
""")
