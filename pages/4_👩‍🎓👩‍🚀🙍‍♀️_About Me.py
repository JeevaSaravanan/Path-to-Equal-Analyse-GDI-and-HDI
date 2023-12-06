import streamlit as st

st.set_page_config(page_title="About Me", page_icon="👩‍🎓👩‍🚀🙍‍♀️", layout="wide")

st.markdown("<h1 style='text-align: center;'>The Path to Equal</h1>",unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col3:
    st.caption("~ Made by Jeeva Saravana Bhavanandam")
st.subheader("", divider='rainbow')

st.sidebar.header("About Me")



st.write(" ")
st.write(" ")
st.write(" ")
st.markdown("### Hi 👋")
st.markdown("I'm Jeeva, a Passionate Software Engineer and AI Enthusiast. I'm in pursuit of expanding my knowledge and open to exploring opportunities along my way. I prefer to keep learning, continue to challenge myself, and do interesting things that matter.*")
with open("Images/Jeeva Saravana Bhavanandam_Resume_.pdf", "rb") as file:
    btn = st.download_button(
            label="Download Resume",
            data=file,
            file_name="fJeeva Saravana Bhavanandam_Resume_.pdf",
          )
st.markdown("""

<h3>At Present📌</h3>

* Master in Data Science 📚 Student at Michigan State University. 
* Research Assistant 👩🏻‍💻 at  Henry Ford Health System.

<h3>What I'm focusing on 👇🏻</h3>

* Learning: Machine Learning, AI 🤖 ...
  <i>Would love to collobrate and receive suggestion.</i>

<h3>In my spare time 👀</h3>

You can find me with books sometimes. I love to read coming of age fiction not big fan of romance though. In my spare time I watch sitcoms and do some sketches.

<b>Favourite Book:</b> 
<ul> <li><i> The Kite Runner by Khaled Hosseini.</i></li> <li><i> Tuesday with Morrie by Mitch Albom.</i></li></ul>
""",unsafe_allow_html=True)
st.markdown("""
<div align="center">
<a href="https://jeevasaravanan.medium.com/"> <img src="https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=white"/></a> <a href="https://www.linkedin.com/in/jeeva-saravanan/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/></a>
</div>
""",unsafe_allow_html=True)


