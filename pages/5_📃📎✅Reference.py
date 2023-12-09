import streamlit as st


st.set_page_config(
    page_title="Reference",
    page_icon="📃📎✅",
    layout="wide"
)




st.markdown("<h1 style='text-align: center;'>The Path to Equal</h1>",unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col3:
    st.caption("~Made by Jeeva Saravana Bhavanandam")
st.subheader("", divider='rainbow')


st.sidebar.header("References")

st.subheader("Reference")
st.write("""
[1] F. B. Khan and A. Noor, "Prediction and Classification of Human Development Index Using Machine Learning Techniques," 2021 5th International Conference on Electrical Information and Communication Technology (EICT), Khulna, Bangladesh, 2021, pp. 1-6, doi: 10.1109/EICT54103.2021.9733645.

            """)

st.write("[2][online] Available: http://hdr.undp.org/en/content/human-development-index-hdi.")

st.write("""[3] F. Astika Saputra, A. Barakbah and P. Riza Rokhmawati, "Data analytics of human development index(hdi) with features descriptive and predictive mining", 2020 International Electronics Symposium (IES), pp. 316-323, 2020.""")

st.write("""[4] C. B. d. Santos, L. A. Pilatti, B. Pedroso, D. R. Carvalho and A. M. Guimarães, "Forecasting the human development index and life expectancy in latin american countries using data mining techniques", Ciência & Saúde Coletiva, vol. 23, pp. 3745-3756, 2018.""")

st.write("""[5] R. T. Vulandari, S. Siswanti, A. K. Kusumawijaya and K. Sandradewi, "Classification of human development index using k-means", Indonesian Journal of Applied Statistics, vol. 2, no. 1, pp. 1-9.""")