import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
from plotly.express import choropleth
import altair as alt
import plotly.graph_objects as go
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, accuracy_score
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay



st.set_page_config(page_title="Prediction and Classification of HDI", page_icon="👾📊📈", layout="wide")

st.markdown("<h1 style='text-align: center;'>The Path to Equal</h1>",unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col3:
    st.caption("~ Made by Jeeva Saravana Bhavanandam")
st.subheader("", divider='rainbow')

st.sidebar.header("Prediction and Classification of HDI")

shdi = pd.read_csv('SHDI-SGDI-Total 7.0.csv')

columns_to_drop = ['iso_code', 'GDLCODE','level','sgdi','shdif','shdim','healthindex','healthindexf','healthindexm','incindex', 'incindexf', 'incindexm','edindexf', 'edindexm','eschf', 'eschm','mschf', 'mschm','lifexpf', 'lifexpm','gnicf','gnicm','lgnic', 'lgnicf', 'lgnicm', 'pop']
shdi.drop(columns=columns_to_drop, inplace=True)

shdi['shdi'] = pd.to_numeric(shdi['shdi'], errors='coerce')
shdi['edindex'] = pd.to_numeric(shdi['edindex'], errors='coerce')
shdi['esch'] = pd.to_numeric(shdi['esch'], errors='coerce')
shdi['msch'] = pd.to_numeric(shdi['msch'], errors='coerce')
shdi['lifexp'] = pd.to_numeric(shdi['lifexp'], errors='coerce')
shdi['gnic'] = pd.to_numeric(shdi['gnic'], errors='coerce')

def categorise_hdi(row):
    shdi_value = row['shdi']
    if (shdi_value >= 0.8):
        return "Very High"
    elif(shdi_value>=0.7):
        return "High"
    elif(shdi_value>=0.550):
        return "Medium"
    else:
        return "Low"
shdi['hdi_category'] = shdi.apply(categorise_hdi, axis=1)

country_list = shdi["country"].unique()

tab1, tab2, tab3 = st.tabs(['Forecast','Prediction','Classification'])

with tab1:
    st.markdown("""
**What is ARIMA?**
An autoregressive integrated moving average model is a form of regression analysis that gauges the strength of one dependent variable relative to other changing variables. The model's goal is to predict future securities or financial market moves by examining the differences between values in the series instead of through actual values.

An ARIMA model can be understood by outlining each of its components as follows:

### Autoregressive (AR) Component:
The autoregressive component involves predicting the future value of a time series based on its past values. The notation AR(p) represents the autoregressive component of order p.

$Y_t$ = $\phi_1 Y_{t-1}$ + $\phi_2 Y_{t-2}$ + $\ldots$ + $\phi_p Y_{t-p}$ + $\epsilon_t$

- $Y_t$ is the value of the time series at time $t$.
- $\phi_1, \phi_2, \ldots, \phi_p$ are the autoregressive parameters.
- $\epsilon_t$ is white noise, representing the error term.

### Integrated (I) Component:
The integrated component involves differencing the time series data to make it stationary, i.e., removing trends and seasonality. The notation I(d) represents the differencing of order d.

$∇^d Y_t = (1 - B)^d Y_t $

- $B$ is the backshift operator ($B Y_t = Y_{t-1}$).
- $d$ is the differencing order.

### Moving Average (MA) Component:
The moving average component involves modeling the relationship between the current value and a stochastic term based on past forecast errors. The notation MA(q) represents the moving average component of order q.

$ Y_t = \epsilon_t + θ_1 \epsilon_{t-1} + θ_2 \epsilon_{t-2} + \ldots + θ_q \epsilon_{t-q} $

- $θ_1, θ_2, \ldots, θ_q$ are the moving average parameters.

### ARIMA Model:
Combining the AR, I, and MA components, an ARIMA model is represented as ARIMA(p, d, q).

$∇^d Y_t$ = $\phi_1 ∇^d Y_{t-1} + \phi_2 ∇^d Y_{t-2} + \ldots + \phi_p ∇^d Y_{t-p} + \epsilon_t + θ_1 \epsilon_{t-1} + θ_2 \epsilon_{t-2} + \ldots + θ_q \epsilon_{t-q} $

Here:
- $p$ is the order of the autoregressive component.
- $d$ is the order of differencing.
- $q$ is the order of the moving average component.

The goal is to estimate the parameters $\phi_1, \phi_2, \ldots, \phi_p, θ_1, θ_2, \ldots, θ_q$ that minimize the difference between the predicted values and the actual values of the time series. This is typically done using methods like maximum likelihood estimation (MLE).""")
    col1, col2,col3 = st.columns(3)
    with col1:
        selected_country= st.selectbox('Country ', country_list)
        region_list = shdi[(shdi["country"]==selected_country)]["region"].unique()
    with col2:
        selected_region= st.selectbox('Region ',region_list)

    df = shdi[(shdi["country"]==selected_country) & (shdi['region']==selected_region)]

    st.subheader("Arima Forecast")

    train=df.copy()

    arima_model = ARIMA(train['shdi'], order=(5,1,0))
    arima_fit = arima_model.fit()

    arima_predictions = arima_fit.forecast(steps=8)
    dict_preds = {"year":list(range(2021,2029)),"shdi":arima_predictions}

    df_preds = pd.DataFrame(dict_preds)

    # Create Line Plot 1
    trace1 = go.Scatter(x=train['year'], y=train['shdi'], mode='lines', name='HDI - Actual', line=dict(color='lightblue'))

    # Create Line Plot 2
    trace2 = go.Scatter(x=df_preds['year'], y=df_preds['shdi'], mode='lines', line_dash='dash',name='HDI - Predicted', line=dict(color='blue'))

    # Create Layout
    layout = go.Layout(title='HDI - Actual vs Predicted', xaxis=dict(title='Year'), yaxis=dict(title='HDI'))

    # Create Figure
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Predicted Value")
    fig1 = px.bar(df_preds, x='year',y='shdi',text_auto=True)
    fig1.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
    st.plotly_chart(fig1, use_container_width=True)


with tab2:
    col1, col2,col3 = st.columns(3)
    with col1:
        selected_country= st.selectbox('Country', country_list)
        region_list = shdi[(shdi["country"]==selected_country)]["region"].unique()
    with col2:
        selected_region= st.selectbox('Region',region_list)

    df = shdi[(shdi["country"]==selected_country) & (shdi['region']==selected_region)]
    


    st.write('Features to be considered for prediction')
    feature_list=[ 'Education Index','Expected years of Schooling','Mean years of Schooling','Life Expentancy','Gross National Income per Captia']
    cols = st.columns(len(feature_list))
    box=[]

    for i in range(0,len(feature_list)):
        with cols[i]:
            ck=st.checkbox(feature_list[i],value=True)
            box.append(ck)
    
    dict_reference = {"Education Index":"edindex","Expected years of Schooling":"esch","Mean years of Schooling":"msch","Life Expentancy":"lifexp","Gross National Income per Captia":"gnic"}
    selected_values = [dict_reference[value] for value, is_true in zip(feature_list, box) if is_true]
    features = df[selected_values+['year']]

    X_train, X_test, y_train, y_test = features[features["year"]<2017], features[features["year"]>=2010] , df[df['year']<2017]['shdi'],df[df['year']>=2010]['shdi']

    random_forest = st.toggle('Show Predictions with Random Forest')
    linear_regression = st.toggle('Show Predictions with Linear Regression')
    elastic_nt = st.toggle('Show Predictions with Elastic Net')

    if(random_forest):

        st.markdown("### Prediction with RandomForest [to know more](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)")

        st.markdown("""
Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Each tree in the ensemble is trained on a random subset of the training data.

Let's denote:

- $X$ as the input features (a vector of predictors),
- $Y$ as the output or target variable,
- $n$ as the number of trees in the forest.

Here's a simplified mathematical representation of the Random Forest algorithm:

### Training:

1. **For each tree $i$ from 1 to $n$:**
   - Randomly select a subset of the training data (with replacement) to create a bootstrap sample.
   - Randomly select a subset of features at each split. The number of features to consider at each split is a hyperparameter and is often denoted as $m$ (where $m < {total number of features}$).

2. **Grow a decision tree $T_i$:**
   - For each node in the tree:
     - Randomly select $m$ features.
     - Split the node using the feature that provides the best split according to a criterion (e.g., Gini impurity for classification, mean squared error for regression).

### Prediction:

To make a prediction for a new input $X_{{new}}$:

1. **For each tree $i$ from 1 to $n$:**
   - Traverse the tree $T_i$ based on the features of $X_{{new}}$ until reaching a leaf node.

2. **Aggregate the predictions:**
   - For classification: Take a majority vote among the classes predicted by all the trees.
   - For regression: Take the average of the predictions from all the trees.

### Overall Prediction:

For a Random Forest with $n$ trees, the final prediction is the combination of predictions from all individual trees.
""")
        st.latex(r'''
        Prediction_{{final}}(X_{{new}}) = (\frac{1}{n})\sum_{i=1}^{n}{Prediction}_{i}(X_{{new}})
        ''')
        st.markdown("""
    This ensemble approach helps improve generalization and reduce overfitting compared to individual decision trees. The randomness introduced during training and prediction makes Random Forest robust and effective for various machine learning tasks.
        """)
        n_estimator = st.slider('n_estimator',min_value=50, max_value=300, value=100, step=25 )
        max_depth = st.slider('max_depth',min_value=1, max_value=20, value=7, step=1 )
        min_samples_split=st.slider('min_samples_split',min_value=1, max_value=20, value=5, step=1 )

        
        rf_model = RandomForestRegressor(n_estimators=n_estimator,max_depth=max_depth,min_samples_split=min_samples_split,random_state=42)

        rf_model.fit(X_train, y_train)

        rf_predictions = rf_model.predict(X_test)

        dict_predictions = {"year":list(range(2010,2022)),"shdi":rf_predictions}

        df_predictions = pd.DataFrame(dict_predictions)

        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
        print(f'Random Forest RMSE: {rf_rmse}')

        # Create Line Plot 1
        trace1 = go.Scatter(x=df['year'], y=df['shdi'], mode='lines', name='HDI - Actual', line=dict(color='orange'))

        # Create Line Plot 2
        trace2 = go.Scatter(x=df_predictions['year'], y=df_predictions['shdi'], mode='lines', line_dash='dash',name='HDI - Predicted', line=dict(color='green'))

        # Create Layout
        layout = go.Layout(title='HDI - Actual vs Predicted', xaxis=dict(title='Year'), yaxis=dict(title='HDI'))
        fig = go.Figure(data=[trace1, trace2], layout=layout)

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    

    if(linear_regression):

        st.markdown("### Prediction with Linear regression [to know more](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)")

        st.markdown("""
Linear regression is a simple and widely used statistical method for modeling the relationship between a dependent variable $Y$ and one or more independent variables $X$. The linear regression model can be mathematically represented as follows:

### Simple Linear Regression:

For a simple linear regression with one independent variable:

$ Y =β_0 +β_1 X + ε $

- $Y$ is the dependent variable.
- $X$ is the independent variable.
- $β_0$ is the y-intercept (the value of $Y$ when $X$ is 0).
- $β_1$ is the slope (the change in $Y$ for a unit change in $X$).
- $ε$ is the error term, representing unobserved factors that affect $Y$ but are not included in the model.

### Multiple Linear Regression:

For multiple linear regression with $n$ independent variables $X_1$, $X_2$, $\ldots$, $X_n$:

$Y$ = $β_0 +β_1 X_1 +β_2 X_2 + $\ldots$ +β_n X_n + ε$

- $Y$ is the dependent variable.
- $X_1, X_2, \ldots, X_n$ are the independent variables.
- $β_0$ is the y-intercept.
- $β_1$, $β_2$, $\ldots$, $β_n$ are the slopes for the respective independent variables.
- $ε$ is the error term.

### Vectorized Form:

In vectorized form, the equation can be written using matrices and vectors:

$ Y = $Xβ + ε $

- $Y$ is a column vector of the dependent variable.
- $X$ is a matrix where each row corresponds to an observation, and each column corresponds to an independent variable.
- $β$ is a column vector of coefficients (including the intercept).
- $ε$ is a column vector of error terms.

###Residual Sum of Squares:

The goal in linear regression is to find the coefficients $β$ that minimize the sum of squared differences between the observed values $Y$ and the values predicted by the model. This is often expressed as the residual sum of squares (RSS):

$RSS(β)$= $\sum_{i=1}^{N} (Y_i - \hat{Y}_i)^2$

where $N$ is the number of observations, $Y_i$ is the observed value, and $\hat{Y}_i$ is the predicted value.

The coefficients $β$ are typically estimated using methods such as ordinary least squares (OLS), which aims to minimize the RSS.
""")
        lin_model = LinearRegression( )

        lin_model.fit(X_train, y_train)

        lin_predictions = lin_model.predict(X_test)

        dict_predictions = {"year":list(range(2010,2022)),"shdi":lin_predictions}

        df_predictions = pd.DataFrame(dict_predictions)

        lin_rmse = np.sqrt(mean_squared_error(y_test, lin_predictions))
        print(f'lin RMSE: {lin_rmse}')

        # Create Line Plot 1
        trace1 = go.Scatter(x=df['year'], y=df['shdi'], mode='lines', name='HDI - Actual', line=dict(color='hotpink'))

        # Create Line Plot 2
        trace2 = go.Scatter(x=df_predictions['year'], y=df_predictions['shdi'], mode='lines', line_dash='dash',name='HDI - Predicted', line=dict(color='blue'))

        # Create Layout
        layout = go.Layout(title='HDI - Actual vs Predicted', xaxis=dict(title='Year'), yaxis=dict(title='HDI'))
        fig = go.Figure(data=[trace1, trace2], layout=layout)

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    

    if elastic_nt:

        st.markdown("### Prediction with Elastic net [to know more](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)")

        st.markdown("""
Elastic Net regression is a linear regression model that combines the penalties of both L1 (Lasso) and L2 (Ridge) regularization methods. It is useful when dealing with datasets that have a large number of features and may exhibit multicollinearity. The objective function of Elastic Net is a combination of the L1 and L2 penalty terms. The mathematical representation of the Elastic Net objective function is as follows:

### Objective Function:

For a multiple linear regression with $n$ observations, $p$ features, a response variable $Y$, and a matrix of predictor variables $X$, the Elastic Net objective function is:
""")
        st.latex(r'''{minimize } \frac{1}{2n} \left\|Y - Xβ\right\|^2 + \lambda \left(α \|β\|_1 + \frac{1}{2} (1 - α) \|β\|_2^2\right)''')
        st.markdown("""
- $Y$ is the response variable.
- $X$ is the matrix of predictor variables.
- $β$ is the vector of coefficients.
- $\lambda$ is the regularization parameter that controls the strength of the regularization.
- $α$ is the mixing parameter that determines the combination of L1 and L2 penalties.
- $\|β\|_1$ is the L1 norm (sum of absolute values of coefficients).
- $\|β\|_2$ is the L2 norm (Euclidean norm or sum of squared values of coefficients).

### Elastic Net Cost Function:

The cost function to be minimized is a combination of the residual sum of squares (RSS) and the penalties:
""")
        st.latex(r'''J(β) = \frac{1}{2n} \left\|Y - Xβ\right\|^2 + \lambda \left(α \|β\|_1 + \frac{1}{2} (1 - α) \|β\|_2^2\right) ''')
        st.markdown("""
The first term represents the RSS (ordinary least squares), and the second term is the regularization term. The regularization term consists of the L1 penalty term ($α \|β\|_1$) and the L2 penalty term (${1}/{2} (1 - α) \|β\|_2^2$), weighted by the mixing parameter $α$.

### Optimization:

The goal is to find the values of $β$ that minimize the cost function. This can be achieved using optimization techniques such as gradient descent or coordinate descent.

### Note:

- When $α = 1$, it becomes Lasso regression.
- When $α = 0$, it becomes Ridge regression.
- For $0 < α < 1$, it is a combination of both L1 and L2 regularization.

The choice of $α$ and $\lambda$ is crucial and depends on the specific requirements and characteristics of the dataset. Cross-validation is often used to find the optimal values for these hyperparameters.
""")
        alpha=  st.slider('alpha',min_value=0.001, max_value=1.0, value=0.0, step=0.005 )

        l1_ratio=  st.slider('l1_ratio',min_value=0.0, max_value=2.0, value=0.5, step=0.5 )

        elastic_net = ElasticNet(alpha=alpha, l1_ratio=0.5)  

        elastic_net.fit(X_train, y_train)

        en_predictions = elastic_net.predict(X_test)

        dict_predictions = {"year":list(range(2010,2022)),"shdi":en_predictions}

        df_predictions = pd.DataFrame(dict_predictions)

        en_rmse = np.sqrt(mean_squared_error(y_test, en_predictions))
        print(f'en RMSE: {en_rmse}')

        # Create Line Plot 1
        trace1 = go.Scatter(x=df['year'], y=df['shdi'], mode='lines', name='HDI - Actual', line=dict(color='mediumturquoise'))

        # Create Line Plot 2
        trace2 = go.Scatter(x=df_predictions['year'], y=df_predictions['shdi'], mode='lines', line_dash='dash',name='HDI - Predicted', line=dict(color='red'))

        # Create Layout
        layout = go.Layout(title='HDI - Actual vs Predicted', xaxis=dict(title='Year'), yaxis=dict(title='HDI'))
        fig = go.Figure(data=[trace1, trace2], layout=layout)

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)



with tab3:

    st.subheader("Classification with K-Nearest Neighbors")
    st.markdown("""
k-Nearest Neighbors (kNN) is a simple and intuitive classification algorithm that works based on the distance between data points. Given a dataset with labeled instances, kNN classifies a new instance by considering the majority class among its k-nearest neighbors in the feature space. Here's the mathematical representation of kNN classification:

### Notation:

- $X_i$ represents the feature vector of the $i$-th instance in the dataset.
- $Y_i$ is the corresponding class label of the $i$-th instance.
- $X_{{new}}$ is the feature vector of the new instance to be classified.
- $k$ is the number of nearest neighbors to consider.

### kNN Algorithm:

1. **Compute Distance:**
   - For each instance $i$ in the dataset, calculate the distance between $X_{{new}}$ and $X_i$. Common distance metrics include Euclidean distance, Manhattan distance, etc.

   ${Distance}(X_{{new}}, X_i) = \sqrt{\sum_{j=1}^{n} (X_{{new}, j} - X_{i, j})^2} $

   where $n$ is the number of features.

2. **Sort Instances:**
   - Sort the instances based on their distances to $X_{{new}}$ in ascending order.

3. **Count Votes:**
   - Consider the labels of the first $k$ instances (the k-nearest neighbors).
   - Count the number of occurrences of each class.

4. **Majority Vote:**
   - Assign $X_{{new}}$ to the class that has the highest count among the k-nearest neighbors.

### Mathematical Representation:

Given a set of labeled instances $\{(X_1, Y_1), (X_2, Y_2), \ldots, (X_m, Y_m)\}$, where $m$ is the number of instances, the kNN classification for a new instance $X_{{new}}$ can be expressed as:

$ \hat{Y}_{{new}}$ = $arg max_{y} \sum_{i=1}^{k} I(Y_i = y) $

where $\hat{Y}_{{new}}$ is the predicted class for $X_{{new}}$, $I(\cdot)$ is the indicator function, and $y$ iterates over the unique class labels of the k-nearest neighbors.

In words, $\hat{Y}_{{new}}$ is the class that occurs most frequently among the k-nearest neighbors of $X_{{new}}$.

The choice of the distance metric and the value of $k$ are important parameters that can significantly impact the performance of the kNN algorithm and may be determined through cross-validation or other model evaluation techniques.
""")

    classifier_knn = KNeighborsClassifier(n_neighbors=4)
    selected_year= st.slider('Year',min_value=1990, max_value=2021, value=2010, step=1)
    dict_reference = {"Education Index":"edindex","Expected years of Schooling":"esch","Mean years of Schooling":"msch","Life Expentancy":"lifexp"}
    selected_category = st.selectbox("Choose Category", list(dict_reference.keys()))
    value = dict_reference[selected_category]
    data=shdi[shdi['year']==selected_year][['shdi',value,'hdi_category']]
    encoder = LabelEncoder()
    data['hdi_category'] = encoder.fit_transform(data['hdi_category'])
    data=data.dropna()


    X=data[["shdi",value]].to_numpy()
    Y=data["hdi_category"].to_numpy()



    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


    classifier_knn.fit(X_train, y_train)


    y_pred_knn = classifier_knn.predict(X_test)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = classifier_knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig=plt.figure(figsize=(8, 3))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='plasma')
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolor='k', cmap='plasma')
    plt.xlabel('HDI')
    plt.ylabel(selected_category)
    plt.title('HDI Category')
    st.pyplot(fig)

    titles_options = [
        ("Confusion matrix", None),
    ]

    class_names=["Very High", "High", "Medium","Low"]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            classifier_knn,
            X_test,
            y_test,
            display_labels=class_names,
            cmap=plt.cm.plasma,
            normalize=normalize,
        )
        disp.ax_.set_title(title)
        


    st.pyplot(plt)

    # # Create a meshgrid for decision boundary plotting
    # h = .8  # Step size in the mesh
    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # # Predict the decision boundary
    # Z = classifier_knn.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)

    # # Convert data to Altair DataFrame
    # source = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
    # source['Target'] = Y.astype(str)

    # # Create Altair scatter plot with decision boundary
    # scatter = alt.Chart(source).mark_circle(size=60).encode(
    #     x='Feature 1',
    #     y='Feature 2',
    #     color=alt.Color('Target', scale=alt.Scale(range=['#1f77b4', '#ff7f0e','#1f4714','#6f4012']), legend=None)
    # ).properties(
    #     width=600,
    #     height=400
    # )

    # decision_boundary = alt.Chart(pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel(), 'z': Z.ravel()})).mark_rect(opacity=0.8).encode(
    #     x='x:Q',
    #     y='y:Q',
    #     color=alt.Color('z:N', scale=None)
    # ).properties(
    #     width=600,
    #     height=400
    # )

    # combined_plot = (scatter + decision_boundary).interactive()
    # st.altair_chart(combined_plot, use_container_width=True)



