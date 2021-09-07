import pandas as pd
import numpy as np
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st


st.write("""
# Weight Prediction
Determine the weight of a person using machine learning and python!
""")

image = Image.open('C:/Users/msdsa/Desktop/Weight Prediction/Weight.png')
st.image(image, caption='ML', use_column_width=True)


df = pd.read_csv('C:/Users/msdsa/Desktop/Weight Prediction/Person_Gender_Height_Weight_Index.csv')

st.subheader('Data Information : ')

st.dataframe(df)

st.write(df.describe())
st.sidebar.header('User Input Parameters')
st.sidebar.subheader('Height is in Cms, Gender : 0 stands for female and 1 for male')


def user_input_features():
    height = st.sidebar.slider('height', 130, 210, 150)
    body_mass = st.sidebar.slider('body_mass', 0, 5, 2)
    gender = st.sidebar.slider('gender', 0, 1, 0)

    data = {'height': height,
            'body_mass': body_mass,
            'gender': gender
            }

    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

image = Image.open('C:/Users/msdsa/Desktop/Weight Prediction/DataViz.png')
st.image(image, use_column_width=True)

st.subheader('User Input parameters')
st.write(df)
# reading csv file
data = pd.read_csv('C:/Users/msdsa/Desktop/Weight Prediction/Person_Gender_Height_Weight_Index.csv')
X = np.array(data[['Height', 'Body_Mass', 'Gender']])
Y = np.array(data['Weight'])
# random forest model
rfc = RandomForestClassifier()
rfc.fit(X, Y)

prediction = rfc.predict(df)
prediction_proba = rfc.predict_proba(df)
st.subheader('Predicted Weight in kg')
st.write(prediction)