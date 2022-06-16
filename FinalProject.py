import streamlit as st
import numpy as np
import pandas as pd

from sklearn import datasets as ds  # data available are iris, digits, wine, breast_cancer, diabetes (reg)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.header('FINAL PROJECT:')
st.title('Classification Machine Learning Web App')
st.write('# Bismillah')

st.sidebar.header('User Input Parameter')
st.sidebar.caption('You may select the data category below')

# Radio selector
selector = st.sidebar.radio('Choose a dataset',('Default', 'User-defined '),index = 0)

# The default data selection (A, A in element of C)
def default_dataset(data_name):
    dataset = None
    if data_name == 'Breast Cancer':
        dataset = ds.load_breast_cancer()
    elif data_name == 'Digits':
        datset = ds.load_digits()
    else:
        dataset = ds.load_iris()
    X = dataset.data
    y = dataset.target
    return X, y
  
# Dataset processor (C)
def user_defined_dataset(category):
    X=[]
    y=[]
    X_features = []
    X1 = []
    # Default category (A)
    if category == 'Default':
       dataset_selection = st.sidebar.selectbox(
                            'Select Dataset',
                            ('Breast Cancer', 'Digits', 'Iris'))
       X, y = default_dataset(dataset_selection)
       X_features = X
    # User self-upload dataset (B)
    else:
        uploaded_file = st.sidebar.file_uploader('Upload a CSV', type='csv')
        
        if uploaded_file!=None:
           st.write(uploaded_file)
           uploaded_data = pd.read_csv(uploaded_file)
           y_target = st.sidebar.selectbox('Select y variable', sorted(uploaded_data))
           X_features = st.sidebar.multiselect('Select x variable(s)', sorted(uploaded_data), default = sorted(uploaded_data)[1],
                     help = "You may select more than one predictor")

           # defines x and y variable
           y = uploaded_data.loc[:,y_target]
           X = uploaded_data.loc[:,X_features]
           X1 = X.select_dtypes(include=['object'])
           X2 = X.select_dtypes(exclude=['object'])

          # Change categorical value into discrete 
           if sorted(X1) != []:
                  X1 = X1.apply(LabelEncoder().fit_transform)
                  X = pd.concat([X2,X1],axis=1)
           y = LabelEncoder().fit_transform(y)
        else:
           st.write('Note: Please upload a CSV file to continue this program.')

    return X,y, X_features, X1

X, y , X_features, cat_var= user_defined_dataset (selector)
