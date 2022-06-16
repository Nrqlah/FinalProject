import streamlit as st
import numpy as np
import pandas as pd

from sklearn import datasets as ds  # data available are iris, digits, wine, breast_cancer, diabetes (reg)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

st.write(f"### بسم الله الرحمن الرحيم")
st.write('## FINAL PROJECT:')
st.title('Classification Machine Learning Web App')


st.sidebar.header('User Input Parameter')
st.sidebar.caption('You may select the data category below')

# Radio selector
selector = st.sidebar.radio('Choose a dataset',('Default', 'User-defined '),index = 0)

# The default data selection (A, A in element of C)
def default_dataset(data_name):
    dataset = None
    if data_name == 'Breast Cancer':
        dataset = ds.load_breast_cancer()
        X = dataset.data
        y = dataset.target
    elif data_name == 'Iris':
        dataset = ds.load_iris()
        X = dataset.data
        y = dataset.target
    else:
        dataset = pd.read_csv('https://raw.githubusercontent.com/Nrqlah/FinalProject/main/student_mat.csv', sep=';')
        X = dataset.drop(['G3'],axis=1)
        y = dataset.target['G3']
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
                            ('Breast Cancer', 'Iris','Student'))
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

# ------------------------------------
# Classidier selector box (E)
classifier = st.sidebar.selectbox('Select classifier',('KNN', 'SVM', 'Random Forest'))

# Slider of test size and random state
test_ratio = st.sidebar.slider('Select testing size or ratio', min_value= 0.10, max_value = 0.30, value=0.2)
random_state = st.sidebar.slider('Select random state range', 1, 9999,value=5555)


# Summary of X
st.subheader(' 1: Summary of X variables')
if len(X)==0:
   st.write('Note: X variables have not been selected.', unsafe_allow_html=True)
else:
   st.write('Shape of X variables :', X.shape)
   st.write('Summary of X variables:', pd.DataFrame(X).describe())

    
# Summary of y
st.subheader(' 2: Summary of y variable')
if len(y)==0:
   st.write('Note: y variable has not been selected.', unsafe_allow_html=True)
elif len(np.unique(y)) <5:
     st.write('Number of classes:', len(np.unique(y)))
else: 
   st.write("<font color='red'>Warning: System detects an unusual number of unique classes. Please make sure that the y variable is a categorical variable. Ignore this warning message if you are sure that the y is a categorical variable.</font>",
            unsafe_allow_html=True)
   st.write('Number of classes:', len(np.unique(y)))

#---------------------------------------------    
# Classifier parameter processor (F)
def parameter(classifier_name):
    par = dict()
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0,value=1.0)
        par['C'] = C
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15,value=5)
        par['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15,value=5)
        par['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100,value=10)
        par['n_estimators'] = n_estimators
    return par

par = parameter(classifier)

#--------------------------------------
# Conector between selector and parameter (E & F)
def get_classifier(classifier_name, par):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=par['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=par['K'])
    else:
        clf = RandomForestClassifier(n_estimators=par['n_estimators'], 
            max_depth=par['max_depth'], random_state=random_state)
    return clf

clf = get_classifier(classifier, par)


#---------------------------------------------------
# Report
st.subheader(' 3: Classification Report')

# Split testing and training
if len(X)!=0 and len(y)!=0: 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
  # Transform scaled data
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)    

  clf.fit(X_train_scaled, y_train)
  y_pred = clf.predict(X_test_scaled)

  st.write('Classifier:',classifier)
  st.write('Classification report:')
  report = classification_report(y_test, y_pred,output_dict=True)
  df = pd.DataFrame(report).transpose()
  st.write(df)
else: 
   st.write('Note: No classification report generated.', unsafe_allow_html=True)

#----------------------------------------------------
