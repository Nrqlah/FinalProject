import streamlit as st
import pandas as pd

st.write('FINAL PROJECT :')
st.title('Classification Machine Learning Web App')
st.write('''#Trying my best''')

st.sidebar.header('User Input Parameter')
st.sidebar.subheader('''Please upload custom data in csv format''')

# User category dataset
 
chosen=st.sidebar.radio('Choose a dataset', ('Default','User-defined'), index=0)

# Function for user select data category (C, set C)
  
def User_defined_dataset(chosen_name):
    X = []
    y = []
    X_features = []
    X1 = []
  
    upload_file = st.sidebar.file_uploader('Upload a csv',type='csv')
           data=pd.read_csv(upload_file)
           y_target = st.sidebar.selectbox('Select a y variable',sorted(data))
           X_features = st.sidebar.multiselect('Select the x variable(s)',
                                                sorted(data)[1],
                                                help='You may select more than one variable')  
           y = data.loc[:,y_target]    # declare the y variable
           X = data.loc[:,X_features]    # declare the x variable(s)
           X1 = X.select_dtypes(include=['object'])
           X2 = X.select_dtypes(exclude=['object'])

           if sorted(X1)!=[]:
              X1 = X1.apply(LabelEncoder().fit_transform)   # Transform x categorical into discrete
              X = pd.concat([X2,X1],axis=1)

           y = LabelEncoder().fit_transform(y)              # Transform y categorical into discrete

        else:
           st.write('Note: Please upload a csv file')
    return X ,y ,X_features ,X1 

# classifier model set up

classifiers = st.sidebar.selectbox('Select classifier',('KNN','SVM','Random Forest'))
                                                      
## Testing and Training set
                                                       
test_ratio = st.sidebar.slider('Select testing set size',min_value=0.1,max_value=0.3,value=0.2)                                  
random_state = st.sidebar.slider('Select random state',1,9999,value=1234)
                                                       
st.write('## 1: Summary of X variables')
                                                       
                    
