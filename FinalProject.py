import streamlit as st
import pandas as pd

st.title('Final Project')
st.write('''#Trying my best''')

st.sidebar.header('User Input Parameter')
st.sidebar.subheader('Please upload custom data in csv format')
 
choice = st.sidebar.radio('Choose a dataset',('Default','User-defined'),index=0)

def default_dataset(name):
  data = None
  if name == 'Iris':
     data = dataset.load_iris()
  else:
     data = dataset.load_wine()
  X = data.data
  y = data.target
  return X, y
    

def User_defined_dataset(chosen_name):
  X = []
  y = []
  X_name = []
  X1 = []
  
  if chosen_name == 'Default':
     dataset_name = st.sidebar.selectbox('Select Dataset',('Iris','Wine'))
     X, y = default_dataset(dataset_name)
     X_name = X
  else:
     upload_file = st.sidebar.file_uploader('Upload a csv',type='csv')
    
     if upload_file!=None:
        st.write(uplod_file)
        data = pd.read_csv(upload_file)
  
        y_name = st.sidebar.selectbox('Select a y variable',sorted(data))
        X_name = st.sidebar.multiselect('Select the x variable(s)',
                                        sorted(data)[1],
                                        help='You may select more than one varible')  
        y = data.loc[:,y_name]
        X = data.loc[:,X_name]
        X1 = X.select_dtypes(include=['object'])
        X2 = X.select_dtypes(exclude=['object'])
  
        if sorted(X1)!=[]:
           X1 = X1.apply(LabelEncoder().fit_transform)
           X = pd.concat([X2,X1],axis=1)
    
        y = LabelEncoder().fit_transform(y)
  
     else:
        st.write('Note: Please upload a csv file')
  return X, y, X_name, X1
 
X, y, X_name, cat_var = User_defined_dataset(chosen)
