import streamlit as st
import pandas as pd

st.write('FINAL PROJECT :')
st.title('Classification Machine Learning Web App')
st.write('''#Trying my best''')

st.sidebar.header('User Input Parameter')
st.sidebar.subheader('''Please upload custom data in csv format''')

# User category dataset
 
chosen=st.sidebar.radio('Choose a dataset', ('Default','User-defined'), index=0)

# Function for default dataset (A, A element in C)

def default_dataset(name):
  dataset=None
  if name=='Diabetes':
       dataset=pd.read_csv('https://raw.githubusercontent.com/Nrqlah/FinalProject/main/diabetes.csv',sep=',')
  else:
       dataset=pd.read_csv('https://raw.githubusercontent.com/Nrqlah/FinalProject/main/student_mat.csv',sep=';')
  X = dataset.data
  y = dataset.target
  
  return X ,y 
    
# Function for user select data category (C, set C)
  
def User_defined_dataset(chosen_name):
    X = []
    y = []
    X_features = []
    X1 = []
  
    if chosen_name == 'Default':
       dataset_name = st.sidebar.selectbox('Select Dataset',('Diabetes','Students'))
       X,y=default_dataset(dataset_name)  # A
       X_features=X 
    # user to upload file (B)
    else:
       upload_file = st.sidebar.file_uploader('Upload a csv',type='csv')
       if upload_file!=None:
          st.write(uplod_file)
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
   
X,y,X_features,user_category=User_defined_dataset(chosen)

# classifier model set up

classifiers = st.sidebar.selectbox('Select classifier',('KNN','SVM','Random Forest'))
                                                      
## Testing and Training set
                                                       
test_ratio = st.sidebar.slider('Select testing set size',min_value=0.1,max_value=0.3,value=0.2)                                  
random_state = st.sidebar.slider('Select random state',1,9999,value=1234)
                                                       
st.write('## 1: Summary of X variables')
                                                       
                    
