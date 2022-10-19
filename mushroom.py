import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

st.markdown("<h1 style='text-align: center; color: black;'>Mushroom/Car Classification</h1>", unsafe_allow_html=True)


st.write('This is a simple web app to show how KNN behaves on two datasets')
st.write('The datasets can be found at: https://archive.ics.uci.edu/ml/datasets/Mushroom and https://archive.ics.uci.edu/ml/datasets/car+evaluation')


#create radio buttons for dataset selection.
dataset = st.radio('Select Dataset', ('Mushroom', 'Car'))

#select the nominalization method
nominalization = st.radio('Select Nominalization Method', ('pd.get_dummies()', 'Data Slayers™️ Original Nominalization Method'))

distance_metric = st.radio('Select Distance Metric', ('Minkowski', 'Euclidean' ,'Data Slayers™️ Original Distance Metric'))

#create a slider for k selection
k = st.slider('k',min_value = 1, max_value = 10, value = 5)

split_ratio = st.slider('Train/Test Split Ratio', min_value = 0.1, max_value = 0.9, value = 0.7)

#load the data
if dataset == 'Mushroom':
    df = pd.read_csv('data/mushroom/agaricus-lepiota.data', header=None)
    df.columns = ['class','cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
elif dataset == 'Car':
    df = pd.read_csv('data/car/car.data', header=None)
    df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety','class']
    

def nominalize(df):
    
    return df
# Nominalize the data
if nominalization == 'pd.get_dummies()':
    #get dummies for everything except the class
    if dataset == 'Mushroom':
        df = pd.get_dummies(df, columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])
    elif dataset == 'Car':
        df = pd.get_dummies(df, columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
    
elif nominalization == 'Data Slayers™️ Original Nominalization Method':
    df = nominalize(df)

#button to hide the dataframe
if st.button('Hide Dataframe'):
    st.write('Dataframe Hidden')

#button to show the dataframe
if st.button('Show Dataframe'):
    st.write(df)




X = df.drop('class', axis=1)
y = df['class']
#split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=split_ratio, random_state=42, shuffle=True)

def ds_distance(x, y, p=2):
    return 0


if distance_metric == 'Minkowski':
    distance = 'minkowski'
elif distance_metric == 'Euclidean':
    distance = 'euclidean'
elif distance_metric == 'Data Slayers™️ Original Distance Metric':
    distance = ds_distance

#train the model
knn = KNeighborsClassifier(n_neighbors=k, metric=distance)

knn.fit(X_train, y_train)

#predict the test data
y_pred = knn.predict(X_test)
#get percent accuracy 
accuracy = knn.score(X_test, y_test)

# only show 4 decimal places
accuracy = round(accuracy, 4)
st.header(f'Accuracy: {accuracy}  (with k = {k})')

#if the ditance matrix, nominalization method, dataset, or ratio is changed
#rerun the code
k_list = []
accuracy_list = []
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k, metric=distance)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    accuracy = round(accuracy, 4)
    k_list.append(k)
    accuracy_list.append(accuracy)
fig, ax = plt.subplots()
ax.plot(k_list, accuracy_list)
ax.set_xlabel('k')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs k')

st.pyplot(fig)


