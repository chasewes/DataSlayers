import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#show and center a jpg "mushroom_car.jpg"
st.image("mushroom_car.jpg", use_column_width=True, width=300)

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
    
# Data Slayers™️ Original Nominalization Method for both the cars and mushroom datasets
def nominalize(df, dataset):
    if dataset == 'Mushroom':
        # map ordinal features to integers
        # df['cap-shape'] = df['cap-shape'].map({'b': 0, 'c': 1, 'x': 2, 'f': 3, 'k': 4, 's': 5})
        # df['cap-surface'] = df['cap-surface'].map({'f': 0, 'g': 1, 'y': 2, 's': 3})
        # df['cap-color'] = df['cap-color'].map({'n': 0, 'b': 1, 'c': 2, 'g': 3, 'r': 4, 'p': 5, 'u': 6, 'e': 7, 'w': 8, 'y': 9})
        # df['bruises'] = df['bruises'].map({'t': 0, 'f': 1})
        # df['odor'] = df['odor'].map({'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8})
        # df['gill-attachment'] = df['gill-attachment'].map({'a': 0, 'd': 1, 'f': 2, 'n': 3})
        # df['gill-spacing'] = df['gill-spacing'].map({'c': 0, 'w': 1, 'd': 2})
        # df['gill-size'] = df['gill-size'].map({'b': 0, 'n': 1})
        # df['gill-color'] = df['gill-color'].map({'k': 0, 'n': 1, 'b': 2, 'h': 3, 'g': 4, 'r': 5, 'o': 6, 'p': 7, 'u': 8, 'e': 9, 'w': 10, 'y': 11})
        # df['stalk-shape'] = df['stalk-shape'].map({'e': 0, 't': 1})
        # df['stalk-root'] = df['stalk-root'].map({'b': 0, 'c': 1, 'u': 2, 'e': 3, 'z': 4, 'r': 5, '?': 6})
        # df['stalk-surface-above-ring'] = df['stalk-surface-above-ring'].map({'f': 0, 'y': 1, 'k': 2, 's': 3})
        # df['stalk-surface-below-ring'] = df['stalk-surface-below-ring'].map({'f': 0, 'y': 1, 'k': 2, 's': 3})
        # df['stalk-color-above-ring'] = df['stalk-color-above-ring'].map({'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8})
        # df['stalk-color-below-ring'] = df['stalk-color-below-ring'].map({'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8})
        # df['veil-type'] = df['veil-type'].map({'p': 0, 'u': 1})
        # df['veil-color'] = df['veil-color'].map({'n': 0, 'o': 1, 'w': 2, 'y': 3})
        # df['ring-number'] = df['ring-number'].map({'n': 0, 'o': 1, 't': 2})
        # df['ring-type'] = df['ring-type'].map({'c': 0, 'e': 1, 'f': 2, 'l': 3, 'n': 4, 'p': 5, 's': 6, 'z': 7})
        # df['spore-print-color'] = df['spore-print-color'].map({'k': 0, 'n': 1, 'b': 2, 'h': 3, 'r': 4, 'o': 5, 'u': 6, 'w': 7, 'y': 8})
        # df['population'] = df['population'].map({'a': 0, 'c': 1, 'n': 2, 's': 3, 'v': 4, 'y': 5})
        # df['habitat'] = df['habitat'].map({'g': 0, 'l': 1, 'm': 2, 'p': 3, 'u': 4, 'w': 5, 'd': 6})
        # df['class'] = df['class'].map({'e': 0, 'p': 1})

        # maping features to statistics
        df['cap-shape'] = df['cap-shape'].map(df.groupby('cap-shape')['class'].mean())
        df['cap-surface'] = df['cap-surface'].map(df.groupby('cap-surface')['class'].mean())
        df['cap-color'] = df['cap-color'].map(df.groupby('cap-color')['class'].mean())
        df['bruises'] = df['bruises'].map(df.groupby('bruises')['class'].mean())
        df['odor'] = df['odor'].map(df.groupby('odor')['class'].mean())
        df['gill-attachment'] = df['gill-attachment'].map(df.groupby('gill-attachment')['class'].mean())
        df['gill-spacing'] = df['gill-spacing'].map(df.groupby('gill-spacing')['class'].mean())
        df['gill-size'] = df['gill-size'].map(df.groupby('gill-size')['class'].mean())
        df['gill-color'] = df['gill-color'].map(df.groupby('gill-color')['class'].mean())
        df['stalk-shape'] = df['stalk-shape'].map(df.groupby('stalk-shape')['class'].mean())
        df['stalk-root'] = df['stalk-root'].map(df.groupby('stalk-root')['class'].mean())
        df['stalk-surface-above-ring'] = df['stalk-surface-above-ring'].map(df.groupby('stalk-surface-above-ring')['class'].mean())
        df['stalk-surface-below-ring'] = df['stalk-surface-below-ring'].map(df.groupby('stalk-surface-below-ring')['class'].mean())
        df['stalk-color-above-ring'] = df['stalk-color-above-ring'].map(df.groupby('stalk-color-above-ring')['class'].mean())
        df['stalk-color-below-ring'] = df['stalk-color-below-ring'].map(df.groupby('stalk-color-below-ring')['class'].mean())
        df['veil-type'] = df['veil-type'].map(df.groupby('veil-type')['class'].mean())
        df['veil-color'] = df['veil-color'].map(df.groupby('veil-color')['class'].mean())
        df['ring-number'] = df['ring-number'].map(df.groupby('ring-number')['class'].mean())
        df['ring-type'] = df['ring-type'].map(df.groupby('ring-type')['class'].mean())
        df['spore-print-color'] = df['spore-print-color'].map(df.groupby('spore-print-color')['class'].mean())
        df['population'] = df['population'].map(df.groupby('population')['class'].mean())
        df['habitat'] = df['habitat'].map(df.groupby('habitat')['class'].mean())
        df['class'] = df['class'].map(df.groupby('class')['class'].mean())

    elif dataset == 'Car':
        # map ordinal features to integers

        df['buying'] = df['buying'].replace({'vhigh':3, 'high':2, 'med':1, 'low':0})
        df['maint'] = df['maint'].replace({'vhigh':3, 'high':2, 'med':1, 'low':0})
        df['doors'] = df['doors'].replace({'5more':5})
        df['persons'] = df['persons'].replace({'more':5})
        df['lug_boot'] = df['lug_boot'].replace({'small':0, 'med':1, 'big':2})
        df['safety'] = df['safety'].replace({'low':0, 'med':1, 'high':2})
        df['class'] = df['class'].replace({'unacc':0, 'acc':1, 'good':2, 'vgood':3})  
    return df

# Nominalize the data
if nominalization == 'pd.get_dummies()':
    #get dummies for everything except the class
    if dataset == 'Mushroom':
        df = pd.get_dummies(df, columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])
    elif dataset == 'Car':
        df = pd.get_dummies(df, columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
    
elif nominalization == 'Data Slayers™️ Original Nominalization Method':
    df = nominalize(df, dataset)

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


