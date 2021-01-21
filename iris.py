# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:39:06 2020

@author: Chamsedine
"""

import streamlit as st
#st.write(st.__version__)
import pandas as pd
import numpy as np
import seaborn as sns
import os
#from streamlit.errors import UnhashableType
#from PIL import Image,ImageFilter,ImageEnhance
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score,confusion_matrix,classification_report

def main(): 
    st.title("Iris EDA App")
    #st.subheader("EDA Web App with Streamlit ")

    # Your code goes below
    # Our Dataset
    my_dataset = "iris.csv"


    # To Improve speed and cache data
    @st.cache(persist=True)#=True #
    def explore_data(dataset):
    	df = pd.read_csv(os.path.join(dataset))
    	return df 

    # Load Our Dataset
    data = explore_data(my_dataset)
    
    st.subheader("Data Understanding")

    # Show Dataset
    if st.checkbox("Preview DataFrame"):
    	if st.button("Head"):
    		st.write(data.head())
    	if st.button("Tail"):
    		st.write(data.tail())
    	else:
    		st.write(data.head(2))

    # Show Entire Dataframe
    if st.checkbox("Show All DataFrame"):
    	st.dataframe(data)

    # Show All Column Names
    if st.checkbox("Show All Column Name"):
    	st.text("Columns:")
    	st.write(data.columns)

    # Show Dimensions and Shape of Dataset
    data_dim = st.radio('What Dimension Do You Want to Show',('Rows','Columns'))
    if data_dim == 'Rows':
    	st.text("Showing Length of Rows")
    	st.write(len(data))
    if data_dim == 'Columns':
    	st.text("Showing Length of Columns")
    	st.write(data.shape[1])

    # Show Summary of Dataset
    if st.checkbox("Show Summary of Dataset"):
    	st.write(data.describe())
    
    #Select your columns    
    if st.checkbox("Select Columns To Show"):
        all_columns = data.columns.tolist()
        selected_columns = st.multiselect('Select',all_columns)
        new_df = data[selected_columns]
        st.dataframe(new_df)

    # Selection of Columns
    #species_option = st.selectbox('Select Columns',('sepal_length','sepal_width','petal_length','petal_width','species'))
    #if species_option == 'sepal_length':
    	#st.write(data['sepal_length'])
    #elif species_option == 'sepal_width':
    	#st.write(data['sepal_width'])
    #elif species_option == 'petal_length':
    	#st.write(data['petal_length'])
    #elif species_option == 'petal_width':
    	#st.write(data['petal_width'])
    #elif species_option == 'species':
    	#st.write(data['species'])
    #else:
    	#st.write("Select A Column")
        
    # Iris Image Manipulation
    st.subheader("Vizualisation Dataset")

    # Show Plots
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if st.checkbox("Simple Bar Plot with Matplotlib "):
    	data.plot(kind='bar')
    	st.pyplot()
        #Viz
    #if st.checkbox("Viz Distribution"):
        #sns.pairplot(data, hue="species")
        #st.pyplot()
    
    #if st.checkbox('scatter'):
        #sns.scatterplot(data['sepal_lenght'],data['sepal_width'], hue = 'species')
        #st.pyplot
    if st.checkbox("Nuage de Point"): 
        fig = data[data.species=='setosa'].plot(kind='scatter',x='sepal_length',y='sepal_width',color='orange', label='setosa')
        data[data.species=='versicolor'].plot(kind='scatter',x='sepal_length',y='sepal_width',color='blue', label='versicolor',ax=fig)
        data[data.species=='virginica'].plot(kind='scatter',x='sepal_length',y='sepal_width',color='green', label='virginica', ax=fig)
        fig.set_xlabel("sepal_Length")
        fig.set_ylabel("sepal_width")
        fig.set_title("sepal Length VS width")
        fig=plt.gcf()
        fig.set_size_inches(10,6)
        st.pyplot()
    
    
       
    
    # Plot the training points
    if st.checkbox("Box Plot"):
        sns.boxplot(x="species", y="sepal_length", data=data)
        st.pyplot()
        
   
    
    #if st.checkbox("Dispersion Plot"):
        #sns.scatterplot('sepal_length','sepal_width', hue = 'species', data = data)
        #st.pyplot


    # Iris Image Manipulation
    #st.subheader("Image manipulation")
    #@st.cache
    #def load_image(img):
    	#im =Image.open(os.path.join(img))
    	#return im

    # Select Image Type using Radio Button
    #species = st.radio('What is the Iris Species do you want to see?',('Setosa','Versicolor','Virginica'))

    #if species == 'Setosa':
    	#st.text("Showing Setosa Species")
    	#st.image(load_image('imgs/iris_setosa.jpg'))
    #elif species == 'Versicolor':
    	#st.text("Showing Versicolor Species")
    	#st.image(load_image('imgs/iris_versicolor.jpg'))
    #elif species == 'Virginica':
    	#st.text("Showing Virginica Species")
    	#st.image(load_image('imgs/iris_virginica.jpg'))



    # Show Image or Hide Image with Checkbox
    #if st.checkbox("Show Image/Hide Image"):
    	#my_image = load_image('imgs/iris_setosa.jpg')
    	#enh = ImageEnhance.Contrast(my_image)
    	#num = st.slider("Set Your Contrast Number",1.0,3.0)
    	#img_width = st.slider("Set Image Width",300,500)
    	#st.image(enh.enhance(num),width=img_width)

    st.subheader('Machine Learning models')
 
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    #from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
 
 
    features= data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    labels = data['species'].values
 
    X_train,X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=1)
 
    alg = ['Decision Tree', 'Support Vector Machine']
    classifier = st.selectbox('Which algorithm?', alg)
    if classifier=='Decision Tree':
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        acc = dtc.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_dtc = dtc.predict(X_test)
        cm_dtc=confusion_matrix(y_test,pred_dtc)
        st.write('Confusion matrix: ', cm_dtc)
    
    elif classifier == 'Support Vector Machine':
        svm=SVC()
        svm.fit(X_train, y_train)
        acc = svm.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_svm = svm.predict(X_test)
        cm=confusion_matrix(y_test,pred_svm)
        st.write('Confusion matrix: ', cm)
        
    st.markdown("""
        Le meilleur modele à retenir est le support vector machine.
        """)
        
    #Comparaison
    #if st.checkbox("Comparaison"):
        #st.write(pd.DataFrame({
        #"True data": y_test,
        #"Predict data":pred_svm
    #}))
    
    #if st.checkbox("Courbe Roc"):
        #probs = svm.predict_proba(X_test)
        #probs = probs[:, 1]
        #auc_svm = roc_auc_score(y_test, probs)
        #print('AUC - Test Set: %.2f%%' % (auc_svm*100))
        #fpr, tpr, thresholds = roc_curve(y_test, probs)
        #plt.plot([0, 1], [0, 1], linestyle='--')
        #plt.plot(fpr, tpr, marker='.')
        #st.pyplot
    

    # About
    #if st.button("About App"):
    	#st.subheader("Iris Dataset EDA App")
    	#st.text("Built with Streamlit")
    	#st.text("Thanks to the Streamlit Team Amazing Work")

    if st.checkbox("By"):
        st.text("Aidara Chamsedine")
        st.text("aidarachamsedine10@gmail.com")


if __name__ == "__main__":
    main()
