import streamlit as st
import numpy as np
import streamlit as st 
from inference import transformer_inference, nn_inference

st.title("IRE Major Project - Author Identification")

st.sidebar.title("Author Identification")
option = st.sidebar.radio('Navigation', 
["Overview",
 "Datasets", 
 "Models", 
 "Application", ])

st.sidebar.write('''This is the final deployed application of the project undertaken by Team-21 for IRE course, 2021''')
st.sidebar.write('''The team members are - Anurag Muthlaya, Kancharla Aditya Hari, Meeraj Kanarparthi, Shivanshu Jain''')

if(option == "Application"):
    st.header("Washington Post Author Detector")
    text_input = st.text_area("Enter article", height=250)

    model_select = st.selectbox("Model Selection", ["CNN","LSTM","GRU","BERT","RoBERTa","DistilBERT"])
    if st.button("Predict") and text_input!="":
        if(model_select.lower() in ["cnn", "lstm", "gru"]):
            results = nn_inference(model_select.lower(), text_input)
        else:
            results = transformer_inference(model_select.lower(), text_input)
        st.dataframe(results)

if(option == "Overview"):
    st.header("Project Overview")
    st.write("""
    Author identification is the task of identifying the author of an unsigned text. In this project, 
    we undertake an extensive study of the domain and experiment with a variety of machine learning and
    deep learning models to solve the problem across multiple datasets. Navigate this page using the sidebar. 
    Models trained on the Washington Post articles have been deployed for demonstration in the application. 
    """)

if(option == "Datasets"):
    st.header("Datasets")
    st.write('''Three datasets with varying topical diversity and text lengths were considered for this 
    project to ensure a robust analysis''')

    st.markdown("### Spooky Authors")
    st.write('''The Spooky Authors dataset contains single line texts from the novels of famous horror writers
    It contains texts from a total of 3 authors.''')
    st.table({"Number of authors":[3], "Number of texts":[19579]})

    st.markdown("### Enron Corpus")
    st.write('''The Enron Corpus contains emails from hundreds of authors. For the project, we cleaned the data 
    to remove email headers and filtered it to the top 15 authors.''')
    st.table({"Number of authors":[15], "Number of texts":[134574]})

    st.markdown("### News Articles")
    st.write('''The News Articles dataset was created as part of the project. For this the Washington Post
    website was scraped for opinion pieces from 63 authors. It contains all opinion pieces posted as of
    October 25, 2021. A filtered version of the dataset with 25 authors was considered for the project.''')
    st.table({"Number of authors":[25], "Number of texts":[4914]})

if(option == "Models"):
    st.header("Models")
    st.write("The following models were used for this project - ")
    st.markdown("## Machine Learning")
    st.markdown("### Features")
    st.write("For these, we constructed document level features by taking average of word vectors of the document. Word2Vec, GLoVE, and fasttext vectors were used for this purpose")
    st.markdown("### Models")
    st.write("Support Vector Machine, DecisionTree, and RandomForest classifiers were used")
    st.markdown("## Deep Learning")
    st.markdown("### CNN")
    st.write("A CNN classifier network was contructed. In this, the input is first embedded using a pretrained word vectors and then passed through convolutinal filters of different sizes. The output of these filters is maxpooled and the concatenated vector is fed to a fully connected layer for classification.")
    st.write("Engineered stylometric features were also experimented with. For this, the representation obtained by concatenating the maxpooled vectors is augmented with the engineer features before classification")
    st.markdown("### RNN")
    st.write("LSTM and GRU classifiers were constructed. For this, the input is first embedded using pretrained word vectors and then passed through a bidirection, 2-layered RNN. The hidden states of the final layer and averaged, and this is passed to a fully connected layer for classification. ")
    st.write("As with CNN, engineered stylometric features were experimented with. This time the averaged hidden states representation was augmented with the engineered features")
    st.markdown("### Transformer")
    st.write("For this, pretrained BERT, RoBERTa and DistilBERT networks were finetuned to the classification task.")

    





    


            



