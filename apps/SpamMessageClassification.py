import streamlit as st


 #!! 2) Building the Model (Random Forest)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
classifier = Pipeline([("tfidf", TfidfVectorizer()) , ("classifier", RandomForestClassifier(n_estimators=100))])

#!! 4) Building the Model (SVM)
from sklearn.svm import SVC
svm = Pipeline([("tfidf", TfidfVectorizer()) , ("classifier", SVC(C = 100, gamma='auto'))])

def app():
     
    functionalities(classifier,svm)  
    controls(classifier, svm)

def functionalities(classifier,svm):
    #!! 1) Data Preprocessing
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    expander = st.expander("See explanation", expanded=False)
    df = pd.read_csv('data\spam.tsv', sep='\t')
    expander.write(df.head())
    expander.write(df.isna().sum())
    expander.write(df.tail())
    expander.write(df.describe())
    expander.write(df['label'].value_counts()/ (len(df)))
    expander.write(df['label'].value_counts())

    ham = df[df['label'] == 'ham']
    spam = df[df['label'] == 'spam']
    ham = ham.sample(spam.shape[0])
    expander.write(ham.shape)
    expander.write(spam.shape)

    data = ham.append(spam, ignore_index=True)
    expander.write(data.shape)
    expander.write(data['label'].value_counts())
    expander.write(data.head())

    plt.hist(data[data['label'] == 'ham']['length'], bins = 100, alpha = 0.7)
    plt.hist(data[data['label'] == 'spam']['length'], bins = 100, alpha = 0.7)
    expander.pyplot(plt)

    plt.hist(data[data['label'] == 'ham']['punct'], bins = 100, alpha = 0.7)
    plt.hist(data[data['label'] == 'spam']['punct'], bins = 100, alpha = 0.7)
    expander.pyplot(plt)

    expander.write(data)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test =  train_test_split(data['message'], data['label'], test_size = 0.3, random_state =0, shuffle = True)
    expander.write(X_train.shape)
    expander.write(X_test.shape)

    #!! 3) Predicting the results (Random Forest)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    expander.write(y_test)
    expander.write(y_pred)
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    accuracy_score(y_test, y_pred)
    confusion_matrix(y_test, y_pred)
    expander.write(classification_report(y_test, y_pred))

    #!! 5) Predicting the results (SVM)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy_score(y_test, y_pred)
    confusion_matrix(y_test, y_pred)
    expander.write(classification_report(y_test, y_pred))

    # Testing
    test1 = ['Hello, You are learning natural Language Processing']
    test2 = ['Hope you are doing good and learning new things !']
    test3 = ['Congratulations, You won a lottery ticket worth $1 Million ! To claim call on 446677']

    expander.write(classifier.predict(test1))
    expander.write(classifier.predict(test2))
    expander.write(classifier.predict(test3))

    expander.write(svm.predict(test1))
    expander.write(svm.predict(test2))
    expander.write(svm.predict(test3)) 

def controls(classifier, svm):
    st.write("SPAM MESSAGE CLASSIFICATION")
     #!! Streamlit - st starts 
    # FN: Define input text 
    def get_text():
        input_text = st.text_input("You: ", key="input")
        return input_text
    # VAR: Get input text value
    user_input = get_text()
    # VAR: Define button 
    submit = st.button('Generate')  

    response = ""
    if(user_input != ""):
        response = getResponse(classifier, svm, user_input)
        
    # EVENT: Submit click 
    if submit:
        st.subheader("Answer:")
        if response == "":
            st.write("enter text to check SPAM")
        else:
            st.write('Classifier Prediction:', response["ClassifierPred"]) 
            st.write('SVM Prediction:', response["SVMPred"]) 
            
    #!! Streamlit - st ends
def getResponse(classifier, svm, user_input):
    # Testing
    test1 = [user_input] 
    
    dictPredictedValues = {
        "ClassifierPred": classifier.predict(test1),
        "SVMPred": svm.predict(test1) 
    }
    return dictPredictedValues
 

    

