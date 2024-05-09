import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


cancer = load_breast_cancer()
df_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df_cancer['label'] = cancer['target']



def main():

    st.set_page_config(page_title="Breast Cancer Prediction App", page_icon="🩺")
    st.title("Breast Cancer Detection App")
    
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
      
    st.image("BCancer.jpg", width=300 ) 

    
    

    st.sidebar.title("Menu")
    options = st.sidebar.radio("Select Option", ("Home", "Exploratory Data Analysis", "Model Building"))

    if options == "Home":
        
        st.write("This app classifies breast cancer tumors as benign or malignant.")
        
        st.write("Welcome to the Breast Cancer Detection App!")
        st.write("Please select an option from the sidebar.")

        st.header("Breast Cancer: An Overview")
        st.write("Breast cancer is a type of cancer that develops in the breast tissue. It is the most common cancer among women worldwide and can also affect men, although it is less common. Breast cancer usually begins in the cells of the milk-producing ducts (ductal carcinoma) or the lobules (lobular carcinoma) of the breast. From there, it can spread to other parts of the body if not detected and treated early.")
        
        st.header("Benign vs. Malignant Tumors")
        st.write("Benign tumors: These are non-cancerous growths that do not spread to other parts of the body. While they may grow in size and cause health issues if they press on nearby organs or tissues, they do not invade surrounding tissue or metastasize (spread) to other parts of the body.")
        st.write("Malignant tumors: These are cancerous growths that have the potential to invade surrounding tissues and spread to other parts of the body through the bloodstream or lymphatic system. Malignant tumors can be life-threatening if left untreated and require prompt medical attention.")
        
        st.header("Classification of Breast Cancer Tumors")
        st.write("In the context of breast cancer, tumors are classified as either benign or malignant based on their characteristics and behavior. This classification is crucial for determining the appropriate treatment approach and prognosis for the patient. Diagnostic tests such as biopsies, imaging studies (e.g., mammograms), and laboratory tests help healthcare providers differentiate between benign and malignant breast tumors.")


    elif options == "Exploratory Data Analysis":
        st.subheader("Exploratory Data Analysis")
        st.write("Let's explore the dataset.")

        
        st.write(df_cancer.head())

        
        #st.write(df_cancer.info())

        
        st.write("Dataset description:")
        st.write(df_cancer.describe())

       
        st.write("Target Variable Distribution:")
        st.write(df_cancer['label'].value_counts())

        
        st.write("Correlation Heatmap:")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_cancer.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

    elif options == "Model Building":
        
        st.subheader("Model Building")
        st.write("Let's build and evaluate machine learning models.")

        
        X = df_cancer.drop(columns=['label'], axis=1)
        y = df_cancer['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

        
        model_selection = st.selectbox("Select Model", ("Logistic Regression", "Support Vector Machine (SVM)", "Voting Classifier"))

        if model_selection == "Logistic Regression":
            #  Logistic Regression model
            st.write("Building Logistic Regression model...")
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

        elif model_selection == "Support Vector Machine (SVM)":
            #  SVM model
            st.write("Building Support Vector Machine (SVM) model...")
            model = SVC()
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
        elif  model_selection ==   "Voting Classifier":
            # Voting Classifier model
            st.write("Building Voting Classifier model...")
            log_reg = LogisticRegression()
            svm = SVC()
            model = VotingClassifier(estimators=[('lr', log_reg), ('svm', svm)], voting='hard')
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)


        
        st.write(f"Evaluating {model_selection} model...")
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        st.write(f"Accuracy on training data: {accuracy_train:.2f}")
        st.write(f"Accuracy on test data: {accuracy_test:.2f}")

        #  classification 
        st.write("Classification Report:")
        classification_report_df = pd.DataFrame(classification_report(y_test, y_pred_test, output_dict=True))
        st.write(classification_report_df)

        
        cm = confusion_matrix(y_test, y_pred_test)

        
        confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                                columns=['predicted_cancer', 'predicted_healthy'])

        
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        
        st.subheader("Sample Input Data for Prediction (Optional)")
        input_data_str = st.text_input("Enter comma-separated values for the input data (optional)")
        if input_data_str:
            input_data = [float(x.strip()) for x in input_data_str.split(',')]
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
            prediction = model.predict(input_data_reshaped)

            
            if prediction[0] == 0:
                st.write('The tumor is Malignant for this given data')
            else:
                st.write('The tumor is Benign for this given data')
        else:
            # Default result using test data
            default_prediction = model.predict(X_test)
            st.write("Default Prediction using Test Data:")
            st.write(default_prediction)


if __name__ == "__main__":
    main()
