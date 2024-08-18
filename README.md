# Breast Cancer Detection App
Streamlit Cloud Demo link :: https://breast-cancer-prediction-kmuoquwks6vixhlco3srec.streamlit.app/
![image](https://github.com/ShagoulHameed/Breast-Cancer-Prediction/assets/154894802/a72718d9-c93e-4ab8-a038-83d666a4d82c)

## Home


This app classifies breast cancer tumors as benign or malignant.

### Breast Cancer: An Overview
![image](https://github.com/ShagoulHameed/Breast-Cancer-Prediction/assets/154894802/fed1e8ea-71c5-4f88-906d-ed023c656f10)


### Benign vs. Malignant Tumors

![image](https://github.com/ShagoulHameed/Breast-Cancer-Prediction/assets/154894802/f56b1c61-2588-44ea-a522-d105ddd3ce4b)


### Classification of Breast Cancer Tumors

![image](https://github.com/ShagoulHameed/Breast-Cancer-Prediction/assets/154894802/ea1c1110-ba63-4d08-856d-774bee0f896c)


## Exploratory Data Analysis

### Exploratory Data Analysis

Let's explore the dataset.
![image](https://github.com/ShagoulHameed/Breast-Cancer-Prediction/assets/154894802/bfffb5ac-d442-4d11-98f7-e9dd11959262)

#### Dataset Summary
```

st.write(df_cancer.head())

st.write(df_cancer.describe())
```
![image](https://github.com/ShagoulHameed/Breast-Cancer-Prediction/assets/154894802/9d15fc66-f395-44d9-8888-dc05baca4ef2)

#### Target Variable Distribution

This section provides insights into the distribution of the target variable in the dataset. Understanding the distribution of the target variable is crucial as it helps in determining the class balance, which in turn affects the modeling approach and performance evaluation metrics.

```
st.write("Target Variable Distribution:")
st.write(df_cancer['label'].value_counts())
```
![image](https://github.com/ShagoulHameed/Breast-Cancer-Prediction/assets/154894802/b8901c36-415f-4e8f-8805-4dd462ab1cef)

#### Correlation Heatmap
The correlation heatmap visualizes the pairwise correlations between the features in the dataset. It helps in identifying patterns and relationships between variables. This information is valuable for feature selection, identifying multicollinearity, and understanding how features influence the target variable.

```
st.write("Correlation Heatmap:")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_cancer.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)
```
![image](https://github.com/ShagoulHameed/Breast-Cancer-Prediction/assets/154894802/bed4fc7b-1882-49da-bff7-eb631b52aca4)

#### Model Building

This section involves training machine learning models on the dataset. It typically includes steps such as data preprocessing, splitting the data into training and testing sets, selecting a suitable model, training the model on the training data, and evaluating its performance.

Let's build and evaluate machine learning models.

#### Model Selection

Model selection involves choosing the most appropriate machine learning algorithm for the given task. It may involve comparing the performance of different models using cross-validation or other evaluation techniques to select the best-performing model.

```
model_selection = st.selectbox("Select Model", ("Logistic Regression", "Support Vector Machine (SVM)"))

if model_selection == "Logistic Regression":
    # Build Logistic Regression model
    st.write("Building Logistic Regression model...")
    model = LogisticRegression()
elif model_selection == "Support Vector Machine (SVM)":
    # Build SVM model
    st.write("Building Support Vector Machine (SVM) model...")
    model = SVC()
```
![image](https://github.com/ShagoulHameed/Breast-Cancer-Prediction/assets/154894802/550e85db-9a96-4d22-8cd5-56f89bdf2bef)

#### Model Selection
Model evaluation assesses the performance of the trained model on unseen data. It typically involves calculating various metrics such as accuracy, precision, recall, F1-score, and ROC-AUC to measure how well the model generalizes to new data.

```
st.write(f"Evaluating {model_selection} model...")
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
st.write(f"Accuracy on training data: {accuracy_train:.2f}")
st.write(f"Accuracy on test data: {accuracy_test:.2f}")
```

#### Classification Report

The classification report provides a summary of the model's performance for each class in the target variable. It includes metrics such as precision, recall, F1-score, and support for each class, providing insights into the model's predictive capabilities for different classes.

```
st.write("Classification Report:")
classification_report_df = pd.DataFrame(classification_report(y_test, y_pred_test, output_dict=True))
st.write(classification_report_df)
```
![image](https://github.com/ShagoulHameed/Breast-Cancer-Prediction/assets/154894802/76428c28-4db3-463d-bccc-dede7f84f042)

#### Confusion Matrix

The confusion matrix is a tabular representation of the model's predictions compared to the actual values. It provides insights into the true positives, true negatives, false positives, and false negatives, allowing for a more detailed understanding of the model's performance across different classes.

```
st.write("Confusion Matrix:")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)
```
![image](https://github.com/ShagoulHameed/Breast-Cancer-Prediction/assets/154894802/7919a7c0-a295-4c50-b85c-67690ba3884d)

####  Sample Input Data for Prediction

This section allows users to input their own data into the model for prediction. It enables users to see how the model performs on new, unseen data and provides a practical demonstration of the model's capabilities.

```
st.subheader("Sample Input Data for Prediction (Optional)")
input_data_str = st.text_input("Enter comma-separated values
```
![image](https://github.com/ShagoulHameed/Breast-Cancer-Prediction/assets/154894802/90f06c71-5ca1-451e-8966-2be1afae6f05)

[LInkedIN](https://www.linkedin.com/in/shagoul-hameed/) 
