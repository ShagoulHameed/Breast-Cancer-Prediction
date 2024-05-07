# Breast Cancer Detection App

## Home

This app classifies breast cancer tumors as benign or malignant.

### Breast Cancer: An Overview

Breast cancer is a type of cancer that develops in the breast tissue. It is the most common cancer among women worldwide and can also affect men, although it is less common. Breast cancer usually begins in the cells of the milk-producing ducts (ductal carcinoma) or the lobules (lobular carcinoma) of the breast. From there, it can spread to other parts of the body if not detected and treated early.

### Benign vs. Malignant Tumors

- **Benign tumors**: Non-cancerous growths that do not spread to other parts of the body. While they may grow in size and cause health issues if they press on nearby organs or tissues, they do not invade surrounding tissue or metastasize (spread) to other parts of the body.
- **Malignant tumors**: Cancerous growths that have the potential to invade surrounding tissues and spread to other parts of the body through the bloodstream or lymphatic system. Malignant tumors can be life-threatening if left untreated and require prompt medical attention.

### Classification of Breast Cancer Tumors

In the context of breast cancer, tumors are classified as either benign or malignant based on their characteristics and behavior. This classification is crucial for determining the appropriate treatment approach and prognosis for the patient. Diagnostic tests such as biopsies, imaging studies (e.g., mammograms), and laboratory tests help healthcare providers differentiate between benign and malignant breast tumors.

## Exploratory Data Analysis

### Exploratory Data Analysis

Let's explore the dataset.

#### Dataset Summary
```

st.write(df_cancer.head())

st.write(df_cancer.describe())
```
#### Target Variable Distribution

This section provides insights into the distribution of the target variable in the dataset. Understanding the distribution of the target variable is crucial as it helps in determining the class balance, which in turn affects the modeling approach and performance evaluation metrics.

```
st.write("Target Variable Distribution:")
st.write(df_cancer['label'].value_counts())
```

#### Correlation Heatmap
The correlation heatmap visualizes the pairwise correlations between the features in the dataset. It helps in identifying patterns and relationships between variables. This information is valuable for feature selection, identifying multicollinearity, and understanding how features influence the target variable.

```
st.write("Correlation Heatmap:")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_cancer.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)
```

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

#### Confusion Matrix

The confusion matrix is a tabular representation of the model's predictions compared to the actual values. It provides insights into the true positives, true negatives, false positives, and false negatives, allowing for a more detailed understanding of the model's performance across different classes.

```
st.write("Confusion Matrix:")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)
```

####  Sample Input Data for Prediction

This section allows users to input their own data into the model for prediction. It enables users to see how the model performs on new, unseen data and provides a practical demonstration of the model's capabilities.

```
st.subheader("Sample Input Data for Prediction (Optional)")
input_data_str = st.text_input("Enter comma-separated values
```
