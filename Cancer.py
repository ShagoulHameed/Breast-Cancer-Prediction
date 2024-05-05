import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # Import SVM
from sklearn.metrics import accuracy_score, classification_report
import pickle
import streamlit as st

# Load the dataset
data = pd.read_csv("C:\\Users\\HameedS\\Desktop\\New folder\\Breast Cancer\\cancer.csv")


print("Missing values:\n", data.isna().sum())

data = data.loc[:, ~data.columns.duplicated()]

data.dropna(axis=1, inplace=True)


lb = LabelEncoder()
data.loc[:, 'diagnosis'] = lb.fit_transform(data['diagnosis'])



sns.countplot(x=data['diagnosis'], label="count", orient='h')
plt.title('Diagnosis Distribution')
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(data.iloc[:, 1:].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

#------------->Logistic Regression START--------------<
# logistic_model = LogisticRegression()
# logistic_model.fit(X_train, y_train)

# y_pred = logistic_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print("Classification Report Logistic Regression:\n", classification_report(y_test, y_pred))

# with open('model.pkl', 'wb') as model_file:
#     pickle.dump(logistic_model, model_file)


# with open('model.pkl', 'rb') as model_file:
#     loaded_model = pickle.load(model_file)

#------------->Logistic Regression END--------------<

#------------->SVM START--------------------<
svm_model = SVC(kernel='linear')  
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report SVM:\n", classification_report(y_test, y_pred))


with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm_model, model_file)


with open('svm_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
#------------->SVM END--------------------<




@st.cache_data
#------------->SVM START--------------------<
def load_model():
    with open('svm_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model
#------------->SVM END--------------------<

#------------->Logistic Regression START--------------<
# def load_model():
#     with open('model.pkl', 'rb') as model_file:
#         model = pickle.load(model_file)
#     return model
#------------->Logistic Regression END--------------<

def predict(model, data):
    
    data = data.dropna(axis=1)
    lb = LabelEncoder()
    #data['diagnosis'] = lb.fit_transform(data['diagnosis'])
    data.loc[:, 'diagnosis'] = lb.fit_transform(data['diagnosis'])    
    X = data.iloc[:, 2:].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    
    predictions = model.predict(X)
    return predictions


def main():
   
    st.set_page_config(page_title="Breast Cancer Prediction App", page_icon="🩺")

    
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

    
    st.title('Breast Cancer Prediction')
    st.image("BCancer.jpg", width=200) 

    data = pd.read_csv("C:\\Users\\HameedS\\Desktop\\New folder\\Breast Cancer\\cancer.csv")

  
    st.subheader("Dataset:")
    st.dataframe(data)

 
    model = load_model()

  
    predictions = predict(model, data)

    predictions_with_id = pd.DataFrame({"ID": data['id'], "Prediction": predictions})


    predictions_with_id['Diagnosis'] = predictions_with_id['Prediction'].apply(lambda x: 'Benign' if x == 0 else 'Malignant')

   
    predictions_with_id = predictions_with_id[['ID', 'Prediction', 'Diagnosis']]

    
    st.subheader("Predictions:")
    st.write(predictions_with_id)
    #st.write(predictions_with_id, header=True)


if __name__ == "__main__":
    main()