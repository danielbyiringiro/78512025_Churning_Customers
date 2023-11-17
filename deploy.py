# Import necessary libraries
import streamlit as st
import pandas as pd
from keras.models import load_model
import pickle

# Function to preprocess input data
def preprocess_input(data):

    # Convert categorical features to numeric
    data['Contract'] = data['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    data['PaymentMethod'] = data['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
    data['InternetService'] = data['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2})
    data['gender'] = data['gender'].map({'Female': 0, 'Male': 1})
    data['TechSupport'] = data['TechSupport'].map({'No': 0, 'Yes': 1})
    data['Partner'] = data['Partner'].map({'No': 0, 'Yes': 1})
    data['PaperlessBilling'] = data['PaperlessBilling'].map({False: 0, True: 1})

    # load the scaler

    scaler = pickle.load(open('scaler.pkl', 'rb'))

    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Transform the numerical features

    data[numerical_features] = scaler.transform(data[numerical_features])
    
    # load pca

    pca = pickle.load(open('pca.pkl', 'rb'))

    features = ['MonthlyCharges', 'TotalCharges', 'tenure', 'Contract', 'PaymentMethod', 'InternetService', 'PaperlessBilling', 'gender', 'TechSupport', 'Partner']
    data_input = data[features]

    # Transform the features

    data = pca.transform(data_input)

    return data

# Function to load the trained model
def load():
    # Load your pre-trained model
    model = load_model('model.h5')
    return model

# Function to make predictions
def predict_churn(model, input_data):
    
    prediction = model.predict(input_data)
    return prediction

# Streamlit app
def main():
    # Set the title of the app
    st.title("Churn Prediction App")

    # Create input form for user to enter feature values
    st.header("User Input Features")

    # Create input fields for each feature
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=0.0)
    tenure = st.number_input("Tenure", min_value=0, max_value=100, value=0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    paperless_billing = st.checkbox("Paperless Billing")
    gender = st.selectbox("Gender", ["Male", "Female"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    predict = st.button("Predict")

    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'MonthlyCharges': [monthly_charges],
        'Contract': [contract],
        'PaymentMethod': [payment_method],
        'InternetService': [internet_service],
        'PaperlessBilling': [paperless_billing],
        'gender': [gender],
        'TechSupport': [tech_support],
        'Partner': [partner],
        'TotalCharges': [total_charges],
        'tenure': [tenure]
    })

    if predict:
        # Preprocess the input data
        input_data = preprocess_input(input_data)

        # Load the pre-trained model
        model = load()

        # Make predictions
        prediction = predict_churn(model, input_data)

        # Display the prediction result
        st.subheader("Prediction")
        if prediction[0][0] >= 0.5:
            st.write(f"The customer is predicted to churn. Confidence: {round(prediction[0][0] * 100, 2)} %.")
        else:
            st.write(f"The customer is predicted not to churn. Confidence: {round(prediction[0][0] * 100 * 2, 2)} %.")

# Run the Streamlit app
if __name__ == "__main__":
    main()

