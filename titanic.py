import pickle
import streamlit as st
import numpy as np

# Load the trained model and scaler
model = pickle.load(open('titanic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def main():
    st.title('Titanic Survival Prediction')
    
    # Text inputs
    passenger_id = st.text_input("Passenger ID")
    name = st.text_input("Name")
    ticket = st.text_input("Ticket")
    
    # Input variables from the user
    pclass = st.number_input("Passenger Class (Pclass)", min_value=1, max_value=3, value=1)
    sex = st.selectbox("Sex (1 for Female, 0 for Male)", options=[1, 0])
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0)
    sibsp = st.number_input("Number of Siblings/Spouses (SibSp)", min_value=0, value=0)
    parch = st.number_input("Number of Parents/Children (Parch)", min_value=0, value=0)
    fare = st.number_input("Fare", min_value=0.0, value=50.0)
    cabin = st.number_input("Cabin (Alloted = 1--Not Alloted = 0)", min_value=0, max_value=1,value=0)
    embarked_q = st.number_input("Embarked Q (0 or 1), if c (0)", min_value=0, max_value=1, value=0)
    embarked_s = st.number_input("Embarked S (0 or 1), if c (0)", min_value=0, max_value=1, value=1)

    # Combine inputs into a single array
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, cabin, embarked_q, embarked_s]])

    # Standardize the input using the loaded scaler
    standardized_input = scaler.transform(input_data)

    # Make prediction
    if st.button("Predict Survival"):
        prediction = model.predict(standardized_input)
        survival_status = "Survived" if prediction[0] == 1 else "Did Not Survive"
        st.success(f"The predicted outcome is: {survival_status}")

if __name__ == '__main__':
    main()