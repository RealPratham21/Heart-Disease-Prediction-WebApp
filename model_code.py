import streamlit as st
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from pyngrok import ngrok
import joblib

# D:/MyData/My Documents/My Downloads/Heart Disease Prediction/

def Diagnose(data):
    try:
        print("Input Data: ", data)
        data = np.array(data).reshape(1, -1)
        scaler = joblib.load('scaler.pkl')
        data = scaler.transform(data)
        
        for i in data[0]:
            print(i, end=' ')
        
        print('\n', data[0][0], data[0][1], data[0][1])
        
        loaded_model = tf.keras.models.load_model('my_model2.keras')

        pred = loaded_model.predict(data)
        print("Predicted (Raw): ", pred)
        pred_class = [1 if i[0] >= 0.5 else 0 for i in pred]
        return pred_class
    except Exception as e:
        return f"Error in prediction: {e}"

def main():
    # App Title and Credits
    st.title("Heart Disease Prediction Web App")
    st.markdown(
        """
        ### Developed by Prathamesh Bhamare
        **GitHub**: [RealPratham21](https://github.com/RealPratham21)  
        **LinkedIn**: [Prathamesh Bhamare](https://www.linkedin.com/in/prathamesh-bhamare-7480b52b2/)  
        **X**: [ftw_pratham27](https://x.com/ftw_pratham27)
        """
    )

    # Input Fields
    st.markdown("---")
    st.header("Enter the following details:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.text_input("Age")
        sex = st.selectbox("Sex", ["1 (Male)", "0 (Female)"])
        cp = st.selectbox("Chest Pain Type (CP)", ["0", "1", "2", "3"])
        trestbps = st.text_input("Resting Blood Pressure (trestbps)")
        chol = st.text_input("Cholesterol Level (chol)")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["1 (True)", "0 (False)"])

    with col2:
        restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", ["0", "1", "2"])
        thalach = st.text_input("Maximum Heart Rate Achieved (thalach)")
        exang = st.selectbox("Exercise Induced Angina (exang)", ["1 (Yes)", "0 (No)"])
        oldpeak = st.text_input("ST Depression Induced by Exercise (oldpeak)")
        slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", ["0", "1", "2"])
        ca = st.text_input("Number of Major Vessels Colored by Fluoroscopy (ca)")
        thal = st.selectbox("Thalassemia (thal)", ["1 (Normal)", "2 (Fixed Defect)", "3 (Reversible Defect)"])

    st.markdown("---")

    # Diagnosis Button
    if st.button("Diagnose"):
        try:
            pred = Diagnose([
                float(age), float(sex.split()[0]), float(cp), float(trestbps), float(chol), float(fbs.split()[0]),
                float(restecg), float(thalach), float(exang.split()[0]), float(oldpeak), float(slope), float(ca), float(thal.split()[0])
            ])

            if pred[0] == 0:
                st.success("You don't have Heart Disease")
            elif pred[0] == 1:
                st.error("You have Heart Disease", icon="⚠️")
        except Exception as e:
            st.error(f"Error in Prediction: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        #### Credits
        Developed with ❤️ by Prathamesh Bhamare.   
        **GitHub**: [RealPratham21](https://github.com/RealPratham21) | 
        **LinkedIn**: [Prathamesh Bhamare](https://www.linkedin.com/in/prathamesh-bhamare-7480b52b2/) | 
        **X**: [ftw_pratham27](https://x.com/ftw_pratham27)
        """
    )

if __name__ == "__main__":
    main()
