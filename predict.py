import streamlit as st
import pandas as pd
import joblib
from io import StringIO
from PIL import Image

# Load the saved model
model = joblib.load('hash_gbc.pkl')

# Set up the sidebar
# Logo image
image = Image.open('files/tmcg.png')
st.sidebar.image(image, width=300)
st.sidebar.title('Upload CSV file')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
st.sidebar.markdown("""
[Example input file](files/sample.csv)
""")


# Main content
st.write("""
### Machine Learning algorithms for predictive modelling of the ability to pay for Pre-Exposure (PrEP) services, a pilot study in Kampala, Uganda.
         """)

# Project Description and Column Key
st.write("""
#### Project Description
HIV prevention services in Uganda including PrEP have remained largely donor-funded. With the changing trends in healthcare financing, there is a need for more sustainable health care financing models. Understanding the nuances in PrEP service delivery especially targeting unserved high risk sub-populations through innovative approaches like cost-sharing, privately purchased etc is worth a venture to explore.

We set to use machine learning and AI models on both public and private datasets to build models that would predict ability to pay for HIV prevention services particularly PrEP. 
""")


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # Display the uploaded data
    st.write('Uploaded Data:')
    st.write(data)

    # Make predictions using the loaded model and add a 'Prediction' column
    predictions = model.predict(data)  # Assuming 'model' is a trained model object
    data['Prediction'] = predictions

    # Display the predictions
    st.write('Predictions:')
    st.write(data)

    # Export results to a CSV file
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=StringIO(csv).read(),
        file_name='predictions.csv',
        mime='text/csv'
    )

