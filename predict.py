import streamlit as st
import pandas as pd
import joblib
from io import StringIO
from PIL import Image
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns


# Load the saved model
#model = joblib.load('files/hash_final.pkl')
model = load('files/hash_model.joblib')

# Set up the sidebar
# Logo image
image = Image.open('files/tmcg.png')
st.sidebar.image(image, width=300)
st.sidebar.title('Upload CSV file')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
st.sidebar.markdown(""" [Example input file](./files/sample.csv)""")



# Display HASH logo image
logo_url = 'files/hash.png'
st.image(logo_url, width=200) 


# Main content
st.write("""
### A predictive model for one's ability to pay for Pre-Exposure Prophylactic (PrEP) services in Uganda.
         """)

# Project Description and Column Key
st.write("""
#### Project Description
HIV prevention services in Uganda including PrEP have remained largely donor-funded. With the changing trends in healthcare financing, there is a need for more sustainable health care financing models. Understanding the nuances in PrEP service delivery especially targeting unserved high risk sub-populations through innovative approaches like cost-sharing, privately purchased etc is worth a venture to explore.

This AI Model was trained on both public and private datasets to predict ability to pay for HIV prevention services particularly PrEP. 
""")


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # Display the uploaded data
    st.write(' #### Uploaded Data Sample:')
    st.write(data.head())

    # Display the uploaded data shape
    st.write(' #### Number of rows and columns:')
    st.write(data.shape)

    st.write(' #### Data Summary:')
    st.write(data.describe())

    # Display column names and types
    st.write(' #### Column Names and Data Types:')
    # Capture the output of data.info()
    buffer = StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Plot the gender pie chart
    st.write(""" #### Gender Disribution:
             Key: 1 = Male, 2 = Female, 99 = Missing
             """)
    plt.figure(figsize=(15,10))
    data['gender'].value_counts().plot(kind='pie', autopct='%.2f%%')
    plt.title('Gender Distribution')
    plt.ylabel('')  # Hide the y-label for a cleaner look

    # Display the plot in Streamlit
    st.pyplot(plt)

     # Empoloyment Status
    st.write(""" #### Employment Status:
                    1 - PROFESSIONAL   
                2 - CLERICAL
                3 - SALES AND SERVICES
                4 - SKILLED MANUAL
                5 - UNSKILLED MANUAL
                6 - DOMESTIC SERVICE
                7 - AGRICULTURE
                96 - OTHER (SPECIFY)
             """)
    plt.figure(figsize=(15,10))
    data['occuptn'].value_counts().head(30).plot(kind='barh', figsize=(20,10))

    # Calculate the percentage for each bar
    total = data['occuptn'].value_counts().sum()
    for i, v in enumerate(data['occuptn'].value_counts().head(30)):
        percentage = f"{(v / total) * 100:.2f}%"
        plt.text(v, i, percentage, color='black', va='center')

    # Display the plot in Streamlit
    st.pyplot(plt)
    
    # Plot the gender pie chart
    st.write(""" #### Residence Disribution:
             Key: 1 = Urban, 2 = Rural
             """)
    # Plot the count plot
    plt.figure(figsize=(15, 10))
    sns.countplot(data=data, x='urban')
    plt.xlabel('Residence')
    plt.ylabel('Count of Respondents')

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Make predictions using the loaded model and add a 'Prediction' column
    predictions = model.predict(data)  # Assuming 'model' is a trained model object
    data['Prediction'] = predictions

    # Display the predictions
    st.write(""" #### Predictions:
             Key: 1 = Able to Pay, 0 = Not able to Pay
             """)
    st.write(data)

    # Export results to a CSV file
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=StringIO(csv).read(),
        file_name='predictions.csv',
        mime='text/csv'
    )

