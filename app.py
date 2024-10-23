import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from PIL import Image

# Set up the sidebar
# Logo image
image = Image.open('files/tmcg.png')
st.sidebar.image(image, width=300)

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

# Collects user input features into dataframe
def user_input_features():
    age                = st.sidebar.slider('Age (Years)', 18, 80, 36)
    gender             = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    education_level    = st.sidebar.selectbox('Education Level', ('No formal education','Some Primary','Completed Primary','Some Secondary','Completed Secondary', 'Dont know'))
    marital_status     = st.sidebar.selectbox('Marital Status', ('Married/Cohabiting', 'Never married','Separated','Widowed','Divorced'))
    religion           = st.sidebar.selectbox('Religion', ('Catholic','Pentecostal','Anglican/Protestant','Other','SDA','Moslems','Orthodox','Other Christians','Traditional','Bahai','Hindu'))
    urban              = st.sidebar.selectbox('Residence', ('Urban', 'Rural'))
    region             = st.sidebar.selectbox('Region', ('Central 2','Mid North','East Central','North East','Mid East','Kampala','Central','West Nile','Mid-West','South West'))
    occuptn            = st.sidebar.selectbox('Occupation', ('Professional','Other','Unskilled Labour','Refused','Agriculture','Domestic','Sales and service','Clerical','Do not know','Skilled manual'))
    wealth_quintile    = st.sidebar.selectbox('Wealth Quintile', ('Highest', 'Middle', 'Lowest', 'Second', 'Fourth'))
    financial_decision = st.sidebar.selectbox('Who makes financial decisions?', ('I do','Spouse/Husband','We both do','Someone else','Refused','Do not know'))
    service_location   = st.sidebar.selectbox('Last Health Visit', ('Govt Hospital','Govt Health Centre','Outreach Service','Field worker/VHT','Other Public Sector','Private Hospital/Clinic','Pharmacy/Drug shop','Private Doctor','Outreach Service', 'Other private medical sector','Shop','Traditional practitioner', 'Other'))
    known_hiv_status   = st.sidebar.selectbox('HIV Status', ('Results not received/refused','Stated HIV negative','Never tested','Stated HIV positive'))
    sexually_active    = st.sidebar.selectbox('Sexually Active', ('Ever had sexual intercourse', 'Never had sexual intercourse'))


    
    data = {
            'gender'             : gender,
            'age'                : age,
            'education_level'    : education_level,
            'marital_status'     : marital_status,
            'religion'           : religion,
            'urban'          : urban,
            'region'             : region,
            'occuptn'         : occuptn,
            'wealth_quintile'    : wealth_quintile,
            'financial_decision'   : financial_decision,
            'service_location' : service_location,
            'known_hiv_status'        : known_hiv_status,
            'sexually_active'   : sexually_active}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Encode categorical variables using map()
dfn = input_df.copy()  # Keep the original input for display

# Map binary categorical values to 0 and 1
encoding_dict = {
    'gender': {'Male': 1, 'Female': 2},
    'education_level'  : {'No formal education':1,'Some Primary':2,'Completed Primary':3,'Some Secondary':4,'Completed Secondary':5, 'Dont know':99},
    'marital_status'   : {'Married/Cohabiting':1, 'Never married':5,'Separated':3,'Widowed':4,'Divorced':2},
    'religion'         : {'Catholic':1,'Pentecostal':5,'Anglican/Protestant':2,'Other':96,'SDA':3,'Moslems':7,'Orthodox':4,'Other Christians':6,'Traditional':9,'Bahai':8,'Hindu':10},
    'urban'            : {'Urban':1, 'Rural':2},
    'region'           : {'Central 2':2,'Mid North':8,'East Central':4,'North East':5,'Mid East':5,'Kampala':3,'Central':1,'West Nile':14,'Mid-West':7,'South West':9},
    'occuptn'          : {'Professional':1,'Other':96,'Unskilled Labour':5,'Refused':9,'Agriculture':7,'Domestic':6,'Sales and service':3,'Clerical':2,'Do not know':8,'Skilled manual':4},
    'wealth_quintile'                  : {'Highest':5, 'Middle':3, 'Lowest':1, 'Second':2, 'Fourth':4},
    'financial_decision'               : {'I do':1,'Spouse/Husband':2,'We both do':3,'Someone else':7,'Refused':4,'Do not know':9},
    'service_location'                 : {'Govt Hospital':1,'Govt Health Centre':2,'Outreach Service':3,'Field worker/VHT':4,'Other Public Sector':5,'Private Hospital/Clinic':6,'Pharmacy/Drug shop':7,'Private Doctor':8,'Outreach Service':9, 'Other private medical sector':11,'Shop':12,'Traditional practitioner':13, 'Other':96},
    'known_hiv_status'                 : {'Results not received/refused':8,'Stated HIV negative':2,'Never tested':99,'Stated HIV positive':1},
    'sexually_active'                  : {'Ever had sexual intercourse':1, 'Never had sexual intercourse':2}
}

for col, mapping in encoding_dict.items():
    input_df[col] = input_df[col].map(mapping)

# Displays the user input features before encoding
st.subheader('User Input features')
st.write(dfn)


# Load the saved classification model
load_clf = load('files/hash_model.joblib')

# Apply model to make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

# Instead of using prediction probability, calculate the prediction score
# The prediction score is the highest probability of the two possible outcomes
prediction_score = np.max(prediction_proba)

# Prediction results
st.subheader('Prediction')
prediction_text = np.array(['Not Able to Pay', 'Able to Pay'])
st.write(prediction_text[prediction])

st.subheader('Prediction Confidence Score')
st.write(f"The model's confidence in this prediction is: **{prediction_score * 100:.2f}%**")

# Dynamic narrative for non-technical users
st.subheader('Explanation of Results')

if prediction[0] == 1:
    st.markdown(f"""
    **Ability to Pay for PrEP Services:** Based on the input data, the model predicts that the individual is **able to pay** for PrEP services. 
    The model is **{prediction_score * 100:.2f}%** confident in this prediction, indicating a high likelihood that the person has the financial capacity to afford PrEP. 
    
    This prediction suggests that financial resources may not be a barrier to accessing PrEP services, but it's still important to consider other factors that may affect access.
    """)
else:
    st.markdown(f"""
    **Ability to Pay for PrEP Services:** The model predicts that the individual is **unlikely** to be able to pay for PrEP services. 
    The model is **{prediction_score * 100:.2f}%** confident in this prediction, suggesting a potential financial barrier to accessing PrEP.
    
    While the likelihood of paying for PrEP appears low, further support and interventions may be necessary to ensure access to this important preventive service.
    """)

