import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.preprocessing import StandardScaler

# ============================================       /     STREAMLIT DASHBOARD      /       ================================================= #
# Comfiguring Streamlit GUI 
st.set_page_config(layout='wide')
# Title
st.header(':violet[Financial Risk Prediction]')
#=============================================================================================================================================#
col1,col2,col3,col4,col5=st.columns(5)

with col1:
    AMT_CREDIT_y=st.text_input('AMT_CREDIT_y')
    AMT_GOODS_PRICE_y=st.text_input('AMT_GOODS_PRICE_y')
    AMT_INCOME_TOTAL=st.text_input('AMT_INCOME_TOTAL')
    DAYS_FIRST_DRAWING=st.text_input('DAYS_FIRST_DRAWING')
    AMT_ANNUITY_y=st.text_input('AMT_ANNUITY_y')

with col2:
    DAYS_FIRST_DUE=st.text_input('DAYS_FIRST_DUE')
    AMT_DOWN_PAYMENT=st.text_input('AMT_DOWN_PAYMENT')
    DAYS_TERMINATION=st.text_input('DAYS_TERMINATION')
    DAYS_EMPLOYED=st.text_input('DAYS_EMPLOYED')
    DAYS_LAST_DUE=st.text_input('DAYS_LAST_DUE')
    
with col3:
    AMT_ANNUITY_x=st.text_input('AMT_ANNUITY_x')
    AMT_APPLICATION=st.text_input('AMT_APPLICATION')
    AMT_GOODS_PRICE_x=st.text_input('AMT_GOODS_PRICE_x')
    SK_ID_CURR=st.text_input('SK_ID_CURR')
    SELLERPLACE_AREA=st.text_input('SELLERPLACE_AREA')   

with col4:
    DAYS_LAST_DUE_1ST_VERSION=st.text_input('DAYS_LAST_DUE_1ST_VERSION')
    SK_ID_PREV=st.text_input('SK_ID_PREV')
    NAME_HOUSING_TYPE=st.text_input('NAME_HOUSING_TYPE: House / apartment:0,Rented apartment:1,With parent:2,Municipal apartment:3,Office apartment:4')
    NAME_EDUCATION_TYPE=st.text_input('NAME_EDUCATION_TYPE:Secondary / secondary special:0,Higher education:1,Incomplete higher:2,Lower secondary:3,Academic degree:4,Unknown:5')
    HOUR_APPR_PROCESS_START_y=st.text_input('HOUR_APPR_PROCESS_START_y')    

with col5:
    NAME_TYPE_SUITE_y=st.text_input('NAME_TYPE_SUITE_y:Unaccompanied:0,Family:1,Spouse, partner:2,Children:3,Other_A:4,Unknown:5,Other_B:6,Group of people:7})')
    AMT_CREDIT_x=st.text_input('AMT_CREDIT_x')
    NAME_INCOME_TYPE=st.text_input('NAME_INCOME_TYPE: Working:0,State servant:1,Commercial associate:2,Pensioner:3,Unemployed:4,Student:5,Unknown:6')
    NAME_CONTRACT_TYPE_y=st.text_input('NAME_CONTRACT_TYPE_y:Cash loans:0,Revolving loans:1,Unknown:2')
    NAME_FAMILY_STATUS=st.text_input('NAME_FAMILY_STATUS:Single / not married:0,Married:1,Civil marriag:2,Widow:3,Separated:4,Unknown:5')    
             
predict=st.button("Predict")    
if predict:
    df=pd.read_csv(r"C:\Users\SKAN\Desktop\Raajee\fin_risk_new_data1.csv")
    df.drop(["Unnamed: 0"],axis=1,inplace=True)    
    #x=df[["AMT_CREDIT_y","AMT_GOODS_PRICE_y","AMT_INCOME_TOTAL","DAYS_FIRST_DRAWING","AMT_ANNUITY_y","DAYS_FIRST_DUE","AMT_DOWN_PAYMENT","DAYS_TERMINATION","DAYS_EMPLOYED","DAYS_LAST_DUE","AMT_ANNUITY_x","AMT_APPLICATION","AMT_GOODS_PRICE_x","SK_ID_CURR","SELLERPLACE_AREA","DAYS_LAST_DUE_1ST_VERSION","SK_ID_PREV","NAME_HOUSING_TYPE","NAME_EDUCATION_TYPE","HOUR_APPR_PROCESS_START_y","NAME_TYPE_SUITE_y","AMT_CREDIT_x","NAME_INCOME_TYPE","NAME_CONTRACT_TYPE_y","NAME_FAMILY_STATUS"]]
    
    with open(r'C:\Users\SKAN\Desktop\Raajee\final_project\fin_risk_model2.pkl', 'rb') as file:
          svc=pickle.load( file)
    arr=np.array([[AMT_CREDIT_y,AMT_GOODS_PRICE_y,AMT_INCOME_TOTAL,DAYS_FIRST_DRAWING,AMT_ANNUITY_y,DAYS_FIRST_DUE,AMT_DOWN_PAYMENT,DAYS_TERMINATION,DAYS_EMPLOYED,DAYS_LAST_DUE,AMT_ANNUITY_x,AMT_APPLICATION,AMT_GOODS_PRICE_x,SK_ID_CURR,SELLERPLACE_AREA,DAYS_LAST_DUE_1ST_VERSION,SK_ID_PREV,NAME_HOUSING_TYPE,NAME_EDUCATION_TYPE,HOUR_APPR_PROCESS_START_y,NAME_TYPE_SUITE_y,AMT_CREDIT_x,NAME_INCOME_TYPE,NAME_CONTRACT_TYPE_y,NAME_FAMILY_STATUS]])  
    
    sc=StandardScaler()
    sample=sc.fit_transform(arr)
   
    result=svc.predict(sample)[0]

    if result==1:
        st.markdown(
            f"<h1 style='color:#ff6666; font-size: 24px;'>The Result is 1</h1>",
            unsafe_allow_html=True,)
        st.markdown(
            f"<h1 style='color:#ff6666; font-size: 24px;'>This Customer belongs to Financial_Default_Risk</h1>",
            unsafe_allow_html=True,)
    else:
        st.markdown(
            f"<h1 style='color:#ff6666; font-size: 24px;'>The Result is 0</h1>",
            unsafe_allow_html=True,)
        st.markdown(
            f"<h1 style='color:#ff6666; font-size: 24px;'>This Customer does not belongs to Financial_Default_Risk</h1>",
        unsafe_allow_html=True,)  

