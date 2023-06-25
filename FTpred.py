import streamlit as st

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import joblib

st.set_page_config(page_title="Fraud Transaction Predictor")
st.write("Enter transaction details below:")

oldbalanceOrg=st.number_input("Previous Balance [Originating Account]",value=15000)
newbalanceOrig=st.number_input("New Balance [Originating Account]",value=0)

oldbalanceDest=st.number_input("Previous Balance [Destination Account]",value=20000)
newbalanceDest=st.number_input("New Balance [Destination Account]",value=25000)

amount=st.number_input("Transaction Amount",value=25000)

Dest_pct_ch=round(100*(newbalanceDest-oldbalanceDest)/(oldbalanceDest+0.0000000001),2)
Orig_pct_ch=round(100*(newbalanceOrig-oldbalanceOrg)/(oldbalanceOrg+0.000000001),2)

isFlaggedFraud=0

s=st.selectbox("Transaction Type",["CASH IN","CASH OUT","DEBIT","PAYMENT","TRANSFER"])

if s=="CASH IN":
    x_t=[0,0,0,0]
elif s=="CASH OUT":
    x_t=[1,0,0,0]
elif s=="DEBIT":
    x_t=[0,1,0,0]
elif s=="PAYMENT":
    x_t=[0,0,1,0]
else:
    x_t=[0,0,0,1]
    
    
t=pd.DataFrame({"type_CASH_OUT":[x_t[0]],"type_DEBIT":[x_t[1]],"type_PAYMENT":[x_t[2]],"type_TRANSFER":[x_t[3]]})


d=pd.DataFrame({"amount":[amount],"oldbalanceOrg":[oldbalanceOrg],"newbalanceOrig":[newbalanceOrig],
                "oldbalanceDest":[oldbalanceDest],"newbalanceDest":[newbalanceDest],
               "isFlaggedFraud":[isFlaggedFraud],"Orig_pct_ch":[Orig_pct_ch],"Dest_pct_ch":[Dest_pct_ch]})

x=pd.concat([t,d],axis=1)

model=joblib.load('ftmodel')

p=model.predict_proba(x)

st.sidebar.metric("Fraud Probability",value=round(p[0,1],2))