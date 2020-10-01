# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 02:20:31 2020

@author: Gokulraj
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:50:04 2020

@author: Gokulraj
"""


import numpy as np
import pickle
import pandas as pd







import streamlit as st 

from PIL import Image
model=pickle.load(open('spamfinder.pkl', 'rb'))
transformer=pickle.load(open('trans.pkl', 'rb'))


#@app.route('/')
def welcome():
    return "Let's Find the Message Spam or Not"

#@app.route('/predict',methods=["Get"])
def predict_message(message):
    message = [message]
    message = transformer.transform(message).toarray()
    prediction=model.predict(message)
    
    if prediction == 0:
        ans = "Ham Message"
        
    else:
        ans = "Spam Message"
    return ans



def main():
    st.title("Spam Finder")
    welcome()
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Message classifier</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    message = st.text_input("Message")
    result=""
    if st.button("Predict"):
        result=predict_message(message)
        
    
    st.success(result)
    if st.button("About"):
        st.text("Used Algorithm : Multinomial Naive Bayes")
        st.text("Accuracy : 98.21%")
        link = '[Code](https://github.com/gokulvm)'
        st.markdown(link, unsafe_allow_html=True)
       
if __name__=='__main__':
    main()
    
       
