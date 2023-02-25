# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 23:55:26 2023

@author: Lenovo
"""

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler, OneHotEncoder, LabelEncoder


# Loading the saved model
#car
car_model= pickle.load(open('D:/Price prediction model/car price prediction/carpriceprediction.pkl','rb'))
car= pd.read_csv('D:/Price prediction model/car price prediction/car_clean.csv')
#laptop
laptop_model= pickle.load(open('D:/Price prediction model/laptop_prediction.pkl','rb'))
laptop = pickle.load(open('D:/Price prediction model/df.pkl','rb'))
#mobile
mobile_model= pickle.load(open('D:/Price prediction model/mobile_model.pkl','rb'))
mobile= pd.read_csv('D:/Price prediction model/mobile_clean.csv')

def main():
    st.sidebar.title('Commodity Price Prediction Model')
    a= st.sidebar.radio('Select Category',('Select','Car price prediction','Laptop price prediction','Mobile price prediction'))
    if a == 'Select':
        st.title('Commodity Price Prediction')
        st.subheader('Welcome to commodity price prediction web application')
        st.caption('Please select your preference in sidebar')
        col1, col2, col3 = st.columns(3)
    
        with col1:
           st.header("Car")
           st.image("https://www.team-bhp.com/forum/attachments/modifications-accessories/2026359d1594118312-pics-tastefully-modified-cars-india-nano-front-3.jpg")
        
        with col2:
           st.header("Laptop")
           st.image("laptop5.jpeg")
        
        with col3:
           st.header("Mobile")
           st.image("https://www.sledge.co.ke/uploads/images/202302/image_750x_63dbb1237ed98.jpg")
       
    
    if a == 'Car price prediction':
        st.title('Car Price Prediction')
        image = Image.open('car.jpeg')
        st.image(image)
        
        st.warning("Kindly fill all details correctly",icon="⚠️")
        cmodel= st.selectbox('Select Car Model', options= sorted(car.name.unique()))
        cyear= st.slider('Select Year',1998,2023)
        kms= st.slider('Select KMS Driven',0,250000)
        fuel= st.selectbox('Select Fuel Type', options= sorted(car.fuel_type.unique()))
        transmission= st.selectbox('Select Transmission Type', options= sorted(car.transmission.unique()))
        owner = st.selectbox('Select Owner Number', options= car.owner.unique())
        
        if st.button('Predict price'):
            df= pd.DataFrame([[cmodel,cyear,kms,fuel,transmission,owner]],columns=['name','year','kms_driven','fuel_type','transmission','owner'])
            prediction = car_model.predict(df)
            prediction= int(prediction)
            st.success("Car price Will Be: " + str(prediction))
            
    if a== 'Laptop price prediction':
        st.title('Laptop Price Prediction')
        image = Image.open('laptop.jpeg')
        st.image(image)
        
        st.warning("Kindly fill all details correctly",icon="⚠️")
        
        # brand
        company = st.selectbox('Brand',laptop['Company'].unique())
        
        # type of laptop
        type = st.selectbox('Type',laptop['TypeName'].unique())
        
        # Ram
        ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])
        
        # weight
        weight = st.number_input('Weight of the Laptop')
        
        # Touchscreen
        touchscreen = st.selectbox('Touchscreen',['No','Yes'])
        
        # IPS
        ips = st.selectbox('IPS',['No','Yes'])
        
        # screen size
        screen_size = st.number_input('Screen Size')
        
        # resolution
        resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
        
        #cpu
        cpu = st.selectbox('CPU',laptop['Cpu brand'].unique())
        
        hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
        
        ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
        
        gpu = st.selectbox('GPU',laptop['Gpu brand'].unique())
        
        os = st.selectbox('OS',laptop['os'].unique())
        
        if st.button('Predict Price'):
            # query
            ppi = None
            if touchscreen == 'Yes':
                touchscreen = 1
            else:
                touchscreen = 0
        
            if ips == 'Yes':
                ips = 1
            else:
                ips = 0
        
            X_res = int(resolution.split('x')[0])
            Y_res = int(resolution.split('x')[1])
            ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
            query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
        
            query = query.reshape(1,12)
            st.success("The predicted price of this configuration is: " + str(int(np.exp(laptop_model.predict(query)[0]))))  
            
    if a== 'Mobile price prediction':
        st.title('Mobile Price Prediction')
        image = Image.open('mobile1.jpeg')
        st.image(image)
        
        st.warning("Kindly fill all details correctly",icon="⚠️")
        #brand
        Brand= st.selectbox('Select Mobile Brand',options= sorted(mobile.Brand.unique()))
        #memory
        memory= st.selectbox('Select Ram', options= sorted(mobile.Memory.unique()))
        #storage
        storage= st.selectbox('Select Storage', options= sorted(mobile.Storage.unique()))
        #ratings
        rating= st.slider('Select Ratings of Mobile',0.0,5.0,step= 0.1)
        #colour
        colour= st.selectbox('Select Color', options= sorted(mobile.colour.unique()))
        
        if st.button('Predict Price'):
            df2= pd.DataFrame([[Brand,memory,storage,rating,colour]],columns=['Brand', 'Memory', 'Storage', 'Rating', 'colour'])
            prediction = mobile_model.predict(df2)
            st.success("Mobile price Will Be: " + str(int(np.exp(prediction))))
            
if __name__=='__main__':
    main()       