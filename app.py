import pickle
import streamlit as st
import numpy as np


data = pickle.load(open('final.pkl','rb'))
ada_clf = pickle.load(open('train.pkl','rb'))

st.title("Medical Sample Collection Process Streamline")

#Test_Name
test_name = st.selectbox('test_name', data["test_name"].unique())
if test_name == "Acute kidney profile":
    test_name = 0
elif test_name == "HbA1c":
    test_name = 5
elif test_name == "Vitamin D-25Hydroxy":
    test_name = 9
elif test_name == "TSH":
    test_name = 8
elif test_name == "Lipid Profile":
    test_name = 6
elif test_name == "Complete Urinalysis":
    test_name = 2
elif test_name == "RTPCR":
    test_name = 7
elif test_name == "H1N1":
    test_name = 4
elif test_name == "Fasting blood sugar":
    test_name = 3
else:
    test_name = 1

#Sample
sample = st.radio('sample', data["sample"].unique())
if sample == "Blood":
    sample = 0
elif sample == "Urin":
    sample = 2
else:
    sample = 1

#Gender

#Way_Of_Storage_Of_Sample
sample_storage = st.radio('sample_storage', data["sample_storage"].unique())
if sample_storage == "Advanced":
    sample_storage = 0
else:
    sample_storage = 1

#Test_Booking_Time_HH_MM
test_booking_time_HH_MM = st.number_input('test_booking_time_HH_MM')

#Scheduled_Sample_Collection_Time_HH_MM
Scheduled_Sample_Collection_Time_HH_MM = st.number_input('Scheduled_Sample_Collection_Time_HH_MM')

#Cut_off_Schedule
Cut_off_Schedule = st.radio('Cut_off_Schedule', data['Cut_off_Schedule'].unique())
if Cut_off_Schedule == "Sample by 5pm":
    Cut_off_Schedule = 1
else:
    Cut_off_Schedule = 0


#Cut_off_time_HH_MM
Cut_off_time_HH_MM = st.number_input('Cut_off_time_HH_MM')

#Agent_ID
Agent_ID = st.number_input('Agent_ID')

#Traffic_Conditions
traffic_conditions = st.radio('traffic_conditions', data['traffic_conditions'].unique())
if traffic_conditions == "Low Traffic":
    traffic_conditions = 1
elif traffic_conditions == "Medium Traffic":
    traffic_conditions = 2
else:
    traffic_conditions = 0


#Agent_Location_KM
Agent_Location_KM = st.number_input('Agent_Location_KM')

#Time_Taken_To_Reach_Patient_MM
Time_Taken_To_Reach_Patient_MM = st.number_input('Time_Taken_To_Reach_Patient_MM')

#Time_For_Sample_Collection_MM
time_for_sample_collection = st.number_input('time_for_sample_collection')

#Lab_Location_KM
lab_location = st.number_input('lab_location')


#Time_Taken_To_Reach_Lab_MM
time_taken_to_reach_lab = st.number_input('time_taken_to_reach_lab')

if st.button('Predict Result'):

    query = np.array([test_name,sample,sample_storage,test_booking_time_HH_MM,Scheduled_Sample_Collection_Time_HH_MM,Cut_off_Schedule,Cut_off_time_HH_MM,Agent_ID,traffic_conditions,Agent_Location_KM,Time_Taken_To_Reach_Patient_MM,time_for_sample_collection,lab_location,time_taken_to_reach_lab])
    query = query.reshape(1,14)

    result = ada_clf.predict(query)

    if result == 'Y':
        st.header("Sample Reached On Time ?\n YES")
    else:
        st.header("Sample Reached On Time ?\n NO")