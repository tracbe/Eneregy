import streamlit as st 
import joblib
# 1 - load save files
# 2 - input requires
# 3 - convert  text to number  ****
# 4 - scaler
# 5 - predection
# 6 - convert number in y  to text ****
model = joblib.load(r"C:/Users/Pc SToRe/Desktop/zayad/energy/Energy_save.pkl")
le_2 = joblib.load(r"C:/Users/Pc SToRe/Desktop/zayad/energy/le_2.pkl")
le_4 = joblib.load(r"C:/Users/Pc SToRe/Desktop/zayad/energy/le_4.pkl")
le_5 = joblib.load(r"C:/Users/Pc SToRe/Desktop/zayad/energy/le_5.pkl")
le_6 = joblib.load(r"C:/Users/Pc SToRe/Desktop/zayad/energy/le_6.pkl")
sc = joblib.load(r"C:/Users/Pc SToRe/Desktop/zayad/energy/sc.pkl")

st.title("Enregy Data")

Start_Hour = st.number_input("Start Hour:" , min_value=0, max_value= 24)
End_Hour = st.number_input("End Hour:" , min_value=0, max_value= 24)
Source = st.selectbox("Source:" ,le_2.classes_ )
Day_of_Year = st.number_input("Day_of_Year:" , min_value=0, max_value= 365)
Day_Name = st.selectbox("Day_Name:" ,le_4.classes_ )
Month_Name = st.selectbox("Month_Name:" ,le_5.classes_ )
Season = st.selectbox("Season:" ,le_6.classes_ )

input_data = [[Start_Hour,	End_Hour,	Source,	Day_of_Year,	Day_Name,	Month_Name,	Season]]

Source = le_2.transform([Source])[0]
Day_Name  = le_4.transform([Day_Name])[0]
Month_Name = le_5.transform([Month_Name])[0]
Season = le_6.transform([Season])[0]

input_data = [[Start_Hour,	End_Hour,	Source,	Day_of_Year,	Day_Name,	Month_Name,	Season]]

input_scale = sc.transform(input_data)

predection = model.predict(input_scale)
st.success(predection)

