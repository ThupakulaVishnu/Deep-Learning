# import streamlit as st

# st.title("Streamlit Text Input")

# name=st.text_input("Enter your name : ")  #Text box

# if name:
#     st.write(f"Hello , {name}")  #It prints the matter from text box

'''
==================================================================================================================================
==================================================================================================================================
'''

# import streamlit as st

# st.title("Streamlit Text Input")

# name=st.text_input("Enter your name : ")  

# age=st.slider("Select your age : ",0,100,25)   #It shows an horizantal bar line to move left and right for selection 
#                                                #Bar starts from 0 and ends at 100 and by default it is at "25"

# st.write(f"Your age is {age}")  #Based on the bar we have  that age is showing this age

# #For text box it shows opions
# options=["Python","Java","C","DSA","C++","JavaScript"]
# choice=st.selectbox("Choose your favorite language : ",options)
# st.write(f"Your selected {choice}.")


# if name:
#     st.write(f"Hello , {name}") 


import streamlit as st
import pandas as pd

st.title("Streamlit Text Input")

name=st.text_input("Enter your name : ")  

age=st.slider("Select your age : ",0,100,25)   
                                            
st.write(f"Your age is {age}")  #Based on the bar we have  that age is showing this age

#For text box it shows opions
options=["Python","Java","C","DSA","C++","JavaScript"]
choice=st.selectbox("Choose your favorite language : ",options)
st.write(f"Your selected {choice}.")


if name:
    st.write(f"Hello , {name}") 

data={
    "Name":["Vishnu","John","jake","Jill"],
    "Age ":[28,24,35,40],
    "City":["A","b","c","d"]
}
df=pd.DataFrame(data)
df.to_csv("Sample.csv")
st.write(df)

upload_file=st.file_uploader("Choose a CSV file",type="csv")  #It accept the csv file

if upload_file is not None: #If file is there 
    df=pd.read_csv(upload_file) #It read csv file
    st.write(df)               #Print that file data