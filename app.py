'''
For running this code compile as "streamlit run app.py"
'''
# importing librarys

import streamlit as st
import numpy as np
import pandas as pd

# '''Title of the aplication'''

st.title("Hello Streamlit")

# Display a Simple Text
st.write("This is a simple text")

# '''Create a simple DataFrame'''
df=pd.DataFrame({
    'first column':[1,2,3,4,5],
    'Second column':[10,20,30,40,50]
})

# '''Displaying DataFrame'''
st.write("Here is the dataframe")
st.write(df)

# '''Create an line chart'''
chart_data=pd.DataFrame(
    np.random.randn(20,3),columns=['a','b','c']
)
st.line_chart(chart_data)

