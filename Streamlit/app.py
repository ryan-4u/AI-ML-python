import streamlit as st
import pandas as pd
# creating titles and heading
st.title("This is my App")
st.header("this is header")
st.subheader("this is subheading")
st.write("this is paragraph")

#taking input
a = st.text_input("Write your name - ")
st.write(a)

# slider
b=st.slider("select one number",1,100)
st.write(b)

#button
if st.button("Click me"):
    st.write("clicked...")

# checknox and buton
st.checkbox("PYTHON")
st.checkbox("C++")

st.radio('Choose Your Favorite',['python','c','java','c++'])

#inserting image
st.image("sample.png", caption="a beautiful background image")

# adding data
data= {
    "name": ["john", "peter", "jack","anuj"],
    "age": [20, 25, 30 ,32]
}
df = pd.DataFrame(data)
st.write(df)
# making its table
st.table(df)