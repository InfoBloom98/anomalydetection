import streamlit as st

# Simple test app
st.title("🧪 Simple Test App")
st.write("If you can see this, Streamlit is working!")

# Add a simple button
if st.button("Click me!"):
    st.write("Button clicked! Streamlit is working correctly.")

# Add some basic widgets
st.header("Basic Widgets Test")
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}!")

age = st.slider("Select your age:", 0, 100, 25)
st.write(f"You are {age} years old.")

# Add a simple chart
import numpy as np
import pandas as pd

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C'])

st.line_chart(chart_data)

st.success("✅ All tests passed! Streamlit is working properly.") 