import streamlit as st
import pandas as pd
import numpy as np
import pydmd
from sklearn.metrics import mean_absolute_error
import altair as alt

st.title("Stock Price Prediction")

# Load data
df = pd.DataFrame()
matx = pd.read_csv("DATA/nse50.csv",header=None) #VARIABLE 1
matx = pd.DataFrame(matx.transpose())
matx = matx.fillna(0)

# Parameters
i = st.slider("Number of days to predict", min_value=1, max_value=100, value=25)
j = st.slider("Number of days to train", min_value=1, max_value=100, value=10)

mat = matx.iloc[:, -(i+j):-j]

days = ["day1", "day2", "day3", "day4", "day5", "day6", "day7", "day8", "day9"]

for i, day in enumerate(days):
    exec(f"{day} = matx.iloc[:, -j+{i}]")


# DMD
dmd = pydmd.DMD(svd_rank=mat.shape[1])
dmd.fit(mat)
dmd.dmd_time['tend'] = dmd.dmd_time['tend'] + 9
pred = dmd.reconstructed_data

# Result
pdays = {}
for i, day in enumerate(days):
    pdays[day] = np.real(pred[:,-(i+9)])

st.title("Predicted stock prices")
for day in days:
    st.line_chart(pdays[day])

data = [pdays[day] for day in days]
st.line_chart(data)
