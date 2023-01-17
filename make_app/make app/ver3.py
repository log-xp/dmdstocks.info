import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import pydmd
import streamlit as st


df = pd.DataFrame()
matx = pd.read_csv("DATA/nse50.csv",header=None) #VARIABLE 1
matx = pd.DataFrame(matx.transpose())
matx = matx.fillna(0)


sym = pd.read_csv("DATA/nse50_sym.csv",header=None) #VARIABLE 2

j=10
df = []
i=25 # VARIABLE 3
  
mat = matx.iloc[:, -(i+j):-j]
day2 = matx.iloc[:, -(i+j)+1]
mat=mat.to_numpy()

day1 = matx.iloc[:, -j]
day2 = matx.iloc[:, -j+1]
day3 = matx.iloc[:, -j+2]
day4 = matx.iloc[:, -j+3]
day5 = matx.iloc[:, -j+4]
day6 = matx.iloc[:, -j+5]
day7 = matx.iloc[:, -j+6]
day8 = matx.iloc[:, -j+7]
day9 = matx.iloc[:, -j+8]

dmd = pydmd.DMD(svd_rank=mat.shape[1])
dmd.fit(mat)
dmd.dmd_time['tend'] = dmd.dmd_time['tend'] + 9

pred = dmd.reconstructed_data
pday1 = np.real(pred[:,-9])
pday2 = np.real(pred[:,-8])
pday3 = np.real(pred[:,-7])
pday4 = np.real(pred[:,-6])
pday5 = np.real(pred[:,-5])
pday6 = np.real(pred[:,-4])
pday7 = np.real(pred[:,-3])
pday8 = np.real(pred[:,-2])
pday9 = np.real(pred[:,-1])

print("day1 : ",mean_absolute_error(day1,pday1))
print("day2 : ",mean_absolute_error(day2,pday2))
print("day3 : ",mean_absolute_error(day3,pday3))
print("day4 : ",mean_absolute_error(day4,pday4))
print("day5 : ",mean_absolute_error(day5,pday5))
print("day6 : ",mean_absolute_error(day6,pday6))
print("day7 : ",mean_absolute_error(day7,pday7))
print("day8 : ",mean_absolute_error(day8,pday8))
print("day9 : ",mean_absolute_error(day9,pday9))
print("-------------------------------------")
print("day1 : ",day1[1]," - pred :",pday1[1])
print("day2 : ",day2[1]," - pred :",pday2[1])
print("day3 : ",day3[1]," - pred :",pday3[1])
print("day4 : ",day4[1]," - pred :",pday4[1])
print("day5 : ",day5[1]," - pred :",pday5[1])