import streamlit as st
from stockmarketprediction import readSymbol
from stockmarketprediction import dmd_analysis
from stockmarketprediction import dmd_pred

# from stockmarketprediction import dmd_pred
# from stockmarketprediction import dmd_pred_plot

st.title("Stock Price Prediction")

stock_exchange = ["nse50_sym", "nse100_SYM"]    
input_1 = st.selectbox("Select a Stock Exchange", stock_exchange)
symbol = readSymbol(input_1)
input_2 = st.selectbox("Choose a Symbol", symbol[0])
# input_3 = st.date_input("Enter a date:")
col1, col2 = st.columns(2)
input_3 = col1.date_input("From:")
input_4 = col2.date_input("To:")
input_5 = st.number_input("Training Set (in days)",step=1,min_value=0)
pred_options = ["Next 5 days", "Next 10 days"]
input_6 = st.selectbox("Predict for",pred_options)

if st.button("Predict"):
   dmd_analysis(symbol[1],9,input_5,5)
   dmd_pred(symbol[1],symbol[0],25,12,input_2)
