import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import pydmd

def readSymbol(stock_exchange):
    symbol = pd.read_csv(f"DATA/{stock_exchange}.csv")
    symbol = symbol.iloc[:,0].tolist()
    if stock_exchange == "nse100_SYM":
        matx = pd.read_csv("DATA/NSE100.csv",header=None)
    if stock_exchange == "nse50_sym":
        matx = pd.read_csv("DATA/nse50.csv",header=None)

    matx = pd.DataFrame(matx.transpose())
    matx = matx.fillna(0)

    return symbol, matx


def dmd_pred_plot(b,sym,dmd,diff,d_ind,top):    
    dmode = dmd.modes[:,d_ind].real
    # Find the indices of the top 5 elements
    top_indices = np.argsort(dmode.real)[-top:]
    print("Recommended stocks are : ")
    for i in top_indices:
        print(sym[i],"[",i,"]")
    # Mark the indices on the plot
    plt.plot(diff)
    plt.scatter(top_indices,diff[top_indices], color='red')
    plt.title(" PERFORMACE COMPARISON - RECOMENDED STOCK")
    plt.show()

def dmd_analysis(matx,result_comparison,j,top=3,i=12) :
    mat = matx.iloc[:, -(i+j):-j]
    mat = mat.to_numpy()

    # dmd 
    dmd = pydmd.DMD(svd_rank=mat.shape[1])
    dmd.fit(mat)
    dmd.dmd_time['tend'] = dmd.dmd_time['tend'] + 9

    # creating the matrix mat 
    # i days are trained j th day from last 

    #setting the test set 
    day0 = matx.iloc[:, -j-1]
    day1 = matx.iloc[:, -j]
    day2 = matx.iloc[:, -j+1]
    day3 = matx.iloc[:, -j+2]
    day4 = matx.iloc[:, -j+3]
    day5 = matx.iloc[:, -j+4]
    day6 = matx.iloc[:, -j+5]
    day7 = matx.iloc[:, -j+6]
    day8 = matx.iloc[:, -j+7]
    day9 = matx.iloc[:, -j+8]


    mat = matx.iloc[:, -(i+j):-j]
    mat=mat.to_numpy()
    day = np.array([day0,day1,day2,day3,day4,day5,day6,day7,day8,day9])
    day_r = day[result_comparison]
    d_ind = np.argmax(np.abs(dmd.amplitudes.real))

    #if any(np.abs(dmd.eigs.real)>1) and dmd.amplitudes.real[d_ind] > 0 :
    if np.abs(dmd.eigs.real[d_ind])>1 and dmd.eigs.imag[d_ind] == 0 and dmd.amplitudes.real[d_ind] > 0 :
        print("^^^^ GrowinG ^^^^")
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(dmd.amplitudes.real)
        dmd_pred_plot(dmd, day_r-day0, d_ind, top)
        plt.show()
 
    else:
        print("^^^^ Shrinking ^^^^")
        print("No growing eigen value found")
        plt.plot(dmd.amplitudes.real)
        plt.show()
        if dmd.amplitudes.real[d_ind] > 0 :
            print("Amplitude is positive")
        plt.title('Amplitudes')
        
        dmd_pred_plot(dmd, day_r-day0, d_ind, top)

def dmd_pred(matx,sym,j,i,stock_name):
    # creating the matrix mat 
    # i days are trained j th day from last 
    mat = matx.iloc[:, -(i+j):-j]
    mat = mat.to_numpy()
    # index where the stock is equal to the symbol
    stock_ind = sym[sym[0] == stock_name].index[0]
    #setting the test set 
    day0 = matx.iloc[:, -j-1]
    day1 = matx.iloc[:, -j]
    day2 = matx.iloc[:, -j+1]
    day3 = matx.iloc[:, -j+2]
    day4 = matx.iloc[:, -j+3]
    day5 = matx.iloc[:, -j+4]
    day6 = matx.iloc[:, -j+5]
    day7 = matx.iloc[:, -j+6]
    day8 = matx.iloc[:, -j+7]
    day9 = matx.iloc[:, -j+8]

    # dmd 
    dmd = pydmd.DMD(svd_rank=mat.shape[1])
    dmd.fit(mat)
    dmd.dmd_time['tend'] = dmd.dmd_time['tend'] + 9
    dmd.reconstructed_data.shape


    # prediction

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



    # error
    print("Day1 |","Prediction :",pday1[stock_ind],"| Error :" ,np.abs(day1[stock_ind]-pday1[stock_ind]),"|")
    print("Day2 |","Prediction :",pday2[stock_ind],"| Error :" ,np.abs(day2[stock_ind]-pday2[stock_ind]),"|")
    print("Day3 |","Prediction :",pday3[stock_ind],"| Error :" ,np.abs(day3[stock_ind]-pday3[stock_ind]),"|")
    print("Day4 |","Prediction :",pday4[stock_ind],"| Error :" ,np.abs(day4[stock_ind]-pday4[stock_ind]),"|")
    print("Day5 |","Prediction :",pday5[stock_ind],"| Error :" ,np.abs(day5[stock_ind]-pday5[stock_ind]),"|")
    print("Day6 |","Prediction :",pday6[stock_ind],"| Error :" ,np.abs(day6[stock_ind]-pday6[stock_ind]),"|")
    print("Day7 |","Prediction :",pday7[stock_ind],"| Error :" ,np.abs(day7[stock_ind]-pday7[stock_ind]),"|")
    print("Day8 |","Prediction :",pday8[stock_ind],"| Error :" ,np.abs(day8[stock_ind]-pday7[stock_ind]),"|")
    print("Day9 |","Prediction :",pday9[stock_ind],"| Error :" ,np.abs(day9[stock_ind]-pday7[stock_ind]),"|")



