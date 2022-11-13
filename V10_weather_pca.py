# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 00:27:09 2022

@author: jeremy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

###########################################DATASET PREPARATION###########################################
# Load dataset into numpy arrays
# Input high level dataset name here:
high_data_np_arr_raw = np.loadtxt("reading_start_at_16hrs_52min.csv")
# Input Mid level dataset name here:
mid_data_np_arr_raw = np.loadtxt("reading_start_at_21hrs_52min.csv")
# Input low level dataset name here:  
low_data_np_arr_raw = np.loadtxt(("reading_start_at_22hrs_40min.csv"))

# Preparing to append a new column for each dataset of interest:
arr_rows = np.size(high_data_np_arr_raw,0) 
arr_cols = np.size(high_data_np_arr_raw,1)
high_data_np_arr_appended = np.zeros((arr_rows,arr_cols+1))

arr_rows = np.size(mid_data_np_arr_raw,0) # Replaces previous variable 
arr_cols = np.size(mid_data_np_arr_raw,1) # Replaces previous variable 
mid_data_np_arr_appended = np.zeros((arr_rows,arr_cols+1))

arr_rows = np.size(low_data_np_arr_raw,0) # Replaces previous variable 
arr_cols = np.size(low_data_np_arr_raw,1) # Replaces previous variable 
low_data_np_arr_appended = np.zeros((arr_rows,arr_cols+1))

# Append new columns into numpy array:
high_data_np_arr_appended[:,:-1] = high_data_np_arr_raw
mid_data_np_arr_appended[:,:-1] = mid_data_np_arr_raw
low_data_np_arr_appended[:,:-1] = low_data_np_arr_raw

# Set a value to differentiate each dataset in the newly appended column:
for i in range(len(high_data_np_arr_appended)):
    high_data_np_arr_appended[i,6] = 3 # Arbitrary number to differentiate datasets
for i in range(len(mid_data_np_arr_appended)):
        mid_data_np_arr_appended[i,6] = 2 # Arbitrary number to differentiate datasets
for i in range(len(low_data_np_arr_appended)):
    low_data_np_arr_appended[i,6] = 1 # Arbitrary number to differentiate datasets
    
# Combine all datasets into a single numpy array:
final_data_np_arr = np.concatenate((high_data_np_arr_appended,mid_data_np_arr_appended,low_data_np_arr_appended),axis=0)

# Convert to Pandas dataframe:
# Column headers are based on RPi Output 
# Hours = number of hours from 0000hrs of which the datapoint was recorded at
# matplotlib_time = date of datarecording # This recording will not be used in this PCA
# PM2.5 and PM10 = PM readings from SDS011 sensor
# Temperature and Humidity = Measurements from DHT22 sensor
df = pd.DataFrame(final_data_np_arr, columns = ['Hours','matplotlib_time','PM2.5','PM10','temperature','humidity','location'])

# Convert location column to string by replacement 
lowstr = "Level 16"
midstr = "Level 8"
highstr = "Level 19"
targets = [lowstr, midstr,highstr] 

# Replace the arbitrary numbers with strings:
df['location'] = df['location'].replace(1,lowstr)
df['location'] = df['location'].replace(2,midstr)
df['location'] = df['location'].replace(3,highstr)

# 'Hours' Column in dataframe does not differentiate midnight(0000hrs) with 2300hrs.
# Magnitude is on extreme ends, however the change in value is only by 1 hour.
# Solution: Categorise all 'Hours' readings into two groups:
# Restinghours = students have returned to hostel and are occupying the apartment
# Workinghours = students have left the hostel and are in class 
restinghours = [23,0,1,2,3,4,5,6,7] #9pm to 7am
workinghours = [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22] # 8am to 10pm

# Replace the Hours reading with a binary value such that the loading plot 
# would show the correlation between principal components and significance of the time of day
# rather than just the number of hours from 0000hrs
df.loc[(df['Hours'].isin(restinghours)),'Hours'] = 0
df.loc[(df['Hours'].isin(workinghours)),'Hours'] = 1

###########################################END OF DATASET PREPARATION###########################################
###########################################CUSTOMISATION OPTIONS################################################

# Easy Access to customisation options: 
PCAcomponents = 5
PCA_analysis = 4 # Set a value from 2 to PCAcomponents

# Scores scatter plot customination options:
colors = ['r', 'g','b'] # Change datapoint colour
alphacustom = 0.5 #Change datapoint transparency 
markercustom = 'x' # Change datapoint marker 
datapointsize = 40 # Change datapoint size

# Loadings line plot customisation options:
scale = 7 # Scale for loadings plot # Makes the line longer for data visualisation
loadingsalpha = 0.7 # Transparency for loadings plot
loadingslinewidth = 2 # Thickness of line
loadingscolor = 'k' # Colour of line

# Loading plot headers:
loadingtitles = ["Time Period","PM2.5","PM10","Temperature","Humidity"]
#######################################END OF CUSTOMISATION OPTIONS#############################################
#######################################PRINCIPAL COMPONENT ANALYSIS#############################################
# Standardize the data values
features = ['Hours','PM2.5','PM10','temperature','humidity'] #sans matplotlib_time and location label
# Separating out the features of interest :
x_unstd = df.loc[:, features].values 
# Separating out the location:
y = df.loc[:,['location']].values 
# Standardizing the features in a numpy array:
x = StandardScaler().fit_transform(x_unstd) # Each feature in x now has mean = 0 and var = 1

# Single Variable Decomposition:
pca = PCA(n_components=PCAcomponents)       # Creates the pca object, able to operate on a 2-D array
principalComponents = pca.fit_transform(x)  # Performs the actual PCA, returns scores as a numpy array
# Converts scores matrix into Pandas dataframe:
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2','PC3','PC4','PC5']) 
# Also get loadings for later plotting
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2','PC3','PC4','PC5'], index=features)

# Concatenate dataframe with the previously removed location lable column along axis = 1
finalDf = pd.concat([principalDf, df[['location']]], axis = 1) # attach array into the scores matrix
###################################END OF PRINCIPAL COMPONENT ANALYSIS#############################################
###################################PCA GRAPH PLOTTING##############################################################
# Scores plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component {}'.format(PCA_analysis), fontsize = 15)
ax.set_title('Scores plot, {} component PCA'.format(PCAcomponents), fontsize = 20)


# Replace color and colors with m and markercustom respectively if diffentiating between dataset levels by markers is desired.
for location, color in zip(targets,colors): 
    indicesToKeep = finalDf['location'] == location
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
               , finalDf.loc[indicesToKeep, 'PC{}'.format(PCA_analysis)]
               , c = color # Remove this line if diffentiating between dataset levels by colour is not desired.
               , s = datapointsize , alpha = alphacustom 
               , marker = markercustom) # Replace marker custom with m if diffentiating between dataset levels by markers is desired.
ax.legend(targets)
ax.grid()
#plt.savefig('Scoresplot_PC1vsPC{}.png'.format(PCA_analysis))

# Biplot of loadings and scores plots
# Scores plot section of biplot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component {}'.format(PCA_analysis), fontsize = 15)
ax.set_title('Biplot, {} component PCA'.format(PCAcomponents), fontsize = 20)
for location, color in zip(targets,colors): 
    indicesToKeep = finalDf['location'] == location
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
               , finalDf.loc[indicesToKeep, 'PC{}'.format(PCA_analysis)]
               , c = color # Remove this line if diffentiating between dataset levels by colour is not desired.
               , s = datapointsize , alpha = alphacustom 
               , marker = markercustom) # Replace marker custom with m if diffentiating between dataset levels by markers is desired.
ax.legend(targets)
ax.grid()
# Loadings plot section of biplot
loading_values = loadings.loc[:].values
finalxcoords = []
finalycoords = []
# Loadings plot based on selected PC with respect to PC1:
loading_values_selectedPC = np.zeros((5,2))
for i in range(PCAcomponents):
    loading_values_selectedPC[i:,0] = loading_values[i:,0] # This refers to loading coordinates for PC1
    loading_values_selectedPC[i:,1] = loading_values[i:,PCA_analysis-1] # This refers to loading coordinates for the desired PC
for i in range(np.size(features)):
    lx, ly = loading_values_selectedPC[i,:]*scale
    xx, yy = np.linspace(0,lx,101), np.linspace(0,ly,101)
    finalxcoords.append(xx[100])
    finalycoords.append(yy[100])
    plt.plot(xx, yy, linewidth = loadingslinewidth, color=loadingscolor,alpha = loadingsalpha)
###################################END OF PCA GRAPH PLOTTING#######################################################
###################################PCA DATA ANALYSIS###############################################################
# Check loading titles against scaled loadings matrix: 
scaledloadings = loadings*scale # loading titles coordinates will correspond to scaled loading values.
for i in range(PCAcomponents):
    plt.text(finalxcoords[i], finalycoords[i], loadingtitles[i], fontsize = 12)
#plt.savefig('pcaV8_PC1vsPC{}.png'.format(PCA_analysis))

# a is the a-th PC
# Generate eigenvalues, g_a, of each PC: 
# g_a = scores_matrix_transposed * scores_matrix
scoresT = np.transpose(principalDf) # Transpose of scores matrix
eigenvalues_arr = scoresT.dot(principalDf) # Actual generation of eigenvalues
eigenvalues = np.zeros((1,PCAcomponents)) # Prepare numpy array to store eigenvalues
eigenvalues = np.diag(eigenvalues_arr) # Store eigenvalues in a single numpy array
sum_eigen = np.sum(eigenvalues) # Sum of eigenvalues for calculation of residual sum of squares(RSS)

# Generate percentages of sum of squares of the entire dataset
# V_a =  100 x eigenvalue / sum of eigenvalues
v = np.zeros(PCAcomponents) # Prepare numpy array of V_a 
for i in range(len(eigenvalues)):
    v[i] = 100*eigenvalues[i]/sum_eigen #stores all percentages of sum of squares into a single numpy array
    
# Generate RSS 
# RSS_a = sum of eigenvalues - cumulative sum of eigenvalues up to a-th PC
# The last RSS value should be 0
RSS = np.zeros(PCAcomponents) # Prepare numpy array of RSS_a 
cumsum_eigen = np.cumsum(eigenvalues) # Cumulative sum of eigenvalues for calculation of RSS
for i in range(PCAcomponents):
    RSS[i] = sum_eigen - cumsum_eigen[i]

# Generate Root-Mean-Square-Error (RMSE)
# RMSE_a = sqrt(RSS_a / n_row_dataset x n_col_dataset)
RMSE = np.zeros(PCAcomponents) # Prepare numpy array of RMSE_a 
for i in range(len(eigenvalues)):
    RMSE[i] = np.sqrt(RSS[i]/(322*5)) 
# A PC is considered signficant while its RMSE is greater than the calibration error of the two sensors.

# For reporting the relative cumulative eigenvalue:
cum_percentage = 100*cumsum_eigen/sum_eigen
for i in range(PCAcomponents):
    print("{:4}{:8}{:8}{:10}{:12}{:7}".format("PC","g_a","V_a","Cum G_a","RSS_a","RMSE_a"))
    print("{:<4}{:5.2f}{:7.2f}{:>10.2f}{:9.2f}{:12.3f}".format(i+1,eigenvalues[i],v[i],cum_percentage[i],RSS[i],RMSE[i]))
