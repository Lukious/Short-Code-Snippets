"""
Data preprocessing

@author: lukious
"""
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import utils
"""

import pandas as pd

counter = 0
Resampled_HR = []
temp_saver = 0
conti_loader = 0
cubic = 0


def re_indexing(data_f):
    data_f = data_f.reset_index()
    data_f = data_f.drop(columns=['index'])
    return data_f

raw_data_Sho = pd.read_csv("./190729_Smart Shoes_HR.csv", encoding='utf-8')  # File Name (Shoes data)
raw_data_Gas = pd.read_csv("./190729_Gas Analysis_HR.csv", encoding='utf-8')  # File Name (Gas analyzer data)
# ISSUE -> When GAS csv files first out is Blank it is not working
# TODO - ADD PREHANDLING WITH OUT PANDAS OR EDIT BEFORE WORK

"""
#ISSUE can't read XLS file @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@HELP!
with open("./190729_Gas Analysis_LSH.csv", 'rb') as f:
  raw_data_Gas = f.read()
"""

# Use this when input data is utf-16 type
"""
with open('./190729_Gas Analysis_LSH.csv',encoding='UTF-16') as f:
raw_data_Gas = pd.read_csv(f)
"""

raw_data_Gas_ts = pd.DataFrame(raw_data_Gas, columns=["t"])
raw_data_Gas_HR = pd.DataFrame(raw_data_Gas, columns=['HR']) # EEm or HR
raw_data_Sho_sen = pd.DataFrame(raw_data_Sho,
                                columns=["Timestamp", "L_sensor1", "L_sensor2", "L_sensor3", "L_sensor4", "R_sensor1",
                                         "R_sensor2", "R_sensor3", "R_sensor4", "L_accX", "L_accY", "L_accZ", "L_gyroX",
                                         "L_gyroY", "L_gyroZ", "R_accX", "R_accY", "R_accZ", "R_gyroX", "R_gyroY",
                                         "R_gyroZ", "COP_Left", "COP_Right", "COP_Front", "COP_Back", "COP_LeftFront",
                                         "COP_LeftBack", "COP_RightFront", "COP_RightBack"])
raw_data_Sho_ts = pd.DataFrame(raw_data_Sho, columns=["Timestamp"])

raw_data_Gas_ts = raw_data_Gas_ts.drop([0, 1])
raw_data_Gas_HR = raw_data_Gas_HR.drop([0, 1])

raw_data_Gas_HR = re_indexing(raw_data_Gas_HR)
raw_data_Gas_ts = re_indexing(raw_data_Gas_ts)

n_raw_data_Gas_ts = raw_data_Gas_ts.get_values()

time_stamp = []

for a in range(int(raw_data_Gas_HR.shape[0])):
    term = n_raw_data_Gas_ts[a]
    term = str(term)
    term_s = term[-4:-2]
    term_m = term[-7:-5]
    term_h = term[-9:-8]
    sec_rcd = int(term_s) + int(term_m) * 60 + int(term_h) * 3600
    time_stamp.append(sec_rcd)

raw_data_Gas_ts = pd.DataFrame(time_stamp, columns=["Time Stamp"])

Generated_HR = []
Gap_list = []

# Delete duplicate things (Same time Stamp)
for c in range(int(len(time_stamp) - 10)):  # Error point
    if int(time_stamp[c + 1]) - int(time_stamp[c]) == 0:
        raw_data_Gas_HR = raw_data_Gas_HR.drop([c + 1])
        raw_data_Gas_ts = raw_data_Gas_ts.drop([c + 1])

        raw_data_Gas_HR = re_indexing(raw_data_Gas_HR)
        raw_data_Gas_ts = re_indexing(raw_data_Gas_ts)

# Generate Gap list
for c in range(int(raw_data_Gas_HR.shape[0]) - 1):
    if int(time_stamp[c + 1]) - int(time_stamp[c]) != 1:
        t1 = int(raw_data_Gas_ts.iloc[c + 1])
        t2 = int(raw_data_Gas_ts.iloc[c])
        Gap_list.append(t1 - t2)

sho_max_sec = int(raw_data_Sho_ts.iloc[int(raw_data_Sho_ts.shape[0]) - 1] / 1000)

n_ts = []
n_HR = []
for c in range(int(raw_data_Gas_HR.shape[0]) - 1):
    gap = 0
    for n in range(int(time_stamp[c + 1]) - int(time_stamp[c])):
        n_ts.append(time_stamp[c] + gap)
        n_HR.append(raw_data_Gas_HR.values[c])
        gap = gap + 1

n_HR_P = pd.DataFrame(n_HR)
n_ts_P = pd.DataFrame(n_ts)
HR_ts = pd.concat([n_ts_P, n_HR_P], axis=1)
n_HR_li = n_HR_P.values.tolist()  # test

# for n in range(int(time_stamp[c+1]) - int(time_stamp[c])):
# n_ts.append(time_stamp[c])
HR_list = []

for c in range(int(raw_data_Sho_ts.shape[0])):
    sho_sec = (raw_data_Sho_ts.iloc[c] / 1000)
    HR_list.append(n_HR_li[int(sho_sec - 10)])  # Use [HR_list] for Regression
    # Error point

HR_list_pd = pd.DataFrame(HR_list)

Usable_RG = raw_data_Sho_sen.drop(
    ["COP_Right", "COP_Front", "COP_Back", "COP_LeftFront", "COP_LeftBack", "COP_RightFront", "COP_RightBack",
     "COP_Left"], axis=1)
Usable_RG = pd.concat([Usable_RG, HR_list_pd], axis=1)
Usable_RG = Usable_RG.rename(columns={0: 'EEm'})
# Usable_RG has HR Data and Shoes sensor data!! Use THIS!!

Usable_RG.to_csv('PreProcessed ' + "Shoes_HR_LSH_TEST.csv", mode='w')