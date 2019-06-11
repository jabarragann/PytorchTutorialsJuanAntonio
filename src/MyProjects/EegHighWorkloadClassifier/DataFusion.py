import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


trial = 7
print("Preprocess Data")

epoc = pd.read_csv("./Data/S1_T{:d}_Epoc.txt".format(trial), sep=' ')
shimmer = pd.read_csv("./Data/S1_T{:d}_Shimmer.txt".format(trial), sep=' ')
secondaryTaskStamps = pd.read_csv("./Data/S1_T{:d}_Timestamps.txt".format(trial), sep=' ', skiprows=[0,1,2,3])
fiveSecondWindowStamps = pd.read_csv("./Data/S1_T{:d}_TimestampEvery5Seconds.txt".format(trial), sep=' ')


epocSlice = epoc[['AF3', 'F7', 'F3', 'FC5', 'T7',
                  'P7', 'O1', 'O2', 'P8', 'T8',
                  'FC6', 'F4', 'F8', 'AF4', 'COMPUTER_TIME']]
epocArr = epocSlice.values

shimmerSlice = shimmer[['PPG','COMPUTER_TIME']]
shimmerArr = shimmerSlice.values

secondaryTaskArr = secondaryTaskStamps.values
windowStampsArr = fiveSecondWindowStamps.values

finalArrayEpoc = np.ones((epocArr.shape[0], epocArr.shape[1] + 3)) * -10
finalArrayEpoc[:, list(range(15))] = epocArr

finalArrayShimmer = np.ones((shimmerArr.shape[0], shimmerArr.shape[1] + 3)) * (-10)
finalArrayShimmer[:, list(range(2))] = shimmerArr

#Apply a band pass Butterworth Filter to each of the 14 channels of the EEG
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=7):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=7):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#EPOC Filtering
lowCut  = 0.5
highCut = 50
for i in range(14):
    temp = butter_bandpass_filter(finalArrayEpoc[:, i], lowCut, highCut, 128, order=7)
    finalArrayEpoc[:, i] = temp

#Shimmer Filtering
temp = butter_bandpass_filter(finalArrayShimmer[:, 0], lowCut, highCut, 128, order=7)
finalArrayShimmer[:, 0] = temp

#Remove 10 first seconds of data to take away outliers and transient state created by the band pass filter
tempFinal = np.ones((epocArr.shape[0] - 700 , epocArr.shape[1]+3)) * -10
tempFinal = finalArrayEpoc[700:, :]
finalArrayEpoc = tempFinal

tempFinal = np.ones((shimmerArr.shape[0] - 700 , shimmerArr.shape[1]+3)) * -10
tempFinal = finalArrayShimmer[700:, :]
finalArrayShimmer = tempFinal

#### EEG PRE PROCESSING ####
#Add Secondary task timestamps to EEG data
idx = 0
for i in range(secondaryTaskArr.shape[0]):
    timestamp = secondaryTaskArr[i,0]
    while timestamp > epocArr[idx,14]:
        idx += 1
        if idx == finalArrayEpoc.shape[0]:
            break

    if idx == finalArrayEpoc.shape[0]:
        break
    if idx < finalArrayEpoc.shape[0]:
        finalArrayEpoc[idx, -3] = timestamp

#Add 5 Seconds Windows timestamps to EEG data
idx = 0
for i in range(windowStampsArr.shape[0]):
    timestamp = windowStampsArr[i,0]
    while timestamp > epocArr[idx,14]:
        idx += 1
        if idx == finalArrayEpoc.shape[0]:
            break

    if idx == finalArrayEpoc.shape[0]:
        break
    if idx < finalArrayEpoc.shape[0]:
        finalArrayEpoc[idx, -2] = timestamp

# Add Label
count = 0
for idx in range(finalArrayEpoc.shape[0]):
    time = finalArrayEpoc[idx, 14]

    if count == secondaryTaskArr.shape[0] -1:
        finalArrayEpoc[idx, -1] = 1 if secondaryTaskArr[count, 1] is True else 0
    else:
        if time > secondaryTaskArr[count,0] and  time < secondaryTaskArr[count+1,0]:
            finalArrayEpoc[idx, -1] = 1 if secondaryTaskArr[count, 1] is True else 0
        elif time > secondaryTaskArr[count+1, 0]:
            finalArrayEpoc[idx, -1] = 1 if secondaryTaskArr[count + 1, 1] is True else 0
            count += 1

# Write Final File
with open('./Data/S1_T{:d}_fusion_epoc.txt'.format(trial),'w') as fout:
    fout.write("AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4 COMPUTER_TIME SECONDARY_TASK 5_SECOND_WINDOW LABEL\n")
    for i in range(finalArrayEpoc.shape[0]):
        formattedData = " ".join(["{:.8f}".format(d) for d in finalArrayEpoc[i]])
        fout.write(formattedData+'\n')


#### SHIMMER PRE PROCESSING ####
#Add Secondary task timestamps to EEG data
idx = 0
for i in range(secondaryTaskArr.shape[0]):
    timestamp = secondaryTaskArr[i,0]
    while timestamp > shimmerArr[idx,1]:
        idx += 1
        if idx == finalArrayShimmer.shape[0]:
            break

    if idx == finalArrayShimmer.shape[0]:
        break
    if idx < finalArrayShimmer.shape[0]:
        finalArrayShimmer[idx, -3] = timestamp

#Add 5 Seconds Windows timestamps to EEG data
idx = 0
for i in range(windowStampsArr.shape[0]):
    timestamp = windowStampsArr[i,0]
    while timestamp > shimmerArr[idx,1]:
        idx += 1
        if idx == finalArrayShimmer.shape[0]:
            break

    if idx == finalArrayShimmer.shape[0]:
        break
    if idx < finalArrayShimmer.shape[0]:
        finalArrayShimmer[idx, -2] = timestamp

# Add Label
count = 0
for idx in range(finalArrayShimmer.shape[0]):
    time = finalArrayShimmer[idx, 1]

    if count == secondaryTaskArr.shape[0] -1:
        finalArrayShimmer[idx, -1] = 1 if secondaryTaskArr[count, 1] is True else 0
    else:
        if time > secondaryTaskArr[count,0] and  time < secondaryTaskArr[count+1,0]:
            finalArrayShimmer[idx, -1] = 1 if secondaryTaskArr[count, 1] is True else 0
        elif time > secondaryTaskArr[count+1, 0]:
            finalArrayShimmer[idx, -1] = 1 if secondaryTaskArr[count + 1, 1] is True else 0
            count += 1

# Write Final File
with open('./Data/S1_T{:d}_fusion_shimmer.txt'.format(trial),'w') as fout:
    fout.write("PPG COMPUTER_TIME SECONDARY_TASK 5_SECOND_WINDOW LABEL\n")
    for i in range(finalArrayShimmer.shape[0]):
        formattedData = " ".join(["{:.8f}".format(d) for d in finalArrayShimmer[i]])
        fout.write(formattedData+'\n')