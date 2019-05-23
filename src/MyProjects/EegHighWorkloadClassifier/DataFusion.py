import pandas as pd
import numpy as np

trial = 1
print("Preprocess Data")

epoc = pd.read_csv("./Data/S1_T{:d}_Epoc.txt".format(trial), sep=' ')
secondaryTaskStamps = pd.read_csv("./Data/S1_T{:d}_Timestamps.txt".format(trial), sep=' ')
fiveSecondWindowStamps = pd.read_csv("./Data/S1_T{:d}_TimestampEvery5Seconds.txt".format(trial), sep=' ')


epocSlice = epoc[['AF3', 'F7', 'F3', 'FC5', 'T7',
                  'P7', 'O1', 'O2', 'P8', 'T8',
                  'FC6', 'F4', 'F8', 'AF4', 'COMPUTER_TIME']]
epocArr = epocSlice.values
secondaryTaskArr = secondaryTaskStamps.values
windowStampsArr = fiveSecondWindowStamps.values


finalArray = np.ones((epocArr.shape[0], epocArr.shape[1]+3))*-10
finalArray[:,list(range(15))] = epocArr


# Secondary task timestamps
idx = 0
for i in range(secondaryTaskArr.shape[0]):
    timestamp = secondaryTaskArr[i,0]
    while timestamp > epocArr[idx,14]:
        idx += 1
        if idx == finalArray.shape[0]:
            break

    if idx == finalArray.shape[0]:
        break
    if idx < finalArray.shape[0]:
        finalArray[idx, -3] = timestamp

# 5 Seconds Windows
idx = 0
for i in range(windowStampsArr.shape[0]):
    timestamp = windowStampsArr[i,0]
    while timestamp > epocArr[idx,14]:
        idx += 1
        if idx == finalArray.shape[0]:
            break

    if idx == finalArray.shape[0]:
        break
    if idx < finalArray.shape[0]:
        finalArray[idx, -2] = timestamp

# Add Label
count = 0
for idx in range(finalArray.shape[0]):
    time = finalArray[idx, 14]

    if count == secondaryTaskArr.shape[0] -1:
        finalArray[idx, -1] = 1 if secondaryTaskArr[count, 1] is True else 0
    else:
        if time > secondaryTaskArr[count,0] and  time < secondaryTaskArr[count+1,0]:
            finalArray[idx,-1] = 1 if secondaryTaskArr[count,1] is True else 0
        elif time > secondaryTaskArr[count+1, 0]:
            finalArray[idx, -1] = 1 if secondaryTaskArr[count+1, 1] is True else 0
            count += 1

# Write Final File
with open('./Data/S1_T{:d}_fusion.txt'.format(trial),'w') as fout:
    fout.write("AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4 COMPUTER_TIME SECONDARY_TASK 5_SECOND_WINDOW LABEL\n")
    for i in range(finalArray.shape[0]):
        formattedData = " ".join(["{:.8f}".format(d) for d in finalArray[i]])
        fout.write(formattedData+'\n')