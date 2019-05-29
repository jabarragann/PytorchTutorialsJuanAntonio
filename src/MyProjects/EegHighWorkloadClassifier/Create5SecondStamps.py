
import pandas as pd


if __name__ == '__main__':

    trial = 7
    file = open("Data/S1_T{:d}_Timestamps.txt".format(trial),"r")
    data = pd.read_csv(file, sep=' ', skiprows=[0,1,2,3])

    timestamps = data.values[:,0]

    outputFile = open("Data/S1_T{:d}_TimestampEvery5Seconds.txt".format(trial),"w")
    outputFile.write("5SecondsWindows\n")

    initTime = timestamps[0]
    finalTime = timestamps[-1] + 30
    increase = 5
    while initTime < finalTime:
        outputFile.write("{:.9f}\n".format(initTime))
        initTime += increase

    print("Finish")

    file.close()
    outputFile.close()



