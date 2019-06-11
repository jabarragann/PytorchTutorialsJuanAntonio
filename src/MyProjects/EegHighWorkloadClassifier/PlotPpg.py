import matplotlib.pyplot as plt
import pandas as pd
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

lowCut  = 0.5
highCut = 50

trial = 5
print("Print PPG signal")
shimmer = pd.read_csv("./Data/S1_T{:d}_shimmer.txt".format(trial), sep=' ')
shimmerSlice = shimmer[['PPG']].values[700:]


plt.plot(shimmerSlice)
shimmerSlice = butter_bandpass_filter(shimmerSlice, lowCut, highCut, 128, order=7)

plt.plot(shimmerSlice)
plt.show()