from scipy.signal import filtfilt,butter,freqs,firwin,sosfilt
import mne
import numpy as np
import matplotlib.pyplot as plt


real_data = mne.io.read_raw_edf("./EEG_Eyes_Open_Closed_raw.edf")


freq1=3 # frequency 3 Hz
freq2=6 # frequency 6 Hz

x = np.arange(0,1 ,1/300)   # start,stop,step

y1 = np.sin(2*np.pi*freq1*x) 
y2 = np.sin(2*np.pi*freq2*x)
y= y1+y2







#Zero-Phase Filtering within specified window sizes 

def window_filfit(data,order,cutoff,win_size,step_size=1):
    data_size = len(data)
    yfs = []
    for i in np.arange(0,data_size,win_size):
        b, a = butter(order,cutoff)
        yf = filtfilt(b,a,data[i:i+win_size])
        for j in yf:
            yfs.append(j)
    return yfs
        


yfs = window_filfit(y,6,.03,150)

plt.figure()
plt.plot(y,label="6 Hz + 3 Hz")
plt.legend()
plt.show()

# b, a = butter(7,0.03, analog=True)
# yf = filtfilt(b,a,y)
plt.figure()
plt.title("window filter params: window size: 100, filter order: 6, cutoff-frequency: 3 Hz ")
plt.plot(y1,label="3 Hz (real)")
plt.plot(yfs,label=f"3 Hz (window filter)")
plt.legend()
plt.show()
plt.figure()
b,a = butter(7,0.03)
yf = filtfilt(b,a,y)

plt.plot(yf,label="3 Hz (butterworth filter)")
plt.plot(y1,label="3 Hz (real)")
plt.legend()



plt.show()
plt.figure()
b, a = butter(7,3, analog=True)
w, h = freqs(b, a)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
#plt.axvline(0.03, color='green')
# plt.legend()
plt.show()