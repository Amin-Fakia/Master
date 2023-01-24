import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import remez, minimum_phase, freqz, group_delay,firwin
from scipy import signal
import mne 


freq1=3 # frequency 3 Hz
freq2=6 # frequency 6 Hz

x = np.arange(0,1,1/300)   # start,stop,step

y1 = np.sin(2*np.pi*freq1*x) 
y2 = np.sin(2*np.pi*freq2*x)
y= y1+y2

fig,ax = plt.subplots(3,2)

ax[0][0].set_ylim((-2,2))
ax[0][0].grid(True, which='both')
# plt.hlines((0),-0.1,x[-1],colors=("k"))
# plt.vlines((0),-2,2,colors=("k"))
ax[0][0].plot(x,y1)
ax[0][0].set_xlabel("Time in s")
ax[0][0].set_ylabel("amplitude")
ax[0][0].legend()
ax[0][0].set_title("3 Hz")


ax[1][0].set_ylim((-2,2))
ax[1][0].grid(True, which='both')
# plt.hlines((0),-0.1,x[-1],colors=("k"))
# plt.vlines((0),-2,2,colors=("k"))
ax[1][0].plot(x,y2)
ax[1][0].set_xlabel("Time in s")
ax[1][0].set_ylabel("amplitude")
ax[1][0].legend()
ax[1][0].set_title("6 Hz")


ax[2][0].set_ylim((-2,2))
ax[2][0].grid(True, which='both')
# plt.hlines((0),-0.1,x[-1],colors=("k"))
# plt.vlines((0),-2,2,colors=("k"))
ax[2][0].plot(x,y,c="red")
ax[2][0].set_xlabel("Time in s")
ax[2][0].set_ylabel("amplitude")
ax[2][0].legend()
ax[2][0].set_title("3 Hz + 6 Hz")
plt.tight_layout()


## filters


fil1 = mne.filter.filter_data(y,l_freq=None,h_freq=3,sfreq=300,phase="minimum")
fil2 = mne.filter.filter_data(y,l_freq=None,h_freq=3,sfreq=300,phase="zero")
fil3 = mne.filter.filter_data(y,l_freq=None,h_freq=3,sfreq=300,phase="zero-double")
# ##
# The filter length is chosen based on the size of the transition regions
#  (6.6 times the reciprocal of the shortest transition band 
#  for fir_window=’hamming’ and fir_design=”firwin2”, and half that for “firwin”).


# fig2,ax2 = plt.subplots(3,1)
ax[0][1].set_ylim((-2,2))
ax[0][1].grid(True, which='both')
ax[0][1].plot(x,fil1)
ax[0][1].plot(x,y1,"--")
ax[0][1].set_title("minimum phase filter")

ax[1][1].set_ylim((-2,2))
ax[1][1].grid(True, which='both')
ax[1][1].plot(x,fil2)
ax[1][1].plot(x,y1,"--")
ax[1][1].set_title("zero-phase filter")

ax[2][1].set_ylim((-2,2))
ax[2][1].grid(True, which='both')
ax[2][1].plot(x,fil3)
ax[2][1].plot(x,y1,"--")
ax[2][1].set_title("zero-double-phase filter")

plt.tight_layout()
plt.show()