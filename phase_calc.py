from numpy.core._multiarray_umath import arctan2
#from matplotlib.pyplot import legend, subplot
from scipy import signal
import matplotlib.pyplot as plt
from pylab import unwrap,imag,real
import streamlit as st
import numpy as np
import mpld3
from scipy.fftpack import fft, ifft
import streamlit.components.v1 as components
import cmath
sample_rate = 300
nyq = sample_rate/2


st.header("Applied Medical Signal Analysis")

st.subheader("Simulation")

freq1 = 4
freq2 = 8
xs = np.linspace(0,1,sample_rate)
ys4 = np.sin(2*np.pi*freq1*xs)
ys8 = np.sin(2*np.pi*freq2*xs)

ys = ys4+ys8

fig,ax = plt.subplots(2)
ax[0].plot(xs,ys)
ax[0].set_ylabel('Amplitude')
ax[0].set_xlabel('time')
ax[0].set_title(f'Signal Simulation of {freq1}+{freq2} Hz')


sp = fft(ys)
N = len(sp)
n = np.arange(N)
T = N/sample_rate
freq = n/T

n_oneside = N//2
f_oneside = freq[:n_oneside]

ax[1].plot(f_oneside,np.abs(sp[:n_oneside]))
ax[1].set_ylabel('Amplitude |X|')
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_title(f'(oneside) FFT of the simulated signal')

fig.tight_layout()
fig_html = mpld3.fig_to_html(fig)
#st.pyplot(fig)
components.html(fig_html, height=500)

st.subheader("Filtering")


#filter_order = st.number_input("Filter Order",min_value=1,max_value=sample_rate,value=231)
#low_freq,high_freq = st.select_slider('Select frequency range',options=range(0,30),value=(1,3))


option = st.selectbox("Which filter type: ",options=["bandpass","lowpass","highpass"],index=0)
cols = st.columns(3)
filter_order = cols[0].number_input("Filter Order",min_value=1,max_value=sample_rate,value=231,step=2)
low_freq = cols[1].number_input("Low Frequency",value=0.01)
high_freq = cols[2].number_input("High Frequency",value=4.0)


if option == "highpass":
    b,a = signal.butter(filter_order,high_freq,btype="highpass",fs=sample_rate)
    b = signal.firwin(filter_order, high_freq, pass_zero='highpass',fs=sample_rate)

if option == "lowpass":
    b,a = signal.butter(filter_order,low_freq,btype="lowpass",fs=sample_rate)
    b = signal.firwin(filter_order, low_freq, pass_zero='lowpass',fs=sample_rate)
if option == "bandpass":
    b,a = signal.butter(filter_order,(low_freq+0.001,high_freq),btype="bandpass",fs=sample_rate)
    b = signal.firwin(filter_order, (low_freq+0.001,high_freq), pass_zero='bandpass',fs=sample_rate)
# b,a = signal.butter(filter_order,(low_freq+0.001,high_freq),btype="bandpass",fs=sample_rate)
w, h = signal.freqz(b)
h_dB = 20 * np.log10 (abs(h))
h_Phase = unwrap(arctan2(imag(h),real(h)))

fig,ax = plt.subplots(2)

ax[0].plot(w/max(w)*nyq,h_Phase)
ax[0].set_ylabel('Phase (radians)')
ax[0].set_xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
ax[0].set_title('Phase response')

ax[1].plot(w/max(w)*nyq,h_dB)
ax[1].set_ylim(-150, 5)
ax[1].set_ylabel('Magnitude (db)')
ax[1].set_xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
ax[1].set_title(r'Frequency response')

ws = w/max(w)*nyq
w_h = [(x,y) for x,y in zip(ws,h_Phase)]

# print(w_h)
# print(w[3]-w[1])
fig.tight_layout()
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=500)

"sampling rate: ",sample_rate


"Phase Response of a digital bandpass butterworth filter with following params: "
f"* *Filter Order: {filter_order}*"

if option == "highpass":
    f"* *{option}: {high_freq}  Hz*"
if option == "lowpass":
   f"* *{option}: {low_freq}  Hz*"
if option == "bandpass":
    f"* *{option}: {low_freq} - {high_freq}  Hz*"



# st.pyplot(fig_html)

st.write(f"frequency resolution: {ws[3] - ws[2]:.5f}")
"frequency - phase reponse in list:"
w_h

"for frequency" ,w_h[14][0] ," Hz the phase shift in (radians?) ", w_h[14][1]

st.info("Applying the FIR filter")

#filtered_ys = signal.lfilter(b,1,ys)
filtered_ys = signal.filtfilt(b,1,ys, padlen=150)
filtered_ys_normal = signal.lfilter(b,1,ys)
fig,ax = plt.subplots(2)
ax[0].plot(xs,filtered_ys,label="zero-phase")
ax[0].set_ylabel('Amplitude')
ax[0].set_xlabel('time')
ax[0].set_title(f'filtered Signal for 4 Hz')
ax[0].plot(xs,ys4,label="original signal")


ax[0].plot(xs,filtered_ys_normal,label="without zero-phase")
ax[0].legend()


sp = np.abs(fft(filtered_ys))
sp2 = np.abs(fft(filtered_ys_normal))
N = len(sp)
n = np.arange(N)
T = N/sample_rate
freq = n/T

n_oneside = N//2
f_oneside = freq[:n_oneside]


ax[1].set_ylabel('Amplitude | X |')
ax[1].set_xlabel('time')
ax[1].set_title(f'FFT of filtered signal')
ax[1].plot(f_oneside,sp[:n_oneside],label="FFT zero-phase")
ax[1].plot(f_oneside,sp2[:n_oneside],label="FFT without zero-phase")

ax[1].legend()
ax[1].fill_between(f_oneside, sp[:n_oneside],alpha=.5)
ax[1].fill_between(f_oneside, sp2[:n_oneside],alpha=.5)
fig.tight_layout()


# ax[2].set_ylabel('Phase')
# ax[2].set_xlabel('Frequency')
# ax[2].set_title(f'FFT Phase')

# phase = np.angle(sp[:n_oneside])
# phase2 = np.angle(sp2[:n_oneside])

# ax[2].plot(f_oneside,phase,label="FFT zero-phase")
# ax[2].plot(f_oneside,phase2,label="FFT without zero-phase")




fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=500)

st.subheader("Inverse FFT")




st.code(
    f"""
# This is without zero phase filtered data
sp2 = np.abs(fft(filtered_ys_normal))
N = len(sp2)
n = np.arange(N)
T = N/sample_rate
freq = n/T

n_oneside = N//2
f_oneside = freq[:n_oneside]

phaseShift_sp2 = sp2 * np.exp(1.0j * {w_h[14][1]:.2f})
test = ifft(phaseShift_sp2)   

"""
)
phaseShift_sp2 = sp2 * np.exp(1.0j * w_h[14][1]) # np.deg2rad(
test = ifft(phaseShift_sp2)

i_data = ifft(sp2)
delay =  0.5*(filter_order-1) / sample_rate
fig,ax = plt.subplots()
ax.plot(xs,i_data,label="filtered signal (ifft)",alpha=0.9)
ax.set_ylabel('Amplitude')
ax.set_xlabel('time')
ax.set_title(f'Inverse Fourier Transform of filtered signal without zero-phase')
ax.plot(xs,ys4,label="original signal",linestyle='dashed')
ax.plot(xs,test,label=f"filtered signal - phase shift",alpha=0.9)
ax.legend()

fig.tight_layout()
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=500)

# st.pyplot(fig)

