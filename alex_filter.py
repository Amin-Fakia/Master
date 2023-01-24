from numpy.core._multiarray_umath import arctan2
#from matplotlib.pyplot import legend, subplot
from scipy import signal
from pylab import *
from numpy import zeros, log10, angle

#Plot frequency and phase response
def mfreqz(b,a=1, nyq=1):
    w,h = signal.freqz(b,a)
    h_dB = 20 * log10 (abs(h))
    subplot(211)
    plot(w/max(w)*nyq,h_dB)
    ylim(-150, 5)
    ylabel('Magnitude (db)')
    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    title(r'Frequency response')
    subplot(212)
    h_Phase = unwrap(arctan2(imag(h),real(h)))
    plot(w/max(w)*nyq,h_Phase)
    ylabel('Phase (radians)')
    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    title(r'Phase response')
    subplots_adjust(hspace=0.5)

#data = random.random(2000)
sample_rate = 150.0

nsamples = 500
t = arange(nsamples) / sample_rate
data = cos(2*pi*2*t)+ cos(2*pi*5*t) + 0.5*cos(2*pi*10*t)
x1 = cos(2*pi*2*t)+ cos(2*pi*5*t)
order = 143
nyq = sample_rate/2
b = signal.firwin(order, 20/nyq)
z = signal.lfilter_zi(b, 1) * data[0]

#------------------------------------------------
 # Plot the magnitude response of the filter.
 #------------------------------------------------

figure(1)
clf()
w, h = freqz(b, worN=8000)
plot((w/pi)*nyq, absolute(h), linewidth=2)
xlabel('Frequency (Hz)')
ylabel('Gain')
title('Frequency Response')
ylim(-0.05, 1.05)
grid(True)

mfreqz(b, nyq=nyq)



#plot(b)
result = zeros(data.size)
#z = 0
for i, x in enumerate(data):
    result[i], z = signal.lfilter(b, 1, [x], zi=z)

filteredSig = signal.lfilter(b, 1, data)
figure(0)
plot(filteredSig)
figure(1)
#for i, x in enumerate(data):
#    result[size(result)-i-1], z = signal.lfilter(b, 1, [x], zi=z)


delay =  0.5*(order-1) / sample_rate
print(delay)
figure(2)
plot(t,data, "r-", label="original Sig")
#plot(t, x1, "g-", label="")
#figure(3)
plot(t - delay, filteredSig)
#plot(t - delay,result, linewidth=2, label="filtered sig")
xlim(0)
legend();