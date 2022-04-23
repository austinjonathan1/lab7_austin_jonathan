import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

np.random.seed(3452)
time_step = 0.1
period = 10

time_vec = np.arange(0, 20, time_step)
sig = (np.sin(20 * np.pi / period * time_vec) + 0.5 * np.random.randn(time_vec.size)) + (np.sin(10 * np.pi / period * time_vec) + 0.5 * np.random.randn(time_vec.size)) + (np.sin(200 * np.pi / period * time_vec)+ 0.5 * np.random.randn(time_vec.size))

plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label = 'Original Signal')
sig_fft = fftpack.fft(sig)

power = np.abs(sig_fft)**2

sample_freq = fftpack.fftfreq(sig.size, d = time_step)

plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')

pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]

np.allclose(peak_freq, 1. / period)

axes = plt.axes([0.55, 0.3, 0.3, 0.5])
plt.title('Peak Frequency')
plt.plot(freqs[:9], power[:9])
plt.setp(axes, yticks = [])

high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = fftpack.ifft(high_freq_fft)

plt.figure(figsize=(6, 5))

plt.title('Time Domain: Original Signal vs. Filtered Signal')
plt.plot(time_vec, sig, label = 'Original Signal')
plt.plot(time_vec, filtered_sig, linewidth = 3, label = 'Filtered Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.legend(loc = 'best')

filtered_sig_fft = fftpack.fft(filtered_sig)
filtered_power = np.abs(filtered_sig_fft)**2
filtered_sig = fftpack.ifft(high_freq_fft)

sig_fft1 = fftpack.fft(filtered_sig)
power1 = np.abs(sig_fft1)**2

sample_freq1 = fftpack.fftfreq(filtered_sig.size, d = time_step)

plt.figure(figsize=(6, 5))
plt.plot(sample_freq1, power1)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')

from scipy import signal
t = time_vec
b, a = signal.butter(5, 0.1)

zi = signal.lfilter_zi(b, a)
z,_=signal.lfilter(b, a, sig, zi = zi * sig[0])

b, a = signal.butter(5, 0.1)
z2, _ = signal.lfilter(b, a, z, zi = zi * z[0])
z3, _ = signal.lfilter(b, a, z, zi = zi * z[0])
z4, _ = signal.lfilter(b, a, z, zi = zi * z[0])
y = .filtfilt(b, a, sig)

plt.figure
plt.plot(t, sig, 'blue', alpha = 0.75)
plt.plot(t, z, 'green', t, z2, 'yellow', t, y, 'pink')
plt.legend(('Original Signal', '1st Filter', '2nd Filter', '3rd Filter'), loc = 'best')
plt.grid(True)
plt.show()
