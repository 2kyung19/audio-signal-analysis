import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

file = "이경은.wav"

#Fourier -> Spectrum
sig, sr = librosa.load(file, sr=None)

fft = np.fft.fft(sig)
print("fft shape : ", fft.shape)

magnitude = np.abs(fft)
print("spectrum shape : ", magnitude.shape)

f = np.linspace(0, sr, len(magnitude))
print("f shape : ", f.shape)

left_spectrum = magnitude[:int(len(magnitude) / 2)]
left_f = f[:int(len(magnitude) / 2)]
print("left_spectrum shape : ", left_spectrum.shape)
print("left_f shape : ", left_f.shape)

plt.figure(figsize=None)
plt.plot(left_f, left_spectrum, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.show()