import matplotlib.pyplot as plt
import librosa.display
import librosa

file = 'MFCC.wav'

y, sr = librosa.load(file, sr=16000)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16, hop_length=512)

fig, ax = plt.subplots(figsize=(15, 6))
show = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax, hop_length=512)
print('MFCCs.shape :', mfccs.shape)
fig.colorbar(show, ax=ax)
ax.set(title='MFCC')
plt.show()
