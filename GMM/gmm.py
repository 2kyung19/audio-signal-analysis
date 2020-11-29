import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

print("Start Learning...")

path = './'

audio = ['F1.wav', 'F2.wav', 'M1.wav', 'M2.wav']
speakers = ['F1','F2','M1','M2']

# learn
all_mfccs = []
for file_name in audio:
    y, sr = librosa.load(path + file_name, sr=16000)
    y = y / np.max(np.abs(y))

    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=24, hop_length=512).T

    mfcc -= np.mean(mfcc, axis=0)
    mfcc /= np.std(mfcc, ddof=0, axis=0)

    all_mfccs.append(mfcc)

gmms = []
for idx in range(len(speakers)):
    gmm = GaussianMixture(n_components=5, covariance_type='tied', max_iter=100, verbose=0)
    training_mfccs = all_mfccs[idx]
    gmm.fit(training_mfccs)
    gmms.append(gmm)

# test
while(1):
    print('\ninput file name or exit')
    print('test file : filename (don\'t write .wav)')
    print('program exit : exit')

    order = input('\ninput text : ')

    if(order == 'exit'):
        print('exit...')
        break

    y, sr = librosa.load(path + order + '.wav', sr=16000)
    y = y / np.max(np.abs(y))

    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=24, hop_length=512).T

    mfcc -= np.mean(mfcc, axis=0)
    mfcc /= np.std(mfcc, ddof=0, axis=0)

    score = []
    for idx, model in enumerate(gmms):
        score.append(model.score(mfcc))
    result = np.argmax(score)
    print(score[result])
    file = open(path + 'result.txt', 'a')
    file.write(speakers[result]+"\n")
    file.close()

    print('success!')
