import os
import pickle
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from hmmlearn import hmm

def trainmodel(model,wavlist):
    X = np.array([])
    for wavfile in wavlist:
        path = os.path.join('trainingwav', wavfile)
        (rate, sig) = wav.read(path)
        mfcc_feat = mfcc(sig, rate, nfft=1024)

        if len(X) == 0:
            X = mfcc_feat
        else:
            X = np.append(X, mfcc_feat, axis=0)

    modelSayi = hmm.GaussianHMM(n_components=5, covariance_type='diag', n_iter=1000)
    modelSayi.fit(X)

    hmmPath = os.path.join('hmm',model)
    file = open(hmmPath,"wb")
    pickle.dump(modelSayi,file)
    file.close()

def main():
    trainmodel('bir', ['trainingwav_1_01.wav','trainingwav_1_02.wav','trainingwav_1_03.wav', 'trainingwav_1_04.wav','trainingwav_1_05.wav'])
    trainmodel('iki',['trainingwav_2_01.wav','trainingwav_2_02.wav','trainingwav_2_03.wav', 'trainingwav_2_04.wav','trainingwav_2_05.wav'])
    trainmodel('uc',['trainingwav_3_01.wav', 'trainingwav_3_02.wav', 'trainingwav_3_03.wav', 'trainingwav_3_04.wav', 'trainingwav_3_05.wav'])
    trainmodel('dort',['trainingwav_4_01.wav','trainingwav_4_02.wav','trainingwav_4_03.wav', 'trainingwav_4_04.wav','trainingwav_4_05.wav'])
    trainmodel('bes',['trainingwav_5_01.wav','trainingwav_5_02.wav','trainingwav_5_03.wav','trainingwav_5_04.wav','trainingwav_5_05.wav'])
    trainmodel('alti',['trainingwav_6_01.wav', 'trainingwav_6_02.wav', 'trainingwav_6_03.wav', 'trainingwav_6_04.wav', 'trainingwav_6_05.wav'])
    trainmodel('yedi',['trainingwav_7_01.wav', 'trainingwav_7_02.wav', 'trainingwav_7_03.wav', 'trainingwav_7_04.wav', 'trainingwav_7_05.wav'])
    trainmodel('sekiz',['trainingwav_8_01.wav', 'trainingwav_8_02.wav', 'trainingwav_8_03.wav', 'trainingwav_8_04.wav', 'trainingwav_8_05.wav'])
    trainmodel('dokuz',['trainingwav_9_01.wav','trainingwav_9_02.wav','trainingwav_9_03.wav', 'trainingwav_9_04.wav','trainingwav_9_05.wav'])
    trainmodel('on',['trainingwav_10_01.wav', 'trainingwav_10_02.wav', 'trainingwav_10_03.wav', 'trainingwav_10_04.wav', 'trainingwav_10_05.wav'])
if __name__ == '__main__':
    main()