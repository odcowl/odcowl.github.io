#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 15:12:44 2021

@author: yujiayang
"""


import scipy, pylab

def stft(x, fs, framesz, hop):
    """x is the time-domain signal
    fs is the sampling frequency
    framesz is the frame size, in seconds
    hop is the the time between the start of consecutive frames, in seconds
    """
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

def istft(X, fs, T, hop):
    """X is the short-time Fourier transform
    fs is the sampling frequency
    T is the total length of the time-domain output in seconds
    hop is the the time between the start of consecutive frames, in seconds
    """
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return x

if __name__ == '__main__':
    import scipy.io.wavfile
    import matplotlib.pylab as plt
    from scipy import signal as sgg
    import numpy as np
    sample_rate, signal = scipy.io.wavfile.read("0c2ca723_nohash_1.wav")
    
    X = stft(signal, sample_rate, 0.020, 0.01)
    print(sample_rate)
    #pylab.figure()
    #pylab.imshow(scipy.absolute(X.T), origin='lower', aspect='auto',
     #            interpolation='nearest')
    #pylab.xlabel('Time')
    #pylab.ylabel('Frequency')
    #pylab.show()
    
    f, t, Zxx = sgg.stft(signal,sample_rate, nperseg=100000)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin = 0, vmax = 2 * np.sqrt(2))