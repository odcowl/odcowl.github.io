#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:17:08 2021

@author: yujiayang
"""


import numpy as np

def calc_stft(signal, sample_rate=48000, frame_size=0.025, frame_stride=0.01, winfunc=np.hamming, NFFT=512):
    
    #Calculate the number of frames from the signal 计算整个信号要分多少个帧
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_length = len(signal)
    num_frames = 1 + int(np.ceil(float(np.abs(signal_length - frame_length))/ frame_step))
    
    #zero padding 补零
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # pad sigal makes sure that all frames have equal number of samples
    # without truncation any samples from the original signal
    pad_signal = np.append(signal,z) #np.append(arr,values)为原始数组arr添加values
    
    #slice the signal into frames from indices
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
            np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    #get windows frames
    frames *= winfunc(frame_length)
    
    # Compute the one-dimensional n-point discrete Fourier Transform(DFT) of
    # a real-valued array by means of an efficient algorithm called Fast Fourier Transform (FFT)
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    
    # Compute power spectrum
    pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)
    
    return pow_frames

if __name__ == '__main__':
    import scipy.io.wavfile
    import matplotlib.pylab as plt
    
    #read a wav file
    sample_rate, signal = scipy.io.wavfile.read("p282_023.wav")
    signal = signal[0:int(2. * sample_rate)]
    
    #calculate the STFT
    pow_spec = calc_stft(signal,sample_rate)
    
    plt.imshow(pow_spec)
    #plt.tight_layout()
    plt.show()