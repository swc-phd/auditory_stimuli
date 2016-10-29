#Code to generate .wav file for auditory stimulation for SWC PhD 1000 channel 
#probe experiment

# Authors:
# Matthew Phillips
# Steven Lenzi
# Jesse Geerts
# Jorge A Menendez

#SSA and mywave
#Take in preferred freq, generate stimulus
#SSA picks two frequencies near best frequency,

#mywave generates a waveform freq + harmonics (up to 4th harmonic)
#generates all 4 imposed on each other
#then only 2nd, 3rd + 4th without original
#sampling rate, amplitude 
#string together in random order, save as a .wav file

import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
plt.close()

# Parameters
best_freq = 5000
a = 2
fs = 44100.0
#saveto = 'C:\\Users\\jgeerts\\Desktop\\'
saveto = '/nfs/nhome/live/jorgem/Desktop'

## Stimulus info
start_blank = 3;
click_duration = 0.5;
stim_duration = 0.1*fs
stim_reps = 20 # 10 reps = 6 min stimulus
bws = np.arange(50,110,10)
bw_density = 10
f = best_freq
isi_duration = 0.4*fs

## Wave-generating function (with tapering)
# f: frequency of wave
# fs: sampling frequency
# a: amplitude of wave
# stim_duration: duration of the wave, in samples
def mywave(f, fs, a, stim_duration):
    tapering = .1
    n = np.sin(np.linspace(0,0.5*np.pi,np.floor(stim_duration*tapering)))
    m = np.concatenate([n,np.ones((1-tapering*2)*stim_duration),n[::-1]])
    y = [a*np.sin(2*np.pi*f*(i/fs)) for i in np.arange(stim_duration)] * m
    return y

## Harmonics
f0 = mywave(f, fs, a, stim_duration)
f1 = mywave(f*2, fs, a, stim_duration)
f2 = mywave(f*3, fs, a, stim_duration)
f3 = mywave(f*4, fs, a, stim_duration)
fall = (f0 + f1 + f2 + f3) / 4
fharms = (f1 + f2 + f3) / 3

## Bandwidths
bw_freqs = -1*np.ones([bws.shape[0],bw_density])
for i,bw in enumerate(bws):
    minf = best_freq - (bw**2)/2
    maxf = best_freq + (bw**2)/2
    bw_freqs[i,:] = np.concatenate([np.round(np.linspace(minf,best_freq,bw_density/2,False)), np.round(np.linspace(maxf,best_freq,bw_density/2,False))])

## Generate waveforms for each frequency in bandwidths
bw_waves = -1*np.ones([bw_freqs.size, stim_duration])
bw_waves_combined = np.zeros([bw_freqs.shape[0], stim_duration])
for bw in np.arange(bw_freqs.shape[0]):
    for i in np.arange(bw_freqs.shape[1]):
        bw_indx = bw*bw_density+i
        bw_waves[bw_indx,:] = mywave(bw_freqs[bw,i],fs,a,stim_duration)
        bw_waves_combined[bw,:] = bw_waves_combined[bw,:] + bw_waves[bw_indx,:]
    bw_waves_combined[bw,:] = (bw_waves_combined[bw,:] + f0)/(bw_density+1)

## Randomize order of presentation
all_waves = np.vstack((f0, f1, f2, f3, fall, fharms, bw_waves, bw_waves_combined))
rand_order = np.random.permutation(np.repeat(np.arange(all_waves.shape[0]),stim_reps))

## String together into one waveform, with ISIs in between
myISI = np.zeros(isi_duration)
stim_waveform = []
for i in rand_order:
    stim_waveform = np.concatenate([stim_waveform,all_waves[i,:],myISI])
clicks = np.tile(np.concatenate([np.zeros(click_duration*fs),np.ones(click_duration*fs)*20,np.zeros(click_duration*fs)]),3)
stim_waveform = np.concatenate([clicks,stim_waveform,clicks])
stim_waveform = np.concatenate([np.zeros(fs*start_blank), stim_waveform]) # add start_blank seconds of silence to the beginning

## Create waveform with stimulus onset times
# one with short square wave at onset time
onset_signal1 = np.zeros_like(stim_waveform)
onset_signal1[(start_blank*fs + 1):-1:(stim_duration+isi_duration)] = 1
# one with square waves lasting the duration of each stimulus
onset_signal2 = np.zeros_like(stim_waveform)
for i in np.arange(rand_order.size):
    t_on = (start_blank*fs + 1) + (stim_duration+isi_duration)*i
    t_off = t_on + stim_duration
    onset_signal2[t_on:t_off] = 1

## Save stereo .wav files with each onset time sequence
scipy.io.wavfile.write(saveto+'bandwidth_stim_stimpulse.wav', 44100, np.int16(np.vstack([stim_waveform,onset_signal1]).T)*32767)
scipy.io.wavfile.write(saveto+'bandwidth_stim_squarewave.wav', 44100,
np.int16(np.vstack([stim_waveform,onset_signal2]).T)*32767)

## Save all_waves, and rand_order
np.savetxt(saveto+'all_waves.txt',all_waves,delimiter=',')
np.savetxt(saveto+'rand_order.txt',rand_order,delimiter=',')


