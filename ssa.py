import os
os.chdir('/Users/stephenlenzi/Desktop/') # specify the current directory (i.e. where to save)

import numpy as np
import scipy
import scipy.io.wavfile

def mywave(f, fs, a, stim_duration):
    
    """
    Wave-generating function (with tapering)
    ::param f: frequency of wave
    ::param fs: sampling frequency
    ::param a: amplitude of wave
    ::param stim_duration: duration of the wave, in samples
    """
    
    tapering = .1
    n = np.sin(np.linspace(0,0.5*np.pi,np.floor(stim_duration*tapering)))
    m = np.concatenate([n,np.ones((1-tapering*2)*stim_duration),n[::-1]])
    y = [a*np.sin(2*np.pi*f*(i/fs)) for i in np.arange(stim_duration)] * m
    return y

def ssa_tones(f, f_difference):
    assert f-f_difference>0
    tone1 = mywave(f,44100,2,0.1*44100)
    tone2 = mywave(f-f_difference,44100,2,0.1*44100)
    return tone1, tone2

def randomised_trials(n_tone1, n_tone2):
    """
    randomised binary trials
    ::param n_f1: number of first trial type
    ::param n_f2: number of second trial type
    ::return trials_shuffled: array of shuffled trials
    """
    
    trials_unshuffled = np.concatenate([np.ones(n_tone1),np.zeros(n_tone2)])
    trials_shuffled = np.random.choice(trials_unshuffled, size=len(trials_unshuffled),replace=False)
   
    return trials_shuffled
   
def create_waveform_trials(trials_shuffled, tone1, tone2, ISI_len):
    """
    creates final waveforms
    """
    
    ISI = np.zeros(ISI_len)
    waveforms = []                                    
    for trial in trials_shuffled:
        if trial == 1:
            waveforms = np.concatenate([waveforms, tone1, ISI])
        if trial == 0:
            waveforms = np.concatenate([waveforms, tone2, ISI])
    return waveforms

def replace_doubles(array):
    """
    find two neighbouring 1s and move the second one randomly
    repeat until no neighbouring 1s
    """
    
    indices= []
    for i,(x,y) in enumerate(zip(array,np.roll(array,-1))):
        if np.logical_and(x,y):
            indices.append(i+1)
            
    if len(indices) == 0:
        return array
    
    array[indices] = 0
    new_indices = np.random.choice(np.arange(len(array)),len(indices),replace=False)
    array[new_indices] = 1
    replace_doubles(array)
    
    return array

n_oddball_tone = 10 # number of total repetitions the oddball tone
n_adapting_tone = 90 # number of total repetitions of the adapting tone
oddball_freq = 500 # frequency of the oddball stimulus
f_difference = 40 # frequency difference of the non-oddball stimulus (lower)

tone1, tone2 = ssa_tones(oddball_freq, f_difference) #tone 1 is oddball
ISI=44100*0.5


trials_shuffled = randomised_trials(n_oddball_tone, n_adapting_tone)
trials_shuffled = replace_doubles(trials_shuffled)
waveforms = create_waveform_trials(trials_shuffled, tone1, tone2, ISI)
waveforms_reversed = create_waveform_trials(trials_shuffled, tone2, tone1, ISI)



# save variables to current directory
np.save('trial_order', trials_shuffled)
np.save('waveforms', waveforms)
np.save('waveforms_reversed', waveforms_reversed)

## Save as .wav file
scipy.io.wavfile.write('/Users/stephenlenzi/Desktop/sound.wav', 44100, np.int16(waveforms)*32767)


