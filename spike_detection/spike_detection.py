import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal

n_channels = 
n_samples = 
shape = (n_samples,n_channels)

data=np.memmap('/home/slenzi/probe/neuropixels/KiloSort_data/20160526_01/20160526_spikeglx_01_g0_t0.imec.ap.bin', shape = (n_samples,n_channels), dtype='int16')

def spike_detect(trace, threshold=2.5):
    """
    takes a single trace, returns a spike location mask
    ::param trace:
    ::param threshold: number of standard deviations
    ::return spike_mask:
    """
    
    trace = trace-scipy.median(trace) #median subraction
    trace_std = np.std(trace)

    above_threshold = (trace < -trace_std*threshold).astype(int) # all points below spike threshold
    threshold_bounds = np.diff(above_threshold) # find places where the spike mask changes value

    putative_event_starts = np.where(threshold_bounds>0)[0] # change from 0 -> 1 is a start
    putative_event_ends = np.where(threshold_bounds<0)[0] # change from 1 -> 0 is an end

    event_maxima = np.zeros(trace.shape[0])

    for start, end in zip(putative_event_starts, putative_event_ends):
        event = trace[start:end]
        minimum_val = min(event)
        event_max_loc = np.where(event==minimum_val)[0]
        event_maxima[start+event_max_loc] = 1

    return event_maxima

def remove_duplicates(trace, points,min_distance_tolerated=20,iterations=2):
    
    shifted_points = np.roll(points,-1)
    delete_list = []
    for i, (max_loc1, max_loc2) in enumerate(zip(points[:-1],shifted_points[:-1])):
        
        # if there is a very close event, take the most negative, delete other
        if max_loc2-max_loc1<min_distance_tolerated:
            if trace[max_loc2]>trace[max_loc1]:
                delete_list.append(i)
            else:
                delete_list.append(i+1)   
    
    points=np.delete(points,delete_list) #remove indices from list
    #for maximum in event_maxima:
    #print(event_maxima_loc-np.roll(event_max_loc,-1))
    #event_maxima[delete_list] = 0
    iterations-=1
    if iterations == 0:
        return points
        
    remove_duplicates(trace, points,min_distance_tolerated=min_distance_tolerated,iterations=iterations)
        
    return points 

def force_minimum(trace,points):
    
    """
    forces detected maxima to their local minimum (rightwards)
    """

    new_points = []
    for i,point in enumerate(points):
        point2 = point+1
        while trace[point]>trace[point2]:
            point=point2
            point2+=1
        new_points.append(point)
    return new_points  


def mask_spikes(traces):
    spikes_mask = np.zeros_like(traces)
    for i, trace in enumerate(traces.T):
        event_maxima = spike_detect(trace)
        points = remove_duplicates(trace, np.where(event_maxima==1)[0],40)
        new_points = force_minimum(trace, points)
        spikes_mask[new_points, i] = 1
    return spikes_mask


def raster_plot(event_mask):
    for i,trace in enumerate(event_mask.T):
        indices=np.where(trace==1)
        plt.eventplot(indices,lineoffsets=i)

        
#mask=mask_spikes(data[600000:800000,65:130])