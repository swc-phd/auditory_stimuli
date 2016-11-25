
# coding: utf-8

# # Import packages and reset device 
# ---------------------------------------------------------------------------------------------------------------------------------------------
# In[32]:

import PyDAQmx as ni
from PyDAQmx import Task
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# In[33]:

ni.DAQmxResetDevice("Dev1")


# # Configuration
# ---------------------------------------------------------------------------------------------------------------------------------------------

# ## Analog output 
# ---------------------------------------------------------------------------------------------------------------------------------------------

# In[34]:

# Define the waveforms to be sent into the galvos:
def gen_waveform( x_freq, y_freq, num_AO_samples, AO_sample_rate, amplitude):
    t = np.arange(0,  num_AO_samples/AO_sample_rate, 1./AO_sample_rate)
    sign_x = amplitude * signal.sawtooth(2.0 * np.pi * x_freq * t)
    sign_y = amplitude * signal.sawtooth(2.0 * np.pi * y_freq * t)

    waveform = np.vstack((sign_x,sign_y))
    waveform = np.ascontiguousarray(waveform)
    
    return waveform, t


# In[35]:

# analog output parameters
num_AO_samples = 10000
AO_sample_rate = 20000.0
amplitude = 3 # This defines the range over which we sample 
x_freq = 2.0
y_freq = 200.0
written = ni.int32()

waveform,t=gen_waveform( x_freq, y_freq, num_AO_samples, AO_sample_rate, amplitude)


AO_task = Task()
AO_task.CreateAOVoltageChan("/Dev1/ao0:1","",-10.0,10.0,ni.DAQmx_Val_Volts,None)
#AO_task.StartTask()

# Specify Sample Clock Timing
AO_task.CfgSampClkTiming( "OnboardClock", AO_sample_rate, ni.DAQmx_Val_Rising, ni.DAQmx_Val_ContSamps, num_AO_samples)
#Trigger on the counter. PFI12 is the output of counter 0
AO_task.CfgDigEdgeStartTrig("/Dev1/PFI12",ni.DAQmx_Val_Rising)


# ## Analog input
# ---------------------------------------------------------------------------------------------------------------------------------------------

# In[36]:

# Define analog input task
AI_task = Task()
AI_task.CreateAIVoltageChan("/Dev1/ai1","", ni.DAQmx_Val_RSE,-10.0,10.0, ni.DAQmx_Val_Volts, None)


# In[37]:

# set analog input parameters
num_images = 1
num_AI_samples = num_AO_samples*num_images
AI_sample_rate = 20000
data = np.zeros((num_AI_samples,), dtype=np.float64)
read = ni.int32()
#AI_task.StopTask()
AI_task.CfgSampClkTiming("OnboardClock", AI_sample_rate, ni.DAQmx_Val_Rising, ni.DAQmx_Val_ContSamps, num_AI_samples )

AI_task.CfgDigEdgeStartTrig("/Dev1/PFI12",ni.DAQmx_Val_Rising)


# ## Shutter
# ---------------------------------------------------------------------------------------------------------------------------------------------

# In[38]:

# parameter
shutter_port = "/Dev1/ao1"

shutter_task = Task()
shutter_task.CreateAOVoltageChan("/Dev1/ao1","",-10.0,10.0,ni.DAQmx_Val_Volts,None)

#shutter_task.StartTask()


def shutter_close(shutter_port):
    value=0
    shutter_task.WriteAnalogScalarF64(1,10.0,value,None)
    
def shutter_open(shutter_port):
    value= 5
    shutter_task.WriteAnalogScalarF64(1,10.0,value,None)


# ## Trigger
# ---------------------------------------------------------------------------------------------------------------------------------------------

# In[39]:

triggerTask = Task()
triggerTask.CreateCOPulseChanFreq("/Dev1/ctr0","",ni.DAQmx_Val_Hz,ni.DAQmx_Val_Low,0.0,AI_sample_rate,0.5)
triggerTask.CfgImplicitTiming(ni.DAQmx_Val_FiniteSamps,1)


# # Run tasks
# ---------------------------------------------------------------------------------------------------------------------------------------------
# In[40]:

#Setup the data that will be output on the trigger
AO_task.WriteAnalogF64(num_AO_samples, 0, 10.0, ni.DAQmx_Val_GroupByChannel, waveform, ni.byref(written), None)


# In[41]:

AI_task.StartTask() #This will start on the trigger
AO_task.StartTask() #This will start on the trigger
triggerTask.StartTask() # Start counter and trigger tasks


# In[42]:

# now, retrieve data
AI_task.ReadAnalogF64(num_AI_samples, 10.0, ni.DAQmx_Val_GroupByChannel, data, num_AI_samples, ni.byref(read), None)


# In[43]:

# And clean up tasks: 
AO_task.StopTask()
AI_task.StopTask()
triggerTask.StopTask()


# In[44]:

plt.plot(data)
plt.show()


# In[ ]:



