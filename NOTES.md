# Research Notes

Here we analyze data from sleep monitor which is a sensor with capacitive readouts from left and right temples and a differential readout, accompanied by acclerometer readings. There is a bio-marker sensor suite available as well which we will use tp benchmark our analysis of sleep monitor.

We have primary data from 6subjects, 2 sessions each of overnight sleep, where the sleep monitor and bio-marker sensor suite is active.


Primarily we want to show::
- Heart rate and Respiratory rate detection using capacitive sensor. A validation study that shows that both the signals are present in the sleep monitor and that rates can be detected
- Analysis of Slow wave sleep and analysis of sleep harmonics
- Detection of sleep apnea events



## Ideas
- In rate detections, we want to see if k is a biomarker, so we go find peaks, plot peaks per minute


## Next Steps(for meetings and brainstroming)

- Validation of cardiac and resp rates with our data
  - accuracy metric for rate detection methods

- slow wave sleep analysis
    - how do we identify events that corespond to slow wave sleep
    - thorax signal correlates to the low freq magnitude events, can we validate that low magnitude thorax corresponds to increase of low freq signal in cap data
    -do low pass filtering

    questions:: 
    - can we detect events like apnea
    - access sleep anpnia event in data
    - sleep staging based rates

    hypothesis
    - slow wave sleep is conected to deep sleep (N2 N3), if that is goin well then REM follows, if its short then REM may not occur


    - Compare spectrogram to the SWS analyssis, see if harmonics are observed.

    ** Projection methods


    - sleep apnea::
    Flow: gives types on apnea
    effort1

    - signal mean and std dev have a relationship with apnea events

    - mean cap as an output signal, accelerometer as input dimenison 1 , and lfp ratio as input diemension 2,  and build a regression from acclero related events + freq related events to mena cap change, does this predict REM events?

    - thorax as an indicator instead of apnea
    - k factor analysis on cardiac data, also resp, vs sleep events relationship
    

- slow wave with thorax amplitude comparision


- CArd and resp freq matching
- PPG in SWS will have higher amps, 
  - in freq domain of PPG we have multiple peaks, 
  - in Cap 

when thorax amp goes down, breathing goes down, this happends during SWS?


slow movement of 
side question, what is SWS defintion

SWS definition: slope band of cap data (mean cap)
PSG SWS definition: 


shaun SWS
- MEan cap value
- initiation by 



### Slow wave sleep
**Ranked criterion**
- Mean Capacitance slowly changes**
- initiation can be detected from head movements**
- Thorax amplitude slow change
- Heart rates increases
- Respiratory rate increases
- Deviation of respiratory rates as computed by cap vs thorax
- Heart Rate Band, major freq in PPG goes up while in CAP it goes down.
- **delta band of EEG(high-> wsw (conventionally))**

 
- persistent ridges **




notes
- observe the PSG based heart rate and resp rate, seehow does this reflect in cap data
- plotting sws time frac vs age



- EEG vs ratio of 0.5 to delta
intuitively the ration should drop when eeg delta power increases( signal of SWS in EEG)



- SWS is from mechanical change of flow, but the EEG respone is not proportional 
- CSF flow fluctuation leads SWS and lymphatic clearance, hormonal changes will 


- Do a pass at rate detection variatio evenets , they relate to SWS?


- Separate thorax band 



07-30

- baseline to zero
- flow trend
- variance(cortical arousal) comapred to EEG f2
- and head movement angle f3

- CH and CLE-CRE correlation reporting


- EEG does not capture cortical arousal fully,

