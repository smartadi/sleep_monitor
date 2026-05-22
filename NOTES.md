# Research Notes

Here we analyze data from sleep monitor which is a sensor with capacitive readouts from left and right temples and a differential readout, accompanied by acclerometer readings. There is a bio-marker sensor suite available as well which we will use tp benchmark our analysis of sleep monitor.

We have primary data from 6subjects, 2 sessions each of overnight sleep, where the sleep monitor and bio-marker sensor suite is active.


Primarily we want to show::
- Heart rate and Respiratory rate detection using capacitive sensor. A validation study that shows that both the signals are present in the sleep monitor and that rates can be detected
- Analysis of Slow wave sleep and analysis of sleep harmonics
- Detection of sleep apnea events



## Ideas
- In rate detections, we want to see if k is a biomarker, so we go find peaks, plot peaks per minute


## Next Steps

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
    


