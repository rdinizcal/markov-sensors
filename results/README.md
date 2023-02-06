# Clustering Results 

The final results are documented in `results/`

## First Cycle

Experimental set from 21/12

1. Extract all records from the input data (set-a, set-b and set-c) -- Downloaded from the physionet source

2. Obtain features for through k-means technique

```
    vital_signals = ['HR','Temp','SaO2','NIDiasABP','NISysABP']
    HR - heart rate
    Temp - temperature
    SaO2 - O2 saturation level
    NIDiasABP - Non-invasive disatolic blood pressure
    NISysABP - Non-invasive systolic blood pressure
```

3. Build Discrete-Time Markov Chain (DTMC) -- filter here samples that will be part of the dtmc (e.g. cardiacsurgeryunit, medicalicu)

## Second Cycle

Experimental set from 26/12

1. 8,000 records from set-a and set-b -- Downloaded from physionet

2. vital_signals = ['HR','Temp','SaO2','NIDiasABP','NISysABP']

3. hardcoded signal ranges

    - Heart rate
        state 0: [0,70]     -> 'high risk'
        state 1: [70,85]    -> 'medium risk'
        state 2: [85,97]    -> 'low risk'
        state 3: [97,115]   -> 'medium risk'
        state 4 : [115,300] -> 'high risk'

    - Temperature
        state 0: [0,32]     -> 'high risk'
        state 1: [32,36]    -> 'medium risk'
        state 2: [36,38]    -> 'low risk'
        state 3: [38,41]    -> 'medium risk'
        state 4: [41,100]   -> 'high risk'

    - Oxygen saturation (SaO2)
        state 0: [0,55]     -> 'high risk'
        state 1: [55,65]    -> 'medium risk'
        state 2: [65+]      -> 'low risk'

    - Distolic Arterial Blood Pressure
        state 0: [0,80]     -> 'low risk'
        state 1: [80,90]    -> 'medium risk'
        state 2: [90,300]   -> 'high risk'

    - Systolic Arterial Blood Pressure
        state 0: [0,120]    -> 'low risk'
        state 1: [120,140]  -> 'medium risk'
        state 2: [140,300]  -> 'high risk'

4. Patient profiles through attribute filtering

    - by age
        0-15:           children 
        16-29:          youth
        30-59:          adult
        60+:            old

    - by BMI (weight/height^2)
        [0,18.5]:       underweight
        [18.5,24.9]:    normal weight
        [25-29.9]:      overweight
        [30-34.9]:      obesity1
        [35-39.9]:      obesity2
        40+:            obesity3
    
    -by ICU
        Cardiac Surgery Unit
        Coronary Surgery Unit
        Medical ICU
        Surgical ICU

