#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:07:15 2020
@author: Fjola Hyseni & Marius Keute
"""
import matplotlib.pyplot as plt
import numpy as np
import biosppy
from biosppy import tools as st
from scipy import signal
import pywt


class ecg_analyzer:
    def __init__(self, ECG, fs=1000):
        """Parameters:
        *************
        ECG: 1d-array of raw ECG data
        fs: sampling frequency in Hz
        
        Attributes:
        ************
        phases: ECG phase estimation
        HR: instantaneous heart rate
        HRV_time_domain: different time domain HRV scores
        HRV_frequency_domain: different frequency domain HRV scores
        SD1SD2: Nonlinear HRV score generated from a Poincare recurrence matrix
        """
        self.ECG = np.squeeze(ECG)
        self.fs = fs
        self.exclude, flip = artifact_removal(self.ECG, fs = fs)
        flip = decide_if_standard_orientation(self.ECG)
        if flip:#len(flip) > 0:
            self.ECG *= -1
            
            
        # plt.figure()
        # plt.plot(self.ECG)
        if np.sum(self.exclude) > 0:

            self.ECG = np.interp(np.arange(len(self.ECG)), np.where(np.invert(self.exclude.astype(bool)))[0], self.ECG[np.invert(self.exclude.astype(bool))])

        # plt.plot(self.ECG)
        
        self.ECG, r_peaks = get_signal_and_peaks(self.ECG, method = 'wavelet', fs = fs)
        
        self.r_peaks = np.delete(r_peaks, np.where((self.ECG[r_peaks] < np.percentile(self.ECG, 95)) & (self.ECG[r_peaks] > np.percentile(self.ECG, 5)))[0])
        
        self.NNi, self.HR = get_instantaneous_HR(self.r_peaks, len(self.ECG), fs)
        self.corRSA = get_corRSA(self.NNi, self.r_peaks, self.ECG)
        
        # plt.plot(self.r_peaks, self.ECG[self.r_peaks], 'or')

        self.categorical_phases, self.analytical_phases = get_ECG_phases(self.ECG, self.r_peaks)
        
        if np.sum(self.exclude) > 0:
            from scipy.interpolate import pchip
            Intp = pchip(np.where(np.invert(self.exclude.astype(bool)))[0], self.HR[np.invert(self.exclude.astype(bool))])
            self.HR = Intp(np.arange(len(self.HR)))
            
            
            # self.HR[self.exclude.astype(bool)] = np.nan
            self.categorical_phases[self.exclude.astype(bool)] = np.nan
            self.analytical_phases[self.exclude.astype(bool)] = np.nan
            
        self.HRV_time_domain = HRV_time_domain(self.NNi)
        self.HRV_frequency_domain = HRV_frequency_domain(self.HR[np.invert(np.isnan(self.HR))], self.fs)
        self.SD1SD2, self.logRSA = HRV_Poincare(self.NNi)
    def get_sliding_HRV(self, window_length = 5000, overlap = 2500):
        """ Calculate time-domain HRV indices in sliding windows
            Returns:
            ************
            sliding_HRV_t: Time-domain HRV indices for all windows
            sliding_HRV_f: Frequency-domain HRV indices for all windows
            timestamps: List of timestamps, giving the center of each window in
            seconds relative to the beginning of the ECG signal.
            Results will not be saved as class attributes, bust must be assigned
            to a new variable.
        """
        startix = 0
        dt = overlap / self.fs
        ts = window_length/(2*self.fs)
        timestamps = []
        sliding_HRV_t = {}
        sliding_HRV_f = {}

        while (startix + window_length) < max(self.r_peaks):
            r_peaks_in_window = np.where((startix < self.r_peaks) & (self.r_peaks < (startix + window_length)))
            nni_in_window = self.NNi[r_peaks_in_window]
            tmp = HRV_time_domain(nni_in_window)
            if len(sliding_HRV_t) == 0:
                sliding_HRV_t = {key:[] for key in tmp.keys()}
            [sliding_HRV_t[key].append(tmp[key]) for key in tmp.keys()]
            
            tmp = HRV_frequency_domain(self.HR[startix:startix+window_length], self.fs)
            if len(sliding_HRV_f) == 0:
                sliding_HRV_f = {key:[] for key in tmp.keys()}
            [sliding_HRV_f[key].append(tmp[key]) for key in tmp.keys()]
            
            
            
            timestamps.append(ts)
            ts += dt
            startix += overlap
        
        return sliding_HRV_t, sliding_HRV_f, timestamps
    
    
#%%


def get_corRSA(NNi, r_peaks, ECG):
    rpkamp = np.zeros(len(NNi))
    for n in range(len(NNi)):
        ix0 = max(0,r_peaks[n]-50)
        ix1 = min(len(ECG)-1, r_peaks[n]+50)
        rpkamp[n] = max(ECG[ix0:ix1])
    return np.corrcoef(NNi, rpkamp)[0,1]

def artifact_removal(ECG_data, fs = 1000):
        bandpass = signal.butter(4,(1,45),btype = 'pass',fs=fs)
        raw_p_peaks = _positive_peaks(signal.filtfilt(*bandpass,ECG_data), sampling_rate = fs, method = 'wavelet')
        raw_n_peaks = _positive_peaks(signal.filtfilt(*bandpass,- ECG_data), sampling_rate = fs, method = 'wavelet')

        # plt.plot(ECG_data)
        # plt.plot(raw_p_peaks, ECG_data[raw_p_peaks], 'or')
        # raw_n_peaks = _positive_peaks(-ECG_data, method = 'hamilton')
        r_peak = []
        r_p = []
        r_peaksepoch = []
        len_epoch = 10 * fs
        epoch_nr = int(np.ceil(len(ECG_data)/len_epoch))
        exclude = np.zeros(len(ECG_data))
        flip_orientation = np.zeros(len(ECG_data))
        min_raw_p_peak = raw_p_peaks[0]
        max_raw_p_peak = raw_p_peaks[-1]
        for i in range(epoch_nr):
            ECG_epoch = ECG_data[i*len_epoch: (i+1)*len_epoch]
            try:    
                min_raw_p_peak = _find_closest_in_list(i*len_epoch, raw_p_peaks, direction= "greater", strictly = True)
                max_raw_p_peak = _find_closest_in_list((i+1)*len_epoch, raw_p_peaks, direction= "smaller", strictly = True)
            except ValueError:
                pass
            min_index = int(np.where(raw_p_peaks == min_raw_p_peak)[0])
            max_index = int(np.where(raw_p_peaks == max_raw_p_peak)[0])
            epoch_nni = np.diff(raw_p_peaks[min_index: max_index + 1])
            if min_raw_p_peak > (i+1)*len_epoch:
                exclude[i*len_epoch: (i+1)*len_epoch] = 1
            elif len(epoch_nni) < 7:
                exclude[i*len_epoch: (i+1)*len_epoch] = 1
            elif np.min(epoch_nni) < 400 or np.max(epoch_nni)> 1400:
                exclude[i*len_epoch: (i+1)*len_epoch] = 1
            elif len(ECG_epoch) < 903:
                print("Length of", i+1,"th epoch is smaller than 903; this does not allow R peak detection to occur at this epoch.")
            else:
            # try: 
                standard_orientation = decide_if_standard_orientation(ECG_epoch) 
                if standard_orientation == False:
                    flip_orientation[i*len_epoch: (i+1)*len_epoch] = 1
                    r_peaksepoch = raw_n_peaks[min_index: max_index + 1]
                else:
                    ECG_data[i*len_epoch: (i+1)*len_epoch] = ECG_data[i*len_epoch: (i+1)*len_epoch]
                    r_peaksepoch = raw_p_peaks[min_index: max_index + 1]
   
            r_p.append(r_peaksepoch)
        r_peak = np.concatenate(r_p)
        r_peaks = [] 
        for i in r_peak: 
            if i not in r_peaks: 
                r_peaks.append(i) 
        r_peaks = np.array(r_peaks)
        r_peaks = r_peaks.astype(int)

        threshix = np.where((ECG_data > 3 * np.percentile(ECG_data, 99)) | (ECG_data < 3 * np.percentile(ECG_data, 1)))[0]
        if len(threshix) > 0:
            for marginval in np.arange(-250,250):
                ix = threshix + marginval
                
                exclude[np.delete(ix, np.where(ix >= len(exclude))[0])] = 1

        flipix = np.where(flip_orientation)[0]
        return exclude, flipix
    


def HRV_Poincare(NNi):
    #returns SD1/SD2, i.e. the variance ratio of the first two principal
    #components of the recurrence matrix (RR_i vs RR_i+1). logRSA is an
    #estimation of respiratory sinus arrythmia calculated from the recurrence
    #matrix
#    logRSA description in:
#        Moser, M., Lehofer, M., Sedminek, A., Lux, M., Zapotoczky, H. G., 
#        Kenner, T., et al. (1994). Heart rate variability as a prognostic 
#        tool in cardiology. a contribution to the problem from a theoretical 
#        point of view. Circulation 90, 1078–1082. doi: 10.1161/01.cir.90.2.1078
#    
    
    
    from sklearn.decomposition import PCA
    recur = np.array([NNi[1:],NNi[:-1]])        
    p=PCA()
    comps = p.fit_transform(recur.T)
    SD1,SD2 = np.std(comps, axis = 0)
    logRSA = np.log10(np.median(np.abs(np.diff(recur, axis = 0))))
    return SD1/SD2, logRSA

def HRV_frequency_domain(HR, fs):
    """Calculates frequency-domain HRV indices.
    Frequency band boundaries taken from:
    Shaffer, F., & Ginsberg, J. P. (2017). 
    An overview of heart rate variability metrics and norms. 
    Frontiers in public health, 5, 258.
    Returns nan for a given frequency band
    if the signal is too short to calculate the PSD.
    """
    # from scipy.fftpack import fft
    # N = len(HR)
    # spc = fft(HR)
    # spc = spc[:int(N/2 +1)]
    
    # psd = (1/(fs*N)) * np.abs(spc)**2
    # psd[1:-1] *= 2
    # frx = np.linspace(0,fs/2, num = int(len(psd)));
    
    
    from scipy import signal, integrate
    # NNi = np.interp(x = range(total_signal_length),xp=r_peaks[:-1], fp=BPM)

    frx,psd = signal.welch(60000/HR,fs =fs, nperseg = int(25*fs))
    dx = np.diff(frx)[0]
    
    def nearest(array, value):
        return np.argmin(np.abs(array - value))
    ULF,VLF,LF,HF,LFHF = np.nan,np.nan,np.nan,np.nan,np.nan
    #ULF and VLF are discarded because we mostly work on short data segments
    # if len(HR)/fs > 334:
    #     ULF = integrate.simps(psd[1:nearest(frx, .003)],dx=dx)
    # if len(HR)/fs > 303:
    #     VLF = integrate.simps(psd[nearest(frx, .0033):nearest(frx, .04)],dx=dx)
    if len(HR)/fs > 7:
        HF = integrate.simps(psd[nearest(frx, .15):nearest(frx, .4)],dx=dx)
    if len(HR)/fs > 25:
        LF = integrate.simps(psd[nearest(frx, .04):nearest(frx, .15)],dx=dx)
        LFHF = LF/HF
    
    return {'ULF':ULF,'VLF':VLF,'LF':LF,'HF':HF,'LFHF':LFHF}

def HRV_time_domain(nni):
    """Calculate time-domain indices from an RRi series
    Parameters
    ----------
    nni : array_like
        sequence containing the NNi series
    Returns
    -------
    results : dict
        Dictionary containing the following time domain indices:
            - RMSSD: root mean squared of the successive differences
            - SDNN: standard deviation of the RRi series
            - NN50: number RRi successive differences greater than 50ms
            - PNN50: percentage of RRi successive differences greater than 50ms
            - MRI: average value of the RRi series
    """

    diff_nni = np.diff(nni)
    rmssd = np.sqrt(np.mean(diff_nni ** 2))
    sdnn = np.std(nni, ddof=1)  # make it calculates N-1
    nn50 = sum(abs(diff_nni) > .050)
    pnn50 = (nn50 / len(nni) * 100)


    
    return dict(zip(['rmssd', 'sdnn', 'nn50', 'pnn50'], [rmssd, sdnn, nn50, pnn50]))


# def interp_NNi_and_HR(NNi, HR, exclude):
    
    
    
    
        
def get_instantaneous_HR(r_peaks, total_signal_length, fs):
    NNi = np.diff(r_peaks)/fs
    delix = np.where(NNi > 1.4)[0]
    NNi=np.delete(NNi,delix)
    r_peaks=np.delete(r_peaks,delix)
    BPM = 60 / NNi
    instantaneous_HR = np.interp(x = range(total_signal_length),xp=r_peaks[:-1], fp=BPM)
    return NNi, instantaneous_HR

def get_ECG_phases(ECG, r_peaks):
    q_peaks, p_peaks, p_start, p_end = ecg_wave_detector_pq(ECG, r_peaks)
    s_peaks, t_peaks, t_start, t_end = ecg_wave_detector_st(ECG, r_peaks)
    categorical_phases = _masks(ECG, r_peaks, p_start, p_end, t_start, t_end)
    
    analytical_phases = np.nan * np.zeros(len(ECG))
    
    for ix in range(len(r_peaks)-1):
        analytical_phases[r_peaks[ix]:r_peaks[ix+1]] = np.linspace(0,2*np.pi, num = r_peaks[ix+1]-r_peaks[ix])    
        
        
    return categorical_phases, analytical_phases

def get_signal_and_peaks(ECG, fs, method = 'wavelet'):
    """ This function orients the signal and defines R peak indices accordingly.
    Parameters
    ----------
    signal : array
        raw_ECG_data.
    standard_orientation : bool_
        True or False.
    raw_p_peaks : array
        Indices of the positive peaks extracted from the raw data.
    raw_n_peaks : array
        Indices of the negative peaks extracted from the raw data.
    
    Returns
    -------
    ECG_data : array
        Standardly oriented ECG data.
    r_peaks : array
        Indices of the R peaks.
    """
    bandpass = signal.butter(4,(3, 45),btype = 'pass', fs = fs)
    ECG = signal.filtfilt(*bandpass, ECG)
    # standard_orientation = decide_if_standard_orientation(ECG, fs)
    # if standard_orientation == False:
    #     ECG *= -1

        
    r_peaks = _positive_peaks(ECG, sampling_rate = fs, method = method)
    
    return ECG, r_peaks


def _positive_peaks(raw_ECG_data, sampling_rate=1000, method = 'wavelet'):
    """Process a raw ECG signal and extracts R peaks.    
    Parameters
    ----------
    signal : array
        Raw ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    method: 'wavelet' or 'hamilton'. Hamilton will find R-peaks based on the 
        iterative hamilton segmenter method. Wavelet find find r-peaks based
        on convolution of the signal with a qrs-complex-shaped wavelet.
    Returns
    -------
    positive_peaks : array
        Positive-peak location indices. 
    """
    
    if method == 'hamilton':
        order = int(0.3 * sampling_rate)
        filtered, _, _ = st.filter_signal(
            raw_ECG_data, ftype='FIR', band='bandpass', order=order, frequency=[3, 45], sampling_rate=sampling_rate)
        positive_peaks, = biosppy.signals.ecg.hamilton_segmenter(
            filtered, sampling_rate=1000.0)
        positive_peaks, = biosppy.signals.ecg.correct_rpeaks(
            signal=filtered, rpeaks=positive_peaks, sampling_rate=1000, tol=0.05)
        # plt.plot(raw_ECG_data)
        # plt.plot(positive_peaks, raw_ECG_data[positive_peaks], 'or')
    elif method == 'wavelet':
        wv = pywt.Wavelet('sym4')
        _,qrs,_ = wv.wavefun(level = 5)
        cv = signal.fftconvolve(raw_ECG_data, qrs, mode = 'same')
        positive_peaks = signal.find_peaks(np.abs(cv), distance = int(sampling_rate/2), prominence = 200)[0]
        # plt.figure()
        # plt.plot(cv)
        # plt.plot(raw_ECG_data)
        # plt.plot(positive_peaks, cv[positive_peaks], 'or')
    else:
        raise ValueError('no valid method selected')
       
    return positive_peaks



def decide_if_standard_orientation(raw_ECG_data, fs = 1000, debug:bool = False) -> bool:
    """Returns a bool, if the signal is in the standard ECG orientation
    Parameters
     ----------
     ECG_data
    Returns
    -------
    bool: 
        True for success, False otherwise, Error message if undecidable
    """
    data_len = len(raw_ECG_data)
    analysis_length = 2000
    n_bins = int(data_len/analysis_length)
    if n_bins == 0:
        raise IndexError("The ECG data is shorter than 2 seconds!")

    orientations = []
    for i in range(0,n_bins):
        raw_p_peaks = _positive_peaks(raw_ECG_data[i*analysis_length:(i+1)*analysis_length])
        raw_n_peaks = _positive_peaks(-raw_ECG_data[i*analysis_length:(i+1)*analysis_length])

        for i in range(0, len(raw_n_peaks)-1):
            if debug:
                print(f"P-peak: {raw_p_peaks[i]}")
                print(f"N-peak: {raw_n_peaks[0]}\n")
            try:
                samples_between_peaks = raw_n_peaks[i] - raw_p_peaks[i]
                if samples_between_peaks > 200:
                    """Cut after negative peak"""
                    orientations.append(raw_n_peaks[i] > raw_p_peaks[i+1])
                elif samples_between_peaks < -200:
                    """Cut after positive peak"""
                    orientations.append(raw_n_peaks[i+1] > raw_p_peaks[i])
                else:
                    orientations.append(raw_n_peaks[i] > raw_p_peaks[i])
            except IndexError:
                continue
            if len(orientations) > 5:
                break
    if debug:
        print(f"Orientations: {orientations}")
    if len(orientations) > 0:
        return 0.5 < np.mean(orientations)
    else:
        return "Data was impossible to analyse"




def _find_closest_in_list(number, array, direction="both", strictly=False):
    """Find the closest number in the array from x.
    Parameters
    ----------
    number : float
        The number.
    array : array
        The array to look into.
    direction : string
        "both" for smaller or greater, "greater" for only greater numbers and "smaller" for the closest smaller.
    strictly : bool
        False for stricly superior or inferior or True for including equal. The default is False.
    Returns
    -------
    closest : int
        The closest number in the array.
    """
    if direction == "both":
        closest = min(array, key=lambda x: abs(x-number))
    if direction == "smaller":
        if strictly is True:
            closest = max(x for x in array if x < number)
        else:
            closest = max(x for x in array if x <= number)
    if direction == "greater":
        if strictly is True:
            closest = min(filter(lambda x: x > number, array))
        else:
            closest = min(filter(lambda x: x >= number, array))

    return(closest)


def _find_peaks(signal):
    """Locate peaks based on the derivative of the graph.
    Parameters
    ----------
    signal : array
        ECG signal.
    Returns
    -------
    peaks : array
        An array containing the peak indices.
    """
    derivative = np.gradient(signal, 2)
    peaks = np.where(np.diff(np.sign(derivative)))
    return(peaks)


def ecg_wave_detector_pq(signal, r_peaks):
    """Returns the localization of the P and Q waves. 
    Note: This function determines the peaks based on RR interval. 
    Thus, if the data starts after a R peak (for instance with a T wave), 
    the function will not be able to detect the first P wave and Q peak.
    Parameters
    ----------
    signal : array
        ECG signal.
    r_peaks : array
       R peak indication indices.
    Returns
    -------
    q_peaks : array
        R peak indication indices.
    p_peaks : array
        P peak indication indices.
    p_start : array
        P wave onsetindices.
    p_end : array
        P wave end indices.
    """
    p_peaks = []
    p_s = 0
    p_e = 0
    sampling_rate = 1000
    order = int(0.3 * sampling_rate)
    for index, rpeak in enumerate(r_peaks[:-1]):
        middle = (r_peaks[index+1] - rpeak) / 2
        quarter = int(middle*1/2)
        tquarter = middle*3/2
        eighth = middle*7/4
        epoch = signal[int(rpeak+tquarter):int(rpeak+eighth)]
        try:
            p_peak = int(rpeak+tquarter) + np.argmax(epoch)
            p_peaks.append(p_peak)
        except ValueError:
            p_peak = int(rpeak+tquarter) + int(eighth/7)
            p_peaks.append(p_peak)
    p_peaks = np.array(p_peaks)
       
    q_peaks = []
    for index, p_peak in enumerate(p_peaks):
        epoch = signal[int(p_peak):int(r_peaks[r_peaks > p_peak][0])]
        try:
            q_peak = p_peak + np.argmin(epoch)
            q_peaks.append(q_peak)
        except ValueError:
            pass
             
    p_start = np.zeros(len(p_peaks))
    p_end = np.zeros(len(p_peaks))
    r_peaks = r_peaks.astype(int)

    for i in range(len(p_peaks)):
        third = int((r_peaks[i+1]- r_peaks[i])/3)
        h = np.histogram(signal[r_peaks[i+1] - third:p_peaks[i]], bins=30)
        y = np.argmax(h[0])
        meanizo = h[1][y]

        p_e = np.argmin(abs(signal[p_peaks[i]-1: q_peaks[i]] - meanizo))
        
        for k in range(100):
            start_val = abs(signal[p_peaks[i]-k] - meanizo)
            if start_val <= 8:
                p_s = p_peaks[i]-k
                break

        p_start[i] =  p_s
        p_end[i] = p_peaks[i] + p_e
    p_start = p_start.astype(int)
    p_end = p_end.astype(int)
    return (q_peaks, p_peaks, p_start, p_end)

def ecg_wave_detector_st(signal, r_peaks):
    """Returns the localization of the S and T waves. 
    Note: This function determines the peaks based on RR interval. 
    Thus, if the data starts after a R peak (for instance with a S peak), 
    the function will not be able to detect the first T wave and S peak.
    Parameters
    ----------
    signal : array
        ECG signal.
    r_peaks : array
       R peak indication indices.
    Returns
    -------
    s_peaks : array
        S peak indication indices.
    t_peaks : array
        T peak indication indices.
    t_start : array
        T wave onsetindices.
    t_end : array
        T wave end indices.
    """
    s_peaks = []
    t_peaks = []
    t_start = []
    t_end = []
    for index, rpeak in enumerate(r_peaks[:-1]):
        middle = (r_peaks[index+1] - rpeak) / 2
        epoch_after = signal[int(rpeak):int(rpeak+middle)]
        
        s_peak_index = np.argmin(epoch_after)
        s_peak = rpeak + s_peak_index
        t_peak_index = s_peak_index + np.argmax(epoch_after[s_peak_index:])
        t_peak = rpeak + t_peak_index
        t_peaks.append(t_peak)
        s_peaks.append(s_peak)
        try:
            inter_st = epoch_after[s_peak_index:t_peak_index]
            inter_st_derivative = np.gradient(inter_st, 2)
            t_wave_start_index = _find_closest_in_list(
                len(inter_st_derivative)/2, _find_peaks(inter_st_derivative)[0])
            t_wave_start = s_peak + t_wave_start_index
            t_wave_end = np.argmin(epoch_after[t_peak_index:])
            t_wave_end = t_peak + t_wave_end
    
            t_start.append(t_wave_start)
            t_end.append(t_wave_end) 

        except ValueError:
            t_wave_start = s_peak
            t_wave_end = np.argmin(epoch_after[t_peak_index:])
            t_wave_end = t_peak + t_wave_end
    
            t_start.append(t_wave_start)
            t_end.append(t_wave_end)
    t_start = np.array(t_start)
    t_end = np.array(t_end)
    return (s_peaks, t_peaks, t_start, t_end)

def _masks(signal,r_peaks, p_start, p_end, t_start, t_end):
    """
    This function serves to create a mask to be able to define the intervals of the phases.
    Parameters
    ----------
    signal : array
        ECG data signal.
    r_peaks: array
        R peak indices.
    p_start : array
        P start indication indices.
    p_end : array
        P end indication indices.
    t_start : array
        T start indication indices.
    t_end : array
        T end indication indices.
    Returns
    -------
    phases: array
        An array of 1, 2, 3,4 that where each of the numbers is used as a mask for a specific phase.
            1- P phase
            2- QRS phase
            3- T phase
            4- TP phase
   """
    phases = np.zeros(len(signal))
    nni = np.diff(r_peaks) 
    ppi =[]
    tti =[]
    qrsi = []
    tpi = []
    zzi = []
    
    for i in range(len(r_peaks)-1): 
        # if nni[i] > 400 and nni[i]  < 1400:
        ppl = p_end[i]-p_start[i]
        if ppl < 400 and ppl > 0:
            phases[p_start[i]: p_end[i]] = 1
            ppi.append(ppl)
        ttl = t_end[i]-t_start[i]
        if ttl < 700 and ttl > 0:
            phases[t_start[i]: t_end[i]] = 3
            tti.append(ttl)
        tpl = p_start[i]-t_end[i]
        if tpl < 1400  and tpl >0:
            phases[t_end[i]: p_start[i]] = 4 
            tpi.append(tpl)
    for i in range(len(r_peaks)-2): 
        # if nni[i] > 400 and nni[i]  < 1400 and nni[i+1] > 400 and nni[i+1]  < 1400:
        qrsl = t_start[i+1] - p_end[i]
        if qrsl < 500 and qrsl > 0:
            phases[p_end[i]: t_start[i+1]] = 2
            qrsi.append(qrsl)
    return phases

    


#%%
if __name__ == "__main__":
    ecg = np.load('/Users/fjola/Desktop/LaRe.npy')
#     import pickle
#     with open('/home/marius/Downloads/exampleECG.p', 'rb') as p:
#         ecg = pickle.load(p)
    #ecg= ECG_data
    a=ecg_analyzer(ecg)
