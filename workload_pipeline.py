import copy
import numpy as np

from scipy.stats import linregress
from scipy.signal import welch, butter, filtfilt

from sklearn.linear_model import LinearRegression


def butterworth_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth filter to a given dataset.
    
    Parameters:
    data: array_like, The data to be filtered.
    lowcut: float, The low cutoff frequency in Hz.
    highcut: float, The high cutoff frequency in Hz.
    fs: int or float, The sampling frequency of the data.
    order: int, The order of the filter.
    
    Returns:
    y: array_like, The filtered data.
    """
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

def apply_butterworth_filter_to_eeg_data(eeg_data):
    # Apply Butterworth filter to the first 32 rows (assuming these are the EEG signals)
    filtered_data = butterworth_filter(eeg_data.T, lowcut=1, highcut=30, fs=250).T  # Replace fs with actual sampling rate if different
    filtered_Fp1 = butterworth_filter(eeg_data[0, :].T, lowcut=1, highcut=7, fs=250).T
    filtered_Fp2 = butterworth_filter(eeg_data[1, :].T, lowcut=1, highcut=7, fs=250).T

    filtered_data[0, :] = filtered_Fp1 # Regr-FPZ
    filtered_data[1, :] = filtered_Fp2
    return filtered_data

def moving_average(signal, window_size=65):
    """
    Apply a simple moving average filter to a signal, ensuring the output
    array is the same size as the input array by handling edges separately.

    Parameters:
    signal: array_like, The input signal to be filtered.
    window_size: int, The size of the moving average window.

    Returns:
    filtered_signal: array_like, The signal after applying the moving average filter,
                     with the same size as the input signal.
    """
    # Initialize the filtered signal array with zeros
    filtered_signal = np.zeros(len(signal))

    # Compute the cumulative sum of the signal
    cumsum = np.cumsum(np.insert(signal, 0, 0))

    # Compute the moving average for each point in the signal
    for i in range(len(signal)):
        # Determine the start and end of the window for each point
        start = i - window_size // 2
        end = i + window_size // 2 + 1

        # Adjust the start and end to handle edge cases
        if start < 0:
            start = 0
        if end > len(signal):
            end = len(signal)

        # Compute the moving average for the current window
        filtered_signal[i] = (cumsum[end] - cumsum[start]) / (end - start)

    return filtered_signal

def process_derivative(filtered_data, delta_t=0.004, window_size=65):
    """
    Processes the derivative of filtered data, normalizes it, squares the normalization,
    and then computes the moving average.
    
    Parameters:
    - filtered_data: numpy array of filtered data from which to compute the derivative.
    - delta_t: time step between data points, used in the derivative calculation.
    - window_size: size of the moving window for averaging.
    
    Returns:
    - averaged_derivative: moving average of the squared normalized derivative.
    """
    # Calculate the derivative
    derivative = np.diff(filtered_data) / delta_t
    derivative = np.append(derivative, derivative[-1])
    
    # Normalize the derivative
    mean = np.mean(derivative)
    std = np.std(derivative)
    normalized_derivative = (derivative - mean) / std
    
    # Square the normalized derivative
    normalized_derivative_2 = normalized_derivative ** 2
    
    # Compute the moving average of the squared normalized derivative
    print(normalized_derivative_2.shape)
    averaged_derivative = moving_average(normalized_derivative_2, window_size)
    print(averaged_derivative.shape)
    
    return averaged_derivative

def compute_regression_coefficients(filtered_data, reference_signals):
    """
    Compute the linear regression coefficients for each signal in filtered_data
    against the reference_signals.

    Parameters:
    filtered_data (np.ndarray): 2D array where each row represents a signal over time.
    reference_signals (list of np.ndarray): List of reference signals to regress against.

    Returns:
    np.ndarray: 2D array of regression coefficients.
    """
    all_coefficients = []  # Initialize an empty list
    model = LinearRegression(fit_intercept=True)

    # Assuming the reference signals are in the correct shape
    X = np.column_stack(reference_signals)

    for i in range(2, filtered_data.shape[0]):
        y = filtered_data[i, :]
        model.fit(X, y)
        coeffs = model.coef_
        all_coefficients.append(coeffs)  # Append new coefficients

    # Convert the list of arrays to a 2D NumPy array
    all_coefficients_array = np.array(all_coefficients)

    return all_coefficients_array

def remove_eye_artifacts(filtered_data, averaged_derivative_Fp1, averaged_derivative_Fp2, all_coefficients_array):
    """
    Remove artifacts from EEG data based on averaged derivatives and regression coefficients.

    Parameters:
    filtered_data (np.ndarray): The EEG data with shape (n_channels, n_samples).
    averaged_derivative_Fp1 (np.ndarray): The averaged derivative of Fp1.
    averaged_derivative_Fp2 (np.ndarray): The averaged derivative of Fp2.
    all_coefficients_array (np.ndarray): The regression coefficients with shape (n_channels-2, 2).

    Returns:
    np.ndarray: The EEG data with artifacts removed.
    """
    # Calculate the logic mask where the threshold is greater
    thresh_fpz = (averaged_derivative_Fp1 + averaged_derivative_Fp2) / 2
    boolean_threshold = thresh_fpz > 1

    # Deep copy the filtered_data to avoid modifying the original data
    no_artefact_data = copy.deepcopy(filtered_data)

    # Iterate through channels and remove data where the threshold condition is met
    for i in range(2, filtered_data.shape[0]):
        no_artefact_data[i, boolean_threshold] -= all_coefficients_array[i-2, 0] * filtered_Fp1[boolean_threshold]
        no_artefact_data[i, boolean_threshold] -= all_coefficients_array[i-2, 1] * filtered_Fp2[boolean_threshold]

    return no_artefact_data

def segment_eeg(signal, dt=0.004, epoch_length=2, shift=0.125):
    """
    Segment an EEG signal into epochs.

    Parameters:
    signal: ndarray, the EEG signal with rows as channels and columns as samples.
    dt: float, the sampling interval in seconds.
    epoch_length: float, the length of each epoch in seconds.
    shift: float, the shift between epochs in seconds.

    Returns:
    epochs: ndarray, segmented EEG signal with shape (n_epochs, n_channels, n_samples_per_epoch)
    """
    # Calculate the number of samples per epoch and the shift in samples
    samples_per_epoch = int(epoch_length / dt)
    shift_samples = int(shift / dt)
    
    # Initialize an empty list to store epochs
    epochs = []
    
    # Determine the total number of epochs that can be extracted
    total_samples = signal.shape[1]
    n_epochs = (total_samples - samples_per_epoch) // shift_samples + 1
    
    # Segment the signal into epochs
    for i in range(n_epochs):
        start = i * shift_samples
        end = start + samples_per_epoch
        if end > total_samples:
            break  # Stop if the next epoch would go beyond the signal length
        epoch = signal[:, start:end]
        epochs.append(epoch)
    
    # Convert list of epochs to a 3D numpy array (n_epochs, n_channels, n_samples_per_epoch)
    epochs = np.array(epochs)
    
    return epochs

def identify_exceeding_channels(epochs, threshold=100):
    """
    Identify channels within each epoch where the magnitude exceeds a specified threshold.

    Parameters:
    epochs: ndarray, the 3D array of epochs (n_epochs, n_channels, n_samples_per_epoch).
    threshold: float, the magnitude threshold.

    Returns:
    exceeding_channels: ndarray, a 2D boolean array indicating for each epoch and each channel
                        whether the magnitude exceeds the threshold.
    """
    # Initialize a 2D boolean array with dimensions (n_epochs, n_channels), all set to False
    n_epochs, n_channels, _ = epochs.shape
    exceeding_channels = np.zeros((n_epochs, n_channels), dtype=bool)
    
    # Iterate through each epoch and each channel
    for epoch_idx, epoch in enumerate(epochs):
        for channel_idx, channel in enumerate(epoch):
            # Check if any value in the current channel of the current epoch exceeds the threshold
            if np.any(np.abs(channel) > threshold):
                # Mark this channel in this epoch as True in the boolean array
                exceeding_channels[epoch_idx, channel_idx] = True
    
    return exceeding_channels

def check_slope_artifacts_per_channel(epochs, sampling_rate=250, slope_threshold=10):
    """
    Check for artifacts based on the slope of the trend in each epoch for each channel separately.

    Parameters:
    epochs: ndarray, the 3D array of epochs (n_epochs, n_channels, n_samples_per_epoch).
    sampling_rate: int, the sampling rate in Hz.
    slope_threshold: float, the threshold for the slope (in mV/s) above which an epoch's channel is marked as an artifact.

    Returns:
    artifacts: ndarray, a 2D boolean array indicating for each epoch and each channel whether it is considered an artifact.
    """
    n_epochs, n_channels, n_samples = epochs.shape
    dt = 1 / sampling_rate  # Time interval between samples
    time = np.arange(n_samples) * dt  # Time vector for regression

    # Initialize a 2D array to mark channels with high slope trends within each epoch
    artifacts = np.zeros((n_epochs, n_channels), dtype=bool)

    for i in range(n_epochs):
        for j in range(n_channels):
            # Perform linear regression to find the slope for each channel within each epoch
            slope, _, _, _, _ = linregress(time, epochs[i, j, :])
            if abs(slope) > slope_threshold:
                artifacts[i, j] = True

    return artifacts

def check_max_abs_diff_artifacts(epochs, diff_threshold=25):
    """
    Check for artifacts by calculating the difference between the maximum and minimum 
    absolute values within each epoch for each channel. An epoch is marked as an artifact 
    if this difference exceeds 25 mV.

    Parameters:
    epochs: ndarray, the 3D array of epochs (n_epochs, n_channels, n_samples_per_epoch).
    diff_threshold: float, the threshold for the difference (in mV) above which an epoch is considered an artifact.

    Returns:
    artifacts: ndarray, a 2D boolean array indicating for each epoch and each channel whether it is considered an artifact.
    """
    n_epochs, n_channels, _ = epochs.shape

    # Initialize an empty array for artifact flags
    artifacts = np.zeros((n_epochs, n_channels), dtype=bool)

    for i in range(1, n_epochs):
        for j in range(n_channels):
            # Find the maximum and minimum absolute values within each channel of each epoch
            max_abs_val = np.max(np.abs(epochs[i, j, :]))            
            max_abs_val_prev = np.max(np.abs(epochs[i-1, j, :])) 
            
            # Calculate the difference between the max and min absolute values
            max_abs_diff = abs(max_abs_val - max_abs_val_prev)

            # Mark as an artifact if the difference exceeds the threshold
            if max_abs_diff > diff_threshold:
                artifacts[i, j] = True

    return artifacts

def calculate_psd_epochs(epochs, sampling_rate=250):
    """
    Calculate the Power Spectral Density (PSD) for each EEG epoch using a Hanning window
    and output as a 3D array.

    Parameters:
    epochs: ndarray, the 3D array of epochs (n_epochs, n_channels, n_samples_per_epoch).
    sampling_rate: int, the sampling rate in Hz.

    Returns:
    psds: ndarray, 3D array containing PSD for each epoch, channel, and frequency (n_epochs, n_channels, n_frequencies).
    freqs: ndarray, the frequencies at which the PSD is computed.
    """
    n_epochs, n_channels, n_samples = epochs.shape

    # Temporarily calculate PSD for the first channel of the first epoch to determine the size
    _, temp_psd = welch(epochs[0, 0, :], fs=sampling_rate, window='hann', nperseg=n_samples)
    n_frequencies = len(temp_psd)

    # Initialize the 3D array for PSDs
    psds = np.zeros((n_epochs, n_channels, n_frequencies))

    # Calculate PSD for each epoch and channel
    for i, epoch in enumerate(epochs):
        for j, channel in enumerate(epoch):
            freqs, psd = welch(channel, fs=sampling_rate, window='hann', nperseg=n_samples)
            psds[i, j, :] = psd
    
    return psds, freqs

def find_max_psd_indices(psds, freqs, min_freq=7.5, max_freq=12.5):
    """
    Find the indices and frequencies of the largest PSD values within a specific frequency range for each PSD.

    Parameters:
    psds: ndarray, 3D array of PSDs (n_epochs, n_channels, n_frequencies).
    freqs: ndarray, 1D array of frequencies corresponding to the PSDs.
    min_freq: float, the minimum frequency of the range of interest.
    max_freq: float, the maximum frequency of the range of interest.

    Returns:
    max_psd_indices: ndarray, 2D array of indices of the largest PSD values within the range (n_epochs, n_channels).
    max_psd_freqs: ndarray, 2D array of frequencies of the largest PSD values within the range (n_epochs, n_channels).
    """
    # Find the indices of the frequency range
    range_indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]

    # Initialize arrays to store the indices and frequencies of the max PSD values
    n_epochs, n_channels, _ = psds.shape
    max_psd_indices = np.zeros((n_epochs, n_channels), dtype=int)
    max_psd_freqs = np.zeros((n_epochs, n_channels))

    # Iterate through each epoch and channel to find the max PSD value and its index within the specified frequency range
    for i in range(n_epochs):
        for j in range(n_channels):
            # Find the index of the max PSD value within the frequency range
            max_idx = range_indices[np.argmax(psds[i, j, range_indices])]
            max_psd_indices[i, j] = max_idx
            max_psd_freqs[i, j] = freqs[max_idx]

    return max_psd_indices, max_psd_freqs

def extract_frequency_band_psds(psds, iaf_ind, combined_artefacts, channel_names=None):
    if channel_names is None:
        channel_names = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8',
                         'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 
                         'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
                         'PO7', 'PO3', 'PO4', 'PO8', 'Oz']
    channel_to_idx = {channel_name: i for i, channel_name in enumerate(channel_names)}

    theta_channels = ['AF3', 'AF4', 'Fz', 'F3', 'F4']
    alpha_channels = ['Pz', 'P3', 'P4', 'PO3', 'PO4']

    theta_indices = [channel_to_idx[channel] for channel in theta_channels]
    alpha_indices = [channel_to_idx[channel] for channel in alpha_channels]

    theta_psds = psds[:, theta_indices, :]
    alpha_psds = psds[:, alpha_indices, :]

    theta_iaf = iaf_ind[:, theta_indices]
    alpha_iaf = iaf_ind[:, alpha_indices]

    # Assuming the frequency resolution of the PSD is 0.5 Hz
    theta_lower = theta_iaf - 6 * 2  # Convert to index offset assuming 0.5 Hz resolution
    theta_upper = theta_iaf - 2 * 2   # Convert to index offset assuming 0.5 Hz resolution

    alpha_lower = theta_iaf - 2 * 2   # Convert to index offset assuming 0.5 Hz resolution
    alpha_upper = theta_iaf + 2 * 2   # Convert to index offset assuming 0.5 Hz resolution

    theta_psds_only = np.zeros(theta_lower.shape)
    alpha_psds_only = np.zeros(alpha_lower.shape)

    for i in range(theta_psds_only.shape[0]):
        for j in range(theta_psds_only.shape[1]):
            lower_limit = int(theta_lower[i, j])
            upper_limit = int(theta_upper[i, j])
            theta_psds_only[i, j] = np.sum(psds[i, theta_indices[j], lower_limit:upper_limit])

    for i in range(alpha_psds_only.shape[0]):
        for j in range(alpha_psds_only.shape[1]):
            lower_limit = int(alpha_lower[i, j])
            upper_limit = int(alpha_upper[i, j])
            alpha_psds_only[i, j] = np.sum(psds[i, alpha_indices[j], lower_limit:upper_limit])

    theta_artefacts = combined_artefacts[:, theta_indices]
    alpha_artefacts = combined_artefacts[:, alpha_indices]

    return theta_psds_only, alpha_psds_only, theta_artefacts, alpha_artefacts

def feature_extraction_pipeline(eeg_data):
    filtered_data = apply_butterworth_filter_to_eeg_data(eeg_data)

    filtered_Fp1 = filtered_data[0, :]
    filtered_Fp2 = filtered_data[1, :]

    averaged_derivative_Fp1 = process_derivative(filtered_Fp1)
    averaged_derivative_Fp2 = process_derivative(filtered_Fp1)

    # Signal that will be used to remove eye artefacs
    reference_signal = np.array([filtered_Fp1, filtered_Fp2]).T
    regression_coefficients = compute_regression_coefficients(filtered_data, reference_signal)
    no_artefact_data = remove_eye_artifacts(filtered_data, 
                                            averaged_derivative_Fp1, 
                                            averaged_derivative_Fp2, 
                                            regression_coefficients)
    
    epochs = segment_eeg(no_artefact_data)
    artefacts_tresh = identify_exceeding_channels(epochs)
    artefacts_trend = check_slope_artifacts_per_channel(epochs)
    artefacts_sts = check_max_abs_diff_artifacts(epochs)
    combined_artefacts = artefacts_tresh | artefacts_trend | artefacts_sts

    psds, freqs = calculate_psd_epochs(epochs)
    iaf_ind, iaf_freq = find_max_psd_indices(psds, freqs)
    theta_psds_only, alpha_psds_only, theta_artefacts, alpha_artefacts = extract_frequency_band_psds(psds, 
                                                                                                 iaf_ind, 
                                                                                                 combined_artefacts)
    return theta_psds_only, alpha_psds_only, theta_artefacts, alpha_artefacts
