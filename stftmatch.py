import segyio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math # Needed for log2 and ceil

# --- Core Helper Functions (load, conform, save, plot) remain the same ---
def load_segy_data(filepath):
    """Loads seismic data and sampling interval from a SEG-Y file."""
    try:
        with segyio.open(filepath, ignore_geometry=True) as f:
            data = f.trace.raw[:].T
            dt = f.bin[segyio.BinField.Interval] * 1e-6
            print(f"Successfully loaded {filepath}. Shape: {data.shape}, dt: {dt*1000:.2f} ms")
            return data, dt
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

def conform_reference_to_source(ref_data, ref_dt, src_data, src_dt):
    """Conforms the reference data to the dimensions and sampling of the source data."""
    # (This function is unchanged from the previous version)
    print("\n--- Conforming reference section to source dimensions ---")
    conformed_ref = ref_data.copy()
    src_n_samples, src_n_traces = src_data.shape
    ref_n_samples, ref_n_traces = ref_data.shape

    if ref_n_traces > src_n_traces:
        start = (ref_n_traces - src_n_traces) // 2
        conformed_ref = conformed_ref[:, start:start + src_n_traces]
    elif ref_n_traces < src_n_traces:
        pad_left = (src_n_traces - ref_n_traces) // 2
        pad_right = src_n_traces - ref_n_traces - pad_left
        conformed_ref = np.pad(conformed_ref, ((0, 0), (pad_left, pad_right)), mode='edge')

    if abs(ref_dt - src_dt) > 1e-9:
        original_time_duration = conformed_ref.shape[0] * ref_dt
        new_n_samples = int(round(original_time_duration / src_dt))
        conformed_ref = signal.resample(conformed_ref, new_n_samples, axis=0)

    current_n_samples = conformed_ref.shape[0]
    if current_n_samples > src_n_samples:
        conformed_ref = conformed_ref[:src_n_samples, :]
    elif current_n_samples < src_n_samples:
        pad_amount = src_n_samples - current_n_samples
        conformed_ref = np.pad(conformed_ref, ((0, pad_amount), (0, 0)), mode='edge')
    
    print(f"Conforming complete. New reference shape: {conformed_ref.shape}\n")
    return conformed_ref

# --- NEW AUTOMATION FUNCTION ---
def _next_power_of_2(x):
    """Calculates the next power of 2 greater than or equal to x."""
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

def estimate_optimal_nperseg(data, dt, period_multiplier=3, f_min=5.0):
    """
    Estimates an optimal nperseg for STFT based on the data's dominant frequency.

    Args:
        data (np.ndarray): The source seismic data.
        dt (float): The sampling interval in seconds.
        period_multiplier (int): The factor to multiply the dominant period by.
        f_min (float): The minimum frequency to consider for the dominant frequency.

    Returns:
        int: The estimated optimal nperseg value.
    """
    print("\n--- Automating STFT window length (nperseg) estimation ---")
    n_samples = data.shape[0]
    
    # 1. Calculate average spectrum
    fft_all = np.fft.fft(data, axis=0)
    avg_amp_spec = np.mean(np.abs(fft_all), axis=1)
    freqs = np.fft.fftfreq(n_samples, d=dt)
    
    # Consider only positive frequencies
    positive_freq_mask = freqs > 0
    freqs = freqs[positive_freq_mask]
    avg_amp_spec = avg_amp_spec[positive_freq_mask]
    
    # 2. Find dominant frequency above f_min
    freq_min_mask = freqs >= f_min
    if not np.any(freq_min_mask):
        print(f"Warning: No frequencies found above f_min={f_min} Hz. Using default nperseg=64.")
        return 64
        
    spec_to_search = avg_amp_spec[freq_min_mask]
    freqs_to_search = freqs[freq_min_mask]
    
    dominant_freq_index = np.argmax(spec_to_search)
    dominant_freq = freqs_to_search[dominant_freq_index]
    print(f"Found dominant frequency: {dominant_freq:.2f} Hz")
    
    # 3. Calculate window length in samples
    if dominant_freq < 1e-2: # Avoid division by zero
        print("Warning: Dominant frequency is near zero. Using default nperseg=64.")
        return 64
    
    dominant_period = 1.0 / dominant_freq
    window_duration = dominant_period * period_multiplier
    window_samples = int(window_duration / dt)
    print(f"Calculated window length: {window_samples} samples (~{window_duration*1000:.1f} ms)")
    
    # 4. Select next power of 2
    nperseg = _next_power_of_2(window_samples)
    print(f"Selected next power of 2 for nperseg: {nperseg}\n")
    
    return nperseg

def stft_matching_trace(source_trace, reference_trace, dt, nperseg, noverlap_ratio=0.75):
    """Performs spectral matching using STFT on a single trace pair."""
    # (This function is unchanged, but now receives nperseg as a parameter)
    fs = 1.0 / dt
    noverlap = int(nperseg * noverlap_ratio)

    f, t, Zxx_source = signal.stft(source_trace, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Zxx_ref = signal.stft(reference_trace, fs=fs, nperseg=nperseg, noverlap=noverlap)

    reference_amplitude = np.abs(Zxx_ref)
    source_phase = np.angle(Zxx_source)
    Zxx_new = reference_amplitude * np.exp(1j * source_phase)

    _, prelim_matched_trace = signal.istft(Zxx_new, fs=fs, nperseg=nperseg, noverlap=noverlap)
    prelim_matched_trace = prelim_matched_trace[:len(source_trace)]

    rms_source = np.sqrt(np.mean(source_trace**2))
    rms_prelim = np.sqrt(np.mean(prelim_matched_trace**2))

    if rms_prelim < 1e-10: return np.zeros_like(source_trace)
    scale_factor = rms_source / rms_prelim
    return prelim_matched_trace * scale_factor

def save_to_segy(source_filepath, output_filepath, matched_data, dt_seconds):
    """Saves the modified trace data to a new SEG-Y file."""
    try:
        with segyio.open(source_filepath, ignore_geometry=True) as src_f:
            spec = segyio.spec()
            spec.format = src_f.format; spec.tracecount = src_f.tracecount
            spec.samples = src_f.samples; spec.sorting = src_f.sorting
            with segyio.create(output_filepath, spec) as dst_f:
                dst_f.text[0] = src_f.text[0]; dst_f.bin = src_f.bin
                dst_f.bin[segyio.BinField.Interval] = int(dt_seconds * 1e6)
                dst_f.header = src_f.header; dst_f.trace = matched_data.T
        print(f"Successfully saved matched data to {output_filepath}")
    except Exception as e:
        print(f"Error saving to {output_filepath}: {e}")

def plot_section_results(source_data, ref_data, matched_data):
    """Visualizes the full seismic sections."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    fig.suptitle('STFT Matching Results - Seismic Sections', fontsize=16)

    clip_val = np.percentile(np.abs(source_data), 99)

    def plot_section(ax, data, title):
        ax.imshow(data, aspect='auto', cmap='gray_r', vmin=-clip_val, vmax=clip_val)
        ax.set_title(title)
        ax.set_xlabel('Trace Number')
    
    plot_section(axs[0], source_data, 'Source Section')
    axs[0].set_ylabel('Time Sample')
    plot_section(axs[1], ref_data, 'Conformed Reference Section')
    plot_section(axs[2], matched_data, 'STFT Matched Section')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    source_segy_file = 'input.sgy'
    reference_segy_file = 'reference.sgy'
    output_segy_file = 'matched_stft.sgy'

    source_data, source_dt = load_segy_data(source_segy_file)
    reference_data, ref_dt = load_segy_data(reference_segy_file)

    if source_data is not None and reference_data is not None:
        
        conformed_ref_data = conform_reference_to_source(
            reference_data, ref_dt, source_data, source_dt
        )
        
        # --- NEW: Automatically estimate the best nperseg ---
        # You can still tune the multiplier if needed. Higher = better freq resolution.
        stft_window_length = estimate_optimal_nperseg(source_data, source_dt, period_multiplier=3)
        
        # Define other STFT parameters
        stft_overlap_ratio = 0.5

        # --- Apply STFT matching to the entire section ---
        n_traces = source_data.shape[1]
        matched_section = np.zeros_like(source_data)

        print(f"Starting STFT matching for {n_traces} traces using nperseg={stft_window_length}...")
        for i in range(n_traces):
            if (i + 1) % 50 == 0:
                print(f"  Processing trace {i + 1}/{n_traces}")

            matched_section[:, i] = stft_matching_trace(
                source_data[:, i],
                conformed_ref_data[:, i],
                source_dt,
                nperseg=stft_window_length,
                noverlap_ratio=stft_overlap_ratio
            )
        print("STFT matching complete.")
        # --- 4. Save and Visualize ---
        save_to_segy(source_segy_file, output_segy_file, matched_section, source_dt)
        plot_section_results(source_data, conformed_ref_data, matched_section)