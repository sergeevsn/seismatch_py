import segyio
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy import signal # <-- Import the signal module for resampling

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
    """
    Conforms the reference data to the dimensions and sampling of the source data.

    Args:
        ref_data (np.ndarray): The reference seismic data.
        ref_dt (float): The sampling interval of the reference data.
        src_data (np.ndarray): The source seismic data.
        src_dt (float): The sampling interval of the source data.

    Returns:
        np.ndarray: The conformed reference data.
    """
    print("\n--- Conforming reference section to source dimensions ---")
    
    conformed_ref = ref_data.copy()
    src_n_samples, src_n_traces = src_data.shape
    ref_n_samples, ref_n_traces = ref_data.shape

    # 1. Conform number of traces (n_traces)
    if ref_n_traces > src_n_traces:
        print(f"Reference has more traces ({ref_n_traces}). Slicing middle {src_n_traces} traces.")
        start = (ref_n_traces - src_n_traces) // 2
        end = start + src_n_traces
        conformed_ref = conformed_ref[:, start:end]
    elif ref_n_traces < src_n_traces:
        print(f"Reference has fewer traces ({ref_n_traces}). Padding to {src_n_traces} traces.")
        pad_total = src_n_traces - ref_n_traces
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        conformed_ref = np.pad(conformed_ref, ((0, 0), (pad_left, pad_right)), mode='edge')

    # 2. Conform sampling interval (dt) by resampling
    if abs(ref_dt - src_dt) > 1e-9: # Compare floats with a tolerance
        print(f"Reference dt ({ref_dt*1000:.2f} ms) differs from source ({src_dt*1000:.2f} ms). Resampling reference.")
        # Calculate new number of samples needed to match source dt
        original_time_duration = conformed_ref.shape[0] * ref_dt
        new_n_samples = int(round(original_time_duration / src_dt))
        conformed_ref = signal.resample(conformed_ref, new_n_samples, axis=0)
    
    # 3. Conform number of samples (n_samples) after any resampling
    current_n_samples = conformed_ref.shape[0]
    if current_n_samples > src_n_samples:
        print(f"Reference is longer ({current_n_samples} samples). Truncating to {src_n_samples} samples.")
        conformed_ref = conformed_ref[:src_n_samples, :]
    elif current_n_samples < src_n_samples:
        print(f"Reference is shorter ({current_n_samples} samples). Padding to {src_n_samples} samples.")
        pad_amount = src_n_samples - current_n_samples
        conformed_ref = np.pad(conformed_ref, ((0, pad_amount), (0, 0)), mode='edge')

    print(f"Conforming complete. New reference shape: {conformed_ref.shape}\n")
    return conformed_ref

def calculate_average_spectrum(data):
    """Calculates the average amplitude spectrum for a seismic section."""
    n_samples = data.shape[0]
    fft_all_traces = fft(data, axis=0)
    amplitude_spectra = np.abs(fft_all_traces)
    average_spectrum = np.mean(amplitude_spectra, axis=1)
    return average_spectrum

def apply_spectral_matching_with_amp_preservation(source_data, reference_spectrum):
    """Applies the reference spectrum while preserving trace-wise RMS amplitude."""
    print("Applying spectral matching with amplitude preservation...")
    n_samples, n_traces = source_data.shape
    matched_data = np.zeros_like(source_data)
    source_fft = fft(source_data, axis=0)
    source_phase = np.angle(source_fft)
    
    for i in range(n_traces):
        current_phase = source_phase[:, i]
        new_spectrum_trace = reference_spectrum * np.exp(1j * current_phase)
        prelim_matched_trace = ifft(new_spectrum_trace).real
        
        rms_source = np.sqrt(np.mean(source_data[:, i]**2))
        rms_prelim = np.sqrt(np.mean(prelim_matched_trace**2))
        
        if rms_prelim < 1e-10:
            scale_factor = 0.0
        else:
            scale_factor = rms_source / rms_prelim
        
        matched_data[:, i] = prelim_matched_trace * scale_factor
    return matched_data

def save_to_segy(source_filepath, output_filepath, matched_data, dt_seconds):
    """Saves the modified trace data to a new SEG-Y file."""
    try:
        with segyio.open(source_filepath, ignore_geometry=True) as src_f:
            spec = segyio.spec()
            spec.format = src_f.format
            spec.tracecount = src_f.tracecount
            spec.samples = src_f.samples
            spec.sorting = src_f.sorting

            with segyio.create(output_filepath, spec) as dst_f:
                dst_f.text[0] = src_f.text[0]
                dst_f.bin = src_f.bin
                # CRITICAL: Update the sampling interval in the output binary header
                dst_f.bin[segyio.BinField.Interval] = int(dt_seconds * 1e6)
                dst_f.header = src_f.header
                dst_f.trace = matched_data.T
        print(f"Successfully saved matched data to {output_filepath}")
    except Exception as e:
        print(f"Error saving to {output_filepath}: {e}")

def plot_results(source_data, ref_data, matched_data, dt):
    """Visualizes the seismic sections and their average spectra."""
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Spectral Matching Results', fontsize=16)

    clip_val = np.percentile(np.abs(source_data), 99)

    def plot_section(ax, data, title):
        ax.imshow(data, aspect='auto', cmap='gray_r', vmin=-clip_val, vmax=clip_val)
        ax.set_title(title)
        ax.set_xlabel('Trace Number')
        ax.set_ylabel('Time Sample')

    plot_section(axs[0, 0], source_data, 'Source Section')
    # Note: We plot the *conformed* reference data for a fair visual comparison
    plot_section(axs[0, 1], ref_data, 'Conformed Reference Section')
    plot_section(axs[0, 2], matched_data, 'Matched Section')

    source_spec = calculate_average_spectrum(source_data)
    ref_spec = calculate_average_spectrum(ref_data)
    matched_spec = calculate_average_spectrum(matched_data)
    
    n_samples = source_data.shape[0]
    freq = np.fft.fftfreq(n_samples, d=dt)[:n_samples//2]

    def plot_spectrum(ax, spec, label, color):
        ax.plot(freq, spec[:n_samples//2], label=label, color=color)

    ax_spec = plt.subplot(2, 1, 2)
    plot_spectrum(ax_spec, source_spec, 'Source Spectrum', 'blue')
    plot_spectrum(ax_spec, ref_spec, 'Reference Spectrum', 'red')
    plot_spectrum(ax_spec, matched_spec, 'Matched Spectrum', 'green')
    ax_spec.set_title('Average Amplitude Spectra')
    ax_spec.set_xlabel('Frequency (Hz)')
    ax_spec.set_ylabel('Amplitude')
    ax_spec.legend()
    ax_spec.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    # DEFINE YOUR FILE PATHS HERE
    source_segy_file = 'input.sgy'
    reference_segy_file = 'reference.sgy'
    output_segy_file = 'matched_spec.sgy'

    source_data, source_dt = load_segy_data(source_segy_file)
    reference_data, ref_dt = load_segy_data(reference_segy_file)

    if source_data is not None and reference_data is not None:
        
        # 1. Conform reference data to source data dimensions
        conformed_reference_data = conform_reference_to_source(
            reference_data, ref_dt, source_data, source_dt
        )

        # 2. Calculate Average Spectrum of the now-conformed reference
        avg_ref_spectrum = calculate_average_spectrum(conformed_reference_data)
        
        # 3. Apply Spectral Matching
        matched_data = apply_spectral_matching_with_amp_preservation(source_data, avg_ref_spectrum)

        # 4. Save the result (using the source's dt)
        save_to_segy(source_segy_file, output_segy_file, matched_data, source_dt)
        
        # 5. Visualize the results
        plot_results(source_data, conformed_reference_data, matched_data, source_dt)