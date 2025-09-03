import segyio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import ssqueezepy as ssq
from tqdm import tqdm

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

def cwt_matching_trace(source_trace, reference_trace, dt, wavelet, nv):
    """
    Performs spectral matching using CWT on a single trace pair via ssqueezepy.
    """
    # Generate scales without the 'fs' parameter
    scales = ssq.utils.make_scales(len(source_trace), nv=nv, wavelet=wavelet)
    
    # Proceed with CWT using the generated scales
    coeffs_source, _ = ssq.cwt(source_trace, wavelet=wavelet, scales=scales, fs=1/dt)
    coeffs_ref, _ = ssq.cwt(reference_trace, wavelet=wavelet, scales=scales, fs=1/dt)
    
    # Match in the time-frequency domain
    reference_amplitude = np.abs(coeffs_ref)
    source_phase = np.angle(coeffs_source)
    new_coeffs = reference_amplitude * np.exp(1j * source_phase)
    
    # Inverse CWT
    prelim_matched_trace = ssq.icwt(new_coeffs, wavelet=wavelet, scales=scales)
    prelim_matched_trace = prelim_matched_trace.real[:len(source_trace)]
    
    # Preserve amplitude
    rms_source = np.sqrt(np.mean(source_trace**2))
    rms_prelim = np.sqrt(np.mean(prelim_matched_trace**2))
    if rms_prelim < 1e-10:
        return np.zeros_like(source_trace)
    scale_factor = rms_source / rms_prelim
    return prelim_matched_trace * scale_factor

def save_to_segy(source_filepath, output_filepath, matched_data, dt_seconds):
    """Saves the modified trace data to a new SEG-Y file."""
    try:
        with segyio.open(source_filepath, ignore_geometry=True) as src_f:
            spec = segyio.spec(); spec.format = src_f.format; spec.tracecount = src_f.tracecount
            spec.samples = src_f.samples; spec.sorting = src_f.sorting
            with segyio.create(output_filepath, spec) as dst_f:
                dst_f.text[0] = src_f.text[0]; dst_f.bin = src_f.bin
                dst_f.bin[segyio.BinField.Interval] = int(dt_seconds * 1e6)
                dst_f.header = src_f.header; dst_f.trace = matched_data.T
        print(f"Successfully saved matched data to {output_filepath}")
    except Exception as e:
        print(f"Error saving to {output_filepath}: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    source_segy_file = 'input.sgy'
    reference_segy_file = 'reference.sgy'
    output_segy_file = 'matched_cwt.sgy'

    number_of_voices = 16

    source_data, source_dt = load_segy_data(source_segy_file)
    reference_data, ref_dt = load_segy_data(reference_segy_file)

    if source_data is not None and reference_data is not None:
        
        conformed_ref_data = conform_reference_to_source(
            reference_data, ref_dt, source_data, source_dt
        )
        
        wavelet_spec = ('morlet', {'mu': 5}) 

        n_traces = source_data.shape[1]
        matched_section = np.zeros_like(source_data)

        print(f"\nStarting CWT matching for {n_traces} traces using ssqueezepy (nv={number_of_voices})...")
        for i in tqdm(range(n_traces)):       

            matched_section[:, i] = cwt_matching_trace(
                source_data[:, i],
                conformed_ref_data[:, i],
                source_dt,
                wavelet=wavelet_spec,
                nv=number_of_voices
            )
        print("CWT matching complete.")

        save_to_segy(source_segy_file, output_segy_file, matched_section, source_dt)