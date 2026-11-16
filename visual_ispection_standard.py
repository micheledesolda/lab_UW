"""Thesis figures: full-record examples and direct-arrival evolution.

Run with the existing project environment:
    poetry run python visual_ispection_standard.py
All trace times below are absolute times after the acquisition trigger, in us.
"""

from pathlib import Path
import traceback
from contextlib import contextmanager
from tempfile import TemporaryDirectory
import gc
import mmap

import numpy as np
from scipy.signal import butter, sosfreqz
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import tukey

# This batch script only exports files; avoid GUI/inline figure retention.
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lab_uw.data_io.data_io import MultiChannelUltrasonicDataHandler, UltrasonicDataHandler
from lab_uw.directory_manager import DirectoryManager
from lab_uw.plotting import Plotter

MECHANICAL_INSPECTION_API = 1


# ---------------- USER CONFIGURATION ----------------
base_dir = "/home/michele/Desktop/Dottorato/active_source_implementation"
machine_name = "big_biax"
experiment_name = "084"
data_type = "uw_data/data_tsv_files"

# IMPORTANT: replace these illustrative limits with those in the inversion.
# This single window controls every direct-arrival figure AND its FFT input.
direct_window_s_us = (10.0, 69.)
direct_window_p_us = (10.0, 69.)

direct_window_overrides_us = {
    "ch:1": direct_window_s_us,
    "ch:2": direct_window_p_us,
}

number_of_example_waveforms = 30   # first, middle, last; per file and channel
example_indices = None           # optional explicit indices, e.g. [0, 1200, 2400]
number_of_overlay_waveforms = 12

freq_min = 0.1                  # MHz; None for a low-pass filter
freq_max = 2.6                   # MHz; None for a high-pass filter
filter_order = 4
spectrum_max_Mhz = 8.0           # plot limit; None displays the Nyquist range
show_filter_response = False      # optional dimensionless secondary axis
show_phase_spectrum = False
spectral_remove_mean = True
spectral_taper_fraction = 0.10     # total Tukey alpha, as in spectral_diagnostics.py
spectral_zero_padding_factor = 4  # interpolates FFT, without adding resolution

# Target workspace for numeric blocks, excluding input, examples and figures.
processing_memory_mb = 32
processing_block_rows = 128
map_max_columns = 1600
map_max_rows = 1024
cache_directory = None  # None uses a temporary folder beside the output images

# Unit of time_ax_acquisition as supplied by reader. Existing plots
# treated it as seconds. If the reader supplies microseconds, set "us" here.
# Explicit metadata['time_ax_acquisition_unit'], when present, takes priority.
acquisition_time_unit = "s"        # allowed values: "s", "ms", "us"

save_pdf = True
plot_settings = {
    "figure_size": (12, 7),
    "fontsize_title": 32,
    "fontsize_labels": 28,
    "fontsize_ticks": 22,
    "fontsize_legend": 20,
    "line_width": 1.8,
    "format": ".png",
    "additional_formats": (".pdf",) if save_pdf else (),
    "dpi": 300,
    "pdf_raster_dpi": 150,
    "spectrum_dynamic_range_decades": 4.0,
    "processing_memory_mb": processing_memory_mb,
    "map_max_columns": map_max_columns,
    "map_max_rows": map_max_rows,
}
# ---------------- END CONFIGURATION ----------------

# Select output categories here; omitted categories are neither plotted nor
# computed. Calls with on_products retain the complete callback payload.
ALL_IMAGE_TYPES = (
    "full_record_examples", "direct_waveforms", "direct_amplitude_maps",
    "direct_spectrum_maps", "direct_spectrum_distributions",
    "direct_spectrum_examples", "direct_waveform_differences",
    "direct_spectral_differences", "direct_difference_spectra",
)

image_types = (
    "full_record_examples",
    "direct_waveforms",
    "direct_amplitude_maps",
    "direct_spectrum_maps",
    # "direct_spectrum_distributions",
    "direct_spectrum_examples",
    # "direct_waveform_differences",
    # "direct_spectral_differences",
    # "direct_difference_spectra",
)


def iter_channels(acquisition):
    if isinstance(acquisition, MultiChannelUltrasonicDataHandler):
        yield from acquisition.items()
    else:
        yield acquisition.metadata.get("channel_name", "channel"), acquisition


def _output_stem(infile_path):
    stem = Path(infile_path)
    while stem.suffix:
        stem = stem.with_suffix("")
    return stem.name


def resolve_direct_window(infile_path, channel_name):
    channel = str(channel_name).strip().lower()
    try:
        return tuple(direct_window_overrides_us[channel])
    except KeyError:
        raise ValueError(
            f"Nessuna finestra definita per il canale {channel_name!r} "
            f"nel file {infile_path!s}."
        ) from None


def choose_examples(number_of_waveforms):
    if example_indices is None:
        if not isinstance(number_of_example_waveforms, int) or number_of_example_waveforms < 1:
            raise ValueError("number_of_example_waveforms must be a positive integer.")
        return np.unique(np.linspace(0, number_of_waveforms - 1,
                                     min(number_of_example_waveforms, number_of_waveforms), dtype=int))
    indices = np.asarray(example_indices)
    if indices.ndim != 1 or not indices.size or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("example_indices must be a non-empty list of integer indices.")
    if np.any(indices < 0) or np.any(indices >= number_of_waveforms):
        raise ValueError(f"example_indices must lie between 0 and {number_of_waveforms - 1}.")
    return np.unique(indices)


def prepare_full_record(uw):
    data = np.asarray(uw.waveform_data)
    metadata = dict(uw.metadata)
    time = np.asarray(metadata["time_ax_waveform"], dtype=float)
    if data.ndim != 2 or not data.size or time.shape != (data.shape[1],):
        raise ValueError("Expected non-empty (waveforms, samples) data with matching trace times.")
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Data must be finite and trace times strictly increasing.")
    dt_us = float(metadata["sampling_rate"])
    if not np.isfinite(dt_us) or dt_us <= 0 or not np.allclose(np.diff(time), dt_us, rtol=1e-5, atol=1e-10):
        raise ValueError("Trace times must be uniformly sampled at metadata['sampling_rate'] us.")
    times = np.asarray(metadata["time_ax_acquisition"], dtype=float)
    if times.shape != (data.shape[0],) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Acquisition times must match waveform rows and strictly increase.")
    unit = metadata.get("time_ax_acquisition_unit", acquisition_time_unit)
    scales = {"s": 1.0, "ms": 1e-3, "us": 1e-6}
    if unit not in scales:
        raise ValueError("Specify acquisition_time_unit as 's', 'ms', or 'us'.")
    metadata.update(
        number_of_samples=data.shape[1], number_of_waveforms=data.shape[0],
        time_ax_waveform=time, time_ax_acquisition=times * scales[unit],
        time_ax_acquisition_unit="s",
    )
    return UltrasonicDataHandler(data, metadata)


def filter_response(frequency_mhz, dt_us):
    if freq_min is None and freq_max is None:
        return np.ones_like(frequency_mhz)
    if freq_min is not None and freq_max is not None:
        cutoff, kind = [freq_min, freq_max], "bandpass"
    elif freq_min is not None:
        cutoff, kind = freq_min, "highpass"
    else:
        cutoff, kind = freq_max, "lowpass"
    sos = butter(filter_order, cutoff, btype=kind, fs=1.0 / dt_us, output="sos")
    _, transfer = sosfreqz(sos, worN=frequency_mhz, fs=1.0 / dt_us)
    return np.abs(transfer) ** 2   # forward/backward zero-phase amplitude response


def _block_rows(full_samples, nfft):
    if not np.isfinite(processing_memory_mb) or processing_memory_mb <= 0:
        raise ValueError('processing_memory_mb must be positive.')
    if not isinstance(processing_block_rows, int) or processing_block_rows < 1:
        raise ValueError('processing_block_rows must be a positive integer.')
    per_row = max(full_samples * 8 * 8, nfft * 16 * 5)
    return max(1, min(processing_block_rows, int(processing_memory_mb * 1024**2 // per_row)))


def _release_disk_arrays(arrays, *, flush=True):
    for array in arrays:
        if flush:
            array.flush()
        mapping = getattr(array, '_mmap', None)
        if mapping is not None and hasattr(mapping, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
            try:
                mapping.madvise(mmap.MADV_DONTNEED)
            except (OSError, ValueError):
                pass


@contextmanager
def channel_products(full, window, examples, cache_parent, *, already_preprocessed=False,
                     image_types_to_process=None):
    """Compute only requested products, processing complete traces in blocks.

    The default retains the complete original payload for external callers.
    Full-dataset derived matrices stay on disk. Filter before cropping so block
    processing preserves the complete-record filter boundary conditions.
    """
    selected = set(ALL_IMAGE_TYPES if image_types_to_process is None else image_types_to_process)
    unknown = selected.difference(ALL_IMAGE_TYPES)
    if unknown:
        raise ValueError(f'Unknown image categories: {sorted(unknown)}')
    count, full_samples = full.waveform_data.shape
    example_waveforms = bool(selected.intersection({'full_record_examples', 'direct_waveforms'}))
    example_spectra = 'direct_spectrum_examples' in selected
    need_examples = example_waveforms or example_spectra
    examples = np.asarray(examples, dtype=int) if need_examples else np.empty(0, dtype=int)
    if need_examples and not examples.size:
        raise ValueError('Selected example plots require at least one example.')
    need_direct = bool(selected.difference({'full_record_examples'}))
    need_filtered = bool(selected.intersection({'direct_waveforms', 'direct_amplitude_maps'}))
    need_amplitude = bool(selected.intersection({'direct_spectrum_maps', 'direct_spectrum_distributions'}))
    wave_changes = count > 1 and 'direct_waveform_differences' in selected
    spectral_changes = count > 1 and 'direct_spectral_differences' in selected
    difference_spectra = count > 1 and 'direct_difference_spectra' in selected
    fft_all = need_amplitude or spectral_changes or difference_spectra
    need_fft = fft_all or example_spectra
    all_rows = need_filtered or fft_all or wave_changes

    metadata = dict(full.metadata)
    sample_slice = slice(None)
    samples = full_samples
    if need_direct:
        direct = UltrasonicDataHandler(full.waveform_data, metadata).select_waveform_window(*window)
        metadata = dict(direct.metadata)
        samples = direct.waveform_data.shape[1]
        if samples < 1:
            raise ValueError(f'Direct window {window} contains no samples.')
        start = int(np.searchsorted(full.metadata['time_ax_waveform'], metadata['time_ax_waveform'][0]))
        sample_slice = slice(start, start + samples)
        del direct

    nfft = nfreq = 0
    frequency = frequency_limit = taper = None
    if need_fft:
        if not isinstance(spectral_zero_padding_factor, int) or spectral_zero_padding_factor < 1:
            raise ValueError('spectral_zero_padding_factor must be a positive integer.')
        if not 0 <= spectral_taper_fraction <= 1:
            raise ValueError('spectral_taper_fraction must lie in [0, 1].')
        nfft = max(256, spectral_zero_padding_factor * 2 ** int(np.ceil(np.log2(samples))))
        all_frequency = rfftfreq(nfft, d=float(full.metadata['sampling_rate']))
        frequency_limit = (all_frequency[-1] if spectrum_max_Mhz is None
                           else min(float(spectrum_max_Mhz), all_frequency[-1]))
        if not frequency_limit > 0:
            raise ValueError('spectrum_max_Mhz must be positive.')
        nfreq = int(np.searchsorted(all_frequency, frequency_limit, side='right'))
        frequency = all_frequency[:nfreq]
        taper = tukey(samples, alpha=spectral_taper_fraction) if spectral_taper_fraction else None

    def transform(block):
        work = np.array(block, dtype=np.float64, copy=True, order='C')
        if spectral_remove_mean:
            work -= work.mean(axis=-1, keepdims=True)
        if taper is not None:
            work *= taper
        return rfft(work, n=nfft, axis=-1, workers=1)[:, :nfreq]

    rows = _block_rows(full_samples, nfft)
    cache_parent = Path(cache_parent)
    cache_parent.mkdir(parents=True, exist_ok=True)
    arrays = []
    with TemporaryDirectory(prefix='uw_plots_', dir=cache_parent) as temporary:
        def disk_array(name, shape, needed):
            if not needed:
                return None
            array = np.lib.format.open_memmap(Path(temporary) / (name + '.npy'),
                                             mode='w+', dtype=np.float64, shape=shape)
            arrays.append(array)
            return array

        try:
            filtered = disk_array('direct', (count, samples), need_filtered)
            amplitude = disk_array('amplitude', (count, nfreq), need_amplitude)
            changes = disk_array('waveform_changes', (count - 1, samples), wave_changes)
            amplitude_changes = disk_array('amplitude_changes', (count - 1, nfreq), spectral_changes)
            change_spectra = disk_array('change_spectra', (count - 1, nfreq), difference_spectra)
            example_filtered_full = (np.empty((len(examples), full_samples), dtype=np.float64)
                                     if example_waveforms else None)
            example_filtered_amplitude = (np.empty((len(examples), nfreq), dtype=np.float64)
                                          if example_spectra else None)
            previous_direct = previous_transform = None
            spectral_peak = pending_bytes = 0.0
            cache_bytes = sum(array.nbytes for array in arrays)
            print(f'    processing: <= {rows} records/block; '
                  f'{cache_bytes / 1024**2:.1f} MiB temporary disk cache', flush=True)
            # With only example plots, visit only those acquisitions.
            work_count = count if all_rows else len(examples)
            for first in range(0, work_count, rows):
                last = min(first + rows, work_count)
                indices = np.arange(first, last) if all_rows else examples[first:last]
                raw = full.waveform_data[first:last] if all_rows else full.waveform_data[indices]
                if not np.isfinite(raw).all():
                    raise ValueError(f'Non-finite waveform samples in acquisitions {indices.tolist()}.')
                block_metadata = dict(full.metadata)
                block_metadata.update(number_of_waveforms=len(indices),
                                      time_ax_acquisition=full.metadata['time_ax_acquisition'][indices])
                if already_preprocessed:
                    handler = None
                    complete = raw
                else:
                    handler = UltrasonicDataHandler(raw, block_metadata).bandpass(
                        freqmin=freq_min, freqmax=freq_max, order=filter_order)
                    complete = handler.waveform_data
                    metadata.update({k: v for k, v in block_metadata.items() if k.startswith('filter_')})
                block = complete[:, sample_slice]
                if filtered is not None:
                    filtered[first:last] = block
                example_rows = np.flatnonzero((examples >= indices[0]) & (examples <= indices[-1]))
                positions = np.searchsorted(indices, examples[example_rows])
                if example_filtered_full is not None:
                    example_filtered_full[example_rows] = complete[positions]
                transformed = transform(block) if fft_all else None
                amp = np.abs(transformed) if fft_all else None
                if amplitude is not None:
                    amplitude[first:last] = amp
                if amp is not None:
                    spectral_peak = max(spectral_peak, float(amp.max()))
                if example_spectra and example_rows.size:
                    example_amp = np.abs(transformed[positions] if fft_all else transform(block[positions]))
                    example_filtered_amplitude[example_rows] = example_amp
                    spectral_peak = max(spectral_peak, float(example_amp.max()))
                if wave_changes:
                    joined = block if previous_direct is None else np.concatenate((previous_direct, block))
                    offset = first if previous_direct is None else first - 1
                    changes[offset:offset + len(joined) - 1] = np.diff(joined, axis=0)
                    previous_direct = np.array(block[-1:], copy=True)
                    del joined
                if spectral_changes or difference_spectra:
                    joined = (transformed if previous_transform is None
                              else np.concatenate((previous_transform, transformed)))
                    offset = first if previous_transform is None else first - 1
                    pairs = slice(offset, offset + len(joined) - 1)
                    if spectral_changes:
                        amplitude_changes[pairs] = np.diff(np.abs(joined), axis=0)
                    if difference_spectra:
                        change_spectra[pairs] = np.abs(np.diff(joined, axis=0))
                    previous_transform = np.array(transformed[-1:], copy=True)
                    del joined
                pending_bytes += cache_bytes * len(indices) / count
                if pending_bytes >= processing_memory_mb * 1024**2:
                    _release_disk_arrays(arrays)
                    Plotter._release_mapped_pages(full.waveform_data)
                    pending_bytes = 0
                del handler, complete, block, transformed, amp, raw
            _release_disk_arrays(arrays)
            difference_metadata = dict(metadata)
            times = metadata['time_ax_acquisition']
            difference_metadata.update(
                number_of_waveforms=max(0, count - 1),
                time_ax_acquisition=times[:-1] + .5 * np.diff(times),
                difference_definition='next_minus_current', difference_time_reference='pair_midpoint')
            example_original_full = full.waveform_data[examples] if need_examples else None
            example_direct = (example_original_full[:, sample_slice] if need_examples and need_direct else None)
            example_transform = transform(example_direct) if example_spectra else None
            example_amplitude = np.abs(example_transform) if example_spectra else None
            if example_amplitude is not None:
                spectral_peak = max(spectral_peak, float(example_amplitude.max()))
            yield dict(filtered=filtered, amplitude=amplitude, changes=changes,
                       amplitude_changes=amplitude_changes, change_spectra=change_spectra,
                       frequency=frequency, frequency_limit=frequency_limit,
                       metadata=metadata, difference_metadata=difference_metadata,
                       example_original_full=example_original_full,
                       example_filtered_full=example_filtered_full,
                       example_direct=example_direct,
                       example_filtered_direct=(example_filtered_full[:, sample_slice]
                                                if example_waveforms and need_direct else None),
                       example_amplitude=example_amplitude,
                       example_filtered_amplitude=example_filtered_amplitude,
                       example_phase=(np.angle(example_transform)
                                      if example_spectra and show_phase_spectrum else None),
                       spectral_peak=spectral_peak or 1.0, block_rows=rows,
                       cache_bytes=cache_bytes, cache_path=temporary)
        finally:
            for array in arrays:
                try:
                    array.flush()
                finally:
                    array._mmap.close()
            arrays.clear()


def _plot_inversion_waveform(plotter, single_waveform, single_waveform_filtered,
                             metadata, freq_max=None, outfile_path=None, *,
                             title='', highlight_window=None, amplitude_ylim=None):
    fig, ax = plt.subplots(figsize=plotter.settings['figure_size'], layout='constrained')
    time = metadata['time_ax_waveform']
    ax.plot(time, single_waveform, label="Input dell'inversione",
            linewidth=plotter.settings['line_width'])
    if highlight_window is not None:
        ax.axvspan(*highlight_window, alpha=.2, color=plotter.get_color('highlight'))
    plotter._style_data_axes(ax, title, r'Time after trigger [$\mu$s]',
                             f"Amplitude [{metadata.get('amplitude_unit', 'a.u.')}]")
    ax.set_xlim(time[0], time[-1])
    if amplitude_ylim is not None:ax.set_ylim(amplitude_ylim)
    ax.legend(fontsize=plotter.settings['fontsize_legend'])
    plotter.output_path_choice(fig=fig, outfile_path=outfile_path)


def _plot_inversion_spectrum(plotter, signal_freqs, amp_spectrum,
                             phase_spectrum, filtered_amp_spectrum, lowpass_filter,
                             freq_max, outfile_path=None, *, title='',
                             amplitude_ylim=None, amplitude_unit='a.u.', **kwargs):
    fig, ax = plt.subplots(figsize=plotter.settings['figure_size'], layout='constrained')
    ax.plot(signal_freqs, filtered_amp_spectrum, label="Input dell'inversione",
            linewidth=plotter.settings['line_width'])
    plotter._style_data_axes(ax, title, 'Frequency [MHz]', f'Amplitude [{amplitude_unit}]')
    ax.set_xlim(0, freq_max)
    if amplitude_ylim is not None:ax.set_ylim(amplitude_ylim)
    ax.legend(fontsize=plotter.settings['fontsize_legend'])
    plotter.output_path_choice(fig=fig, outfile_path=outfile_path)


def process_channel(uw, infile_path, channel_name, *, plotter=None, output_dirs=None,
                    already_preprocessed=False, window_us=None, on_products=None):
    """Use output_dirs keys to select plots; callbacks retain all products."""
    # MECHANICAL_SELECTION_INSPECTION_V1: safe to call after a GUI selection.
    initial_figures = set(plt.get_fignums())
    if plotter is None:
        plotter = Plotter(settings=plot_settings)
    if output_dirs is None:
        raise ValueError('output_dirs must map figure categories to output folders.')
    selected = set(output_dirs).intersection(ALL_IMAGE_TYPES)
    if not selected and on_products is None:
        return
    full = prepare_full_record(uw)
    window = (resolve_direct_window(infile_path, channel_name) if window_us is None
              else tuple(window_us))
    count = full.waveform_data.shape[0]
    need_examples = on_products is not None or bool(selected.intersection({
        'full_record_examples', 'direct_waveforms', 'direct_spectrum_examples'}))
    examples = choose_examples(count) if need_examples else np.empty(0, dtype=int)
    safe_channel = str(channel_name).replace(':', '_').replace(' ', '_')
    name = f'{_output_stem(infile_path)}_{safe_channel}'
    window_tag = f'direct_{window[0]:g}-{window[1]:g}us'

    def output(category, suffix=''):
        tag = 'full' if category == 'full_record_examples' else window_tag
        return str(Path(output_dirs[category]) / f'{name}_{tag}{suffix}')

    if 'direct_waveforms' in selected and (not isinstance(number_of_overlay_waveforms, int) or number_of_overlay_waveforms < 1):
        raise ValueError('number_of_overlay_waveforms must be a positive integer.')
    acquisition = full.metadata['time_ax_acquisition']
    print(f'  {channel_name}: {count} waveforms; direct window {window} us; '
          f'examples={examples.tolist()}', flush=True)
    output_parent = Path(next(iter(output_dirs.values()))).parent if output_dirs else Path.cwd()
    cache_parent = (Path(cache_directory) if cache_directory is not None
                    else output_parent / 'processing_cache')
    def render(method, *args, **kwargs):
        existing = set(plt.get_fignums())
        try:
            return method(*args, **kwargs)
        finally:
            # Collect AFTER the plot method returns, when its local artist
            # and renderer references are no longer keeping buffers alive.
            for number in set(plt.get_fignums()) - existing:
                plt.close(number)
            gc.collect()

    waveform_plot = (lambda *a, **k: _plot_inversion_waveform(plotter, *a, **k)) if already_preprocessed else plotter.signal_vs_filtered_signal_plot
    spectrum_plot = (lambda *a, **k: _plot_inversion_spectrum(plotter, *a, **k)) if already_preprocessed else plotter.filtered_amp_and_phase_spectrum_plot

    product = None
    try:
        with channel_products(full, window, examples, cache_parent,
                              already_preprocessed=already_preprocessed,
                              image_types_to_process=None if on_products is not None else selected) as product:
            if 'full_record_examples' in selected:
                full_peak = max(float(np.max(np.abs(product['example_original_full']))),
                                float(np.max(np.abs(product['example_filtered_full'])))) or 1.0
            if 'direct_waveforms' in selected:
                direct_peak = max(float(np.max(np.abs(product['example_direct']))),
                                  float(np.max(np.abs(product['example_filtered_direct'])))) or 1.0
            for row, index in enumerate(examples):
                suffix = f'_wave_{index:06d}'
                if 'full_record_examples' in selected:
                    render(waveform_plot,
                        single_waveform=product['example_original_full'][row],
                        single_waveform_filtered=product['example_filtered_full'][row], metadata=full.metadata,
                        freq_max=freq_max, outfile_path=output('full_record_examples', suffix),
                        title=f'Full waveform record: acquisition {index}\nExperiment time: {acquisition[index]:.3f} s',
                        highlight_window=window, amplitude_ylim=(-1.08 * full_peak, 1.08 * full_peak))
                if 'direct_waveforms' in selected:
                    render(waveform_plot,
                        single_waveform=product['example_direct'][row],
                        single_waveform_filtered=product['example_filtered_direct'][row], metadata=product['metadata'],
                        freq_max=freq_max, outfile_path=output('direct_waveforms', suffix),
                        title=f'Direct arrival: acquisition {index}\nExperiment time: {acquisition[index]:.3f} s',
                        amplitude_ylim=(-1.08 * direct_peak, 1.08 * direct_peak))
            if 'direct_waveforms' in selected:
                render(plotter.uw_all_plot,
                    data=product['filtered'], metadata=product['metadata'], step_wf_to_plot=1,
                    highlight_start=0, highlight_end=0, max_waveforms=number_of_overlay_waveforms,
                    xlim_plot=window[1], ticks_steps_waveforms=None,
                    outfile_path=output('direct_waveforms', '_overlay'))
            if 'direct_amplitude_maps' in selected:
                render(plotter.amplitude_map, product['filtered'], product['metadata'],
                                      outfile_path=output('direct_amplitude_maps'))
            frequency, limit = product['frequency'], product['frequency_limit']
            if 'direct_spectrum_maps' in selected:
                render(plotter.amplitude_spectrum_map, frequency, product['amplitude'], product['metadata'], limit,
                                               outfile_path=output('direct_spectrum_maps'))
            if 'direct_spectrum_distributions' in selected:
                render(plotter.amplitude_spectrum_distribution, frequency, product['amplitude'], product['metadata'], limit,
                                                        outfile_path=output('direct_spectrum_distributions'))
            if 'direct_spectrum_examples' in selected:
                gain = filter_response(frequency, float(full.metadata['sampling_rate'])) if show_filter_response else None
                for row, index in enumerate(examples):
                    phase = None
                    if show_phase_spectrum:
                        phase = product['example_phase'][row].copy()
                        phase[1:][np.abs(np.diff(phase)) > np.pi] = np.nan
                    render(spectrum_plot,
                        signal_freqs=frequency, amp_spectrum=product['example_amplitude'][row], phase_spectrum=phase,
                        filtered_amp_spectrum=product['example_filtered_amplitude'][row], lowpass_filter=gain, freq_max=limit,
                        show_phase=show_phase_spectrum, show_filter=show_filter_response,
                        amplitude_ylim=(0, 1.05 * product['spectral_peak']),
                        amplitude_unit=full.metadata.get('amplitude_unit', 'a.u.'),
                        title=f'Direct-arrival spectrum: acquisition {index}\nExperiment time: {acquisition[index]:.3f} s',
                        outfile_path=output('direct_spectrum_examples', f'_wave_{index:06d}'))
            if count > 1:
                if 'direct_waveform_differences' in selected:
                    render(plotter.amplitude_map, product['changes'], product['difference_metadata'], difference=True,
                                          outfile_path=output('direct_waveform_differences'))
                if 'direct_spectral_differences' in selected:
                    render(plotter.amplitude_spectrum_map, frequency, product['amplitude_changes'], product['difference_metadata'],
                                                   limit, difference=True, outfile_path=output('direct_spectral_differences'))
                if 'direct_difference_spectra' in selected:
                    render(plotter.amplitude_spectrum_map, frequency, product['change_spectra'], product['difference_metadata'],
                                                   limit, title='Spectrum of consecutive waveform differences',
                                                   outfile_path=output('direct_difference_spectra'))
            elif selected.intersection({'direct_waveform_differences', 'direct_spectral_differences', 'direct_difference_spectra'}):
                print('    Single waveform: no consecutive-difference maps.')
            if on_products is not None:
                render(on_products, product, plotter)
    finally:
        for number in set(plt.get_fignums()) - initial_figures:
            plt.close(number)
        if product is not None:
            product.clear()
        gc.collect()


def main():
    unknown = set(image_types).difference(ALL_IMAGE_TYPES)
    if unknown:
        raise ValueError(f'Unknown image categories: {sorted(unknown)}')
    if not image_types:
        print('No image categories selected; nothing to process.')
        return
    manager = DirectoryManager(base_dir=base_dir)
    inputs = manager.make_infile_path_list(machine_name, experiment_name, data_type=data_type)
    folders = manager.make_images_folders(machine_name, experiment_name, image_types)
    output_dirs = dict(zip(image_types, folders))
    plotter = Plotter(settings=plot_settings)
    failures = 0
    for infile_path in sorted(inputs):
        acquisition = uw = None
        print(f'PROCESSING UW DATA IN {infile_path}:', flush=True)
        try:
            acquisition = UltrasonicDataHandler.open(Path(infile_path))
            # Release each completed channel before moving to the next one.
            names = list(acquisition.channels) if isinstance(acquisition, MultiChannelUltrasonicDataHandler) else [None]
            for name in names:
                uw = acquisition.channels.pop(name) if name is not None else acquisition
                channel_name = name if name is not None else uw.metadata.get('channel_name', 'channel')
                try:
                    process_channel(uw, infile_path, channel_name, plotter=plotter, output_dirs=output_dirs)
                except Exception:
                    failures += 1
                    traceback.print_exc()
                finally:
                    uw = None
                    gc.collect()
        except Exception:
            failures += 1
            traceback.print_exc()
        finally:
            # Do not retain the previous input while the next reader allocates.
            acquisition = uw = None
            gc.collect()
    if failures:
        raise SystemExit(f'Processing completed with {failures} failed file(s)/channel(s). See tracebacks above.')


if __name__ == '__main__':
    main()
