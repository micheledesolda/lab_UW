# lab_uw/plotting.py

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib import animation
import cycler
import numpy as np
from typing import Optional, Dict, List, Union, Tuple
import pickle
import os
import gc
import mmap

class Plotter:
    """
    A class containing methods for plotting various aspects of waveform data and simulations.
    """

    # Define default settings for plots
    FONT_TYPE = "DejaVu Sans"
    FONT_SIZE = 32
    FIGURE_SIZE = (16, 9)
    FORMAT = ".png"

    PALETTE = {
        'ink': '#202124',
        'white': '#ffffff',
        # BLU shades
        'blu1': '#203a7a',
        'blu2': '#2d4e9e',
        'blu3': '#617fbe',
        'blu4': '#89a5d6',
        'blu5': '#b3c4e5',
        'blu6': '#cad4ec',

        # GRIGIO shades
        'gray1': '#58595b',
        'gray2': '#76777a',
        'gray3': '#9ea0a2',
        'gray4': '#b5b6b8',
        'gray5': '#d5d6d7',
        'gray6': '#ebebec',

        # VERDE shades
        'green1': '#134239',
        'green2': '#479482',
        'green3': '#6fa599',

        # GIALLO shades
        'yellow1': '#f2cb59',
        'yellow2': '#f5d781',
        'yellow3': '#f8e2a7',

        # ROSSO shades
        'red1':    '#d9795c',
        'red2':    '#d04837',
        'red3':    '#d9795c',

        # MARRONE shades
        'brown1':  '#bf9d8f',
        'brown2':  '#d5dbd2',
        'brown3':  '#f2cf9f',
    }

    SEMANTIC = {
        'primary':      'ink',     # 
        'background':    'white',     # very light grey
        'axes_face':     'white',

        'observed':      'blu2',      # darkest blue for data
        'synthetic':     'red2',      # bright red
        'highlight':     'yellow1',   # for emphasis

        'misfit_light':  'blu1',      # pale blue span
        'misfit_dark':   'blu6',   # contrasting span

        'accent':        'green2',    # for PZT blocks, etc.
        'shadow':        'gray3',     # grid & minor elements
    }

    DEFAULT_SETTINGS = {
        'fontsize_title': FONT_SIZE,
        'fontsize_subplot_title': 28,
        'fontsize_labels': 28,
        'fontsize_ticks': 22,
        'fontsize_legend': 20,
        'line_width': 1.8,
        'figure_size': FIGURE_SIZE,
        'format': FORMAT,
    }


    def __init__(self, settings: Optional[Dict] = None):
        if settings is None:
            self.settings = self.DEFAULT_SETTINGS.copy()
        else:
            self.settings = {**self.DEFAULT_SETTINGS, **settings}

        cycle = [
            self.get_color('observed'),
            self.get_color('synthetic'),
            self.get_color('highlight'),
            self.get_color('accent'),
        ]
        plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', cycle)

        # 2) set background & grid colors
        plt.rcParams['figure.facecolor'] = self.get_color('background')
        plt.rcParams['axes.facecolor']   = self.get_color('axes_face')
        plt.rcParams['grid.color']       = self.get_color('shadow')

    def get_color(self, role: str) -> str:
        """
        Return the hex code for a semantic color role.
        """
        key = self.SEMANTIC.get(role)
        if key is None:
            raise KeyError(f"Unknown semantic role '{role}'")
        return self.PALETTE[key]
    
    def output_path_choice(
        self,
        fig: plt.Figure,
        outfile_path: Optional[Union[str, Path]] = None,
        format: Optional[str] = None
    ) -> None:
        """
        Save or display the figure based on the provided outfile_path.
        """
        if format is None:
            format = self.settings['format']

        if outfile_path:
            outfile_path = Path(outfile_path)
            format = '.' + str(format).lstrip('.')
            if outfile_path.suffix != format:
                known_formats = {'.png', '.pdf', '.svg', '.jpg', '.jpeg', '.tif', '.tiff', '.eps', '.webp'}
                outfile_path = (outfile_path.with_suffix(format)
                                if outfile_path.suffix.lower() in known_formats
                                else Path(str(outfile_path) + format))
            outfile_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fig.savefig(outfile_path, dpi=self.settings.get('dpi', 300), bbox_inches='tight')
                for extra_format in self.settings.get('additional_formats', ()):
                    extra_suffix = '.' + str(extra_format).lstrip('.')
                    if extra_suffix != outfile_path.suffix:
                        extra_dpi = (self.settings.get('pdf_raster_dpi', self.settings.get('dpi', 300))
                                     if extra_suffix == '.pdf' else self.settings.get('dpi', 300))
                        fig.savefig(outfile_path.with_suffix(extra_suffix),
                                    dpi=extra_dpi, bbox_inches='tight')
            finally:
                fig.clear()
                plt.close(fig)
                gc.collect()
        else:
            plt.show()

    def _style_data_axes(self, ax, title, xlabel, ylabel):
        """Common readable style for experimental data figures."""
        color = self.get_color('primary')
        ax.set_title(title, fontsize=self.settings['fontsize_title'], pad=18, color=color)
        ax.set_xlabel(xlabel, fontsize=self.settings['fontsize_labels'], labelpad=12, color=color)
        ax.set_ylabel(ylabel, fontsize=self.settings['fontsize_labels'], labelpad=12, color=color)
        ax.tick_params(axis='both', labelsize=self.settings['fontsize_ticks'], colors=color)
        ax.locator_params(axis='x', nbins=5)
        ax.xaxis.get_offset_text().set_fontsize(self.settings['fontsize_ticks'])
        ax.yaxis.get_offset_text().set_fontsize(self.settings['fontsize_ticks'])
        ax.grid(alpha=0.18)

    def _style_data_colorbar(self, fig, image, ax, label):
        colorbar = fig.colorbar(image, ax=ax, pad=0.025)
        colorbar.set_label(label, fontsize=self.settings['fontsize_labels'], labelpad=14)
        colorbar.ax.tick_params(labelsize=self.settings['fontsize_ticks'])
        colorbar.ax.yaxis.get_offset_text().set_fontsize(self.settings['fontsize_ticks'])
        return colorbar

    @staticmethod
    def _coordinate_edges(values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError('Map coordinates must be finite non-empty 1D arrays.')
        if values.size == 1:
            return np.array([values[0] - 0.5, values[0] + 0.5])
        spacing = np.diff(values)
        if np.any(spacing <= 0):
            raise ValueError('Map coordinates must be strictly increasing.')
        return np.r_[values[0] - spacing[0] / 2,
                     values[:-1] + spacing / 2,
                     values[-1] + spacing[-1] / 2]

    @staticmethod
    def _release_mapped_pages(array):
        """Release read disk-cache pages on platforms supporting madvise."""
        current = array
        while current is not None:
            if isinstance(current, np.memmap):
                mapping = getattr(current, '_mmap', None)
                if mapping is not None and hasattr(mapping, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
                    try:
                        current.flush()
                        mapping.madvise(mmap.MADV_DONTNEED)
                    except (OSError, ValueError):
                        pass
                break
            current = getattr(current, 'base', None)

    def _bounded_map(self, values, x, y, *, signed):
        """Pool every input cell into a bounded display grid, in small blocks.

        Signed cells retain the value with largest absolute magnitude;
        non-negative cells retain their maximum. This affects display only:
        the processing pipeline computes every original adjacent difference.
        Group boundaries retain the original, possibly nonuniform time edges.
        """
        values = np.asanyarray(values)
        if values.shape != (len(x), len(y)):
            raise ValueError('Map data must have shape (acquisitions, samples).')
        nx = min(len(x), int(self.settings.get('map_max_columns', 1600)))
        ny = min(len(y), int(self.settings.get('map_max_rows', 1024)))
        if min(nx, ny) < 1:
            raise ValueError('Map dimensions and display limits must be positive.')
        row_edges = np.linspace(0, len(x), nx + 1, dtype=int)
        col_edges = np.linspace(0, len(y), ny + 1, dtype=int)
        display = np.empty((nx, ny), dtype=np.float32)
        budget = float(self.settings.get('processing_memory_mb', 32)) * 1024 ** 2
        block_rows = max(1, min(128, int(budget // max(1, len(y) * 32))))
        for column in range(nx):
            maximum = np.full(len(y), -np.inf)
            minimum = np.full(len(y), np.inf) if signed else None
            for start in range(row_edges[column], row_edges[column + 1], block_rows):
                block = values[start:min(start + block_rows, row_edges[column + 1])]
                if not np.isfinite(block).all():
                    raise ValueError('Map data must be finite.')
                np.maximum(maximum, np.max(block, axis=0), out=maximum)
                if signed:
                    np.minimum(minimum, np.min(block, axis=0), out=minimum)
                elif np.any(block < 0):
                    raise ValueError('Amplitude spectra must be non-negative.')
            maximum = np.maximum.reduceat(maximum, col_edges[:-1])
            if signed:
                minimum = np.minimum.reduceat(minimum, col_edges[:-1])
                display[column] = np.where(np.abs(minimum) > np.abs(maximum), minimum, maximum)
            else:
                display[column] = maximum
            if column % 32 == 0:
                self._release_mapped_pages(values)
        self._release_mapped_pages(values)
        xedges = self._coordinate_edges(x)[row_edges]
        yedges = self._coordinate_edges(y)[col_edges]
        return display, xedges, yedges

    def _experimental_heatmap(self, values, x, y, *, title, ylabel,
                              colorbar_label, outfile_path, signed=False,
                              log_scale=False, cmap=None, color_limit=None):
        gc.collect()
        values, xedges, yedges = self._bounded_map(values, x, y, signed=signed)
        fig, ax = plt.subplots(figsize=self.settings['figure_size'], layout='constrained')
        peak = float(np.max(np.abs(values)))
        if signed:
            limit = peak if color_limit is None else float(color_limit)
            if not np.isfinite(limit) or limit < 0:
                raise ValueError('color_limit must be finite and non-negative.')
            limit = limit or 1.0
            norm = mcolors.Normalize(vmin=-limit, vmax=limit)
            cmap = cmap or 'RdBu_r'
        elif log_scale and peak > 0:
            if np.any(values < 0):
                raise ValueError('Logarithmic amplitude maps require non-negative data.')
            # Global color scaling in original units: no normalization per trace.
            decades = float(self.settings.get('spectrum_dynamic_range_decades', 4.0))
            if not 0 < decades <= 300:
                raise ValueError('spectrum_dynamic_range_decades must lie in (0, 300].')
            floor = max(peak * 10.0 ** (-decades), np.finfo(float).tiny)
            norm = mcolors.LogNorm(vmin=floor, vmax=max(peak, floor * 10))
            np.maximum(values, floor, out=values)
            cmap = cmap or 'magma'
        else:
            norm = mcolors.Normalize(vmin=0.0, vmax=peak or 1.0)
            cmap = cmap or 'magma'
        # imshow avoids the dense per-cell vertices and RGBA arrays of QuadMesh.
        # Nonuniform coordinates use QuadMesh only AFTER the bounded pooling.
        uniform = np.allclose(np.diff(xedges), np.diff(xedges)[0]) and np.allclose(np.diff(yedges), np.diff(yedges)[0])
        if uniform:
            image = ax.imshow(values.T, aspect='auto', origin='lower', interpolation='nearest',
                              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap, norm=norm)
        else:
            image = ax.pcolormesh(xedges, yedges, values.T, shading='flat', cmap=cmap,
                                  norm=norm, rasterized=True)
        self._style_data_axes(ax, title, 'Experiment time [s]', ylabel)
        ax.grid(False)
        if len(y) > 1:
            ax.set_ylim(float(y[0]), float(y[-1]))
        self._style_data_colorbar(fig, image, ax, colorbar_label)
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def uw_all_plot(self, data, metadata, step_wf_to_plot, highlight_start,
                    highlight_end, xlim_plot, ticks_steps_waveforms,
                    outfile_path=None, *, max_waveforms=20,
                    title='Direct arrivals'):
        """Overlay a limited number of unnormalized traces, colored by shot time."""
        data = np.asarray(data)
        if data.ndim != 2 or not data.size:
            raise ValueError('data must be a non-empty 2D array.')
        if step_wf_to_plot < 1 or max_waveforms < 1:
            raise ValueError('Trace step and max_waveforms must be positive.')
        time = np.asarray(metadata['time_ax_waveform'])
        indices = np.arange(0, data.shape[0], int(step_wf_to_plot))
        if indices.size > max_waveforms:
            indices = indices[np.linspace(0, indices.size - 1, max_waveforms, dtype=int)]
        if indices[-1] != data.shape[0] - 1 and indices.size < max_waveforms:
            indices = np.r_[indices, data.shape[0] - 1]
        acquisition = np.asarray(metadata.get('time_ax_acquisition', np.arange(data.shape[0])))
        if acquisition.shape != (data.shape[0],):
            raise ValueError('Acquisition times must match the waveform rows.')
        low, high = float(acquisition.min()), float(acquisition.max())
        norm = mcolors.Normalize(low, high if high > low else low + 1)
        cmap = plt.get_cmap(self.settings.get('cmap_time', 'viridis'))
        fig, ax = plt.subplots(figsize=self.settings['figure_size'], layout='constrained')
        for index in indices:
            ax.plot(time, data[index], color=cmap(norm(acquisition[index])),
                    linewidth=self.settings['line_width'], alpha=0.85)
        if not 0 <= highlight_start <= highlight_end <= data.shape[0]:
            raise ValueError('Highlight indices are outside the waveform rows.')
        for index in range(highlight_start, highlight_end):
            ax.plot(time, data[index], color=self.get_color('synthetic'), linewidth=2.0)
        units = metadata.get('amplitude_unit', 'a.u.')
        self._style_data_axes(ax, title, r'Time after trigger [$\mu$s]', f'Amplitude [{units}]')
        ax.set_xlim(time[0], min(xlim_plot, time[-1]) if xlim_plot else time[-1])
        if ticks_steps_waveforms and ticks_steps_waveforms > 0:
            ticks = np.arange(time[0], time[-1] + 1e-10, ticks_steps_waveforms)
            if 2 <= ticks.size <= 8:
                ax.set_xticks(ticks)
        self._style_data_colorbar(fig, plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax,
                                 'Experiment time [s]')
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def amplitude_map(self, data, metadata, outfile_path=None, *,
                      difference=False, title=None, color_limit=None):
        """Map direct-arrival samples, or signed next-minus-current differences."""
        units = metadata.get('amplitude_unit', 'a.u.')
        label = f'Amplitude difference [{units}]' if difference else f'Amplitude [{units}]'
        self._experimental_heatmap(
            data, metadata['time_ax_acquisition'], metadata['time_ax_waveform'],
            title=title or ('Consecutive waveform differences' if difference else 'Direct-arrival amplitude'),
            ylabel=r'Time after trigger [$\mu$s]', colorbar_label=label,
            outfile_path=outfile_path, signed=True, color_limit=color_limit,
            cmap=self.settings.get('cmap', 'RdBu_r'))

    def amplitude_spectrum_map(self, signal_freqs, amp_spectrum, metadata,
                               freq_max, outfile_path=None, *, difference=False,
                               title=None, log_scale=True, color_limit=None):
        """Map one-sided spectra; signed differences use a zero-centered scale.

        The caller computes spectra from the selected direct-arrival window.
        Frequencies are already one-sided and must not be halved a second time.
        """
        frequency = np.asarray(signal_freqs)
        amplitudes = np.asanyarray(amp_spectrum)
        if amplitudes.ndim != 2 or amplitudes.shape[1] != frequency.size:
            raise ValueError('Spectra must have shape (acquisitions, frequencies).')
        stop = int(np.searchsorted(frequency, frequency[-1] if freq_max is None else freq_max, side='right'))
        if stop == 0:
            raise ValueError('No frequencies in the requested plotting range.')
        units = metadata.get('amplitude_unit', 'a.u.')
        label = f'Spectral difference [{units}]' if difference else f'Spectral amplitude [{units}]'
        self._experimental_heatmap(
            amplitudes[:, :stop], metadata['time_ax_acquisition'], frequency[:stop],
            title=title or ('Consecutive spectral amplitude differences' if difference else 'Direct-arrival spectrum'),
            ylabel='Frequency [MHz]', colorbar_label=label,
            outfile_path=outfile_path, signed=difference, log_scale=log_scale,
            color_limit=color_limit,
            cmap=self.settings.get('cmap_difference', 'RdBu_r') if difference else self.settings.get('cmap_spectrum', 'magma'))

    def amplitude_spectrum_distribution(self, signal_freqs, amp_spectrum,
                                        metadata, freq_max, outfile_path=None):
        """Accumulate an exact spectral histogram in bounded row blocks."""
        gc.collect()
        frequency = np.asarray(signal_freqs)
        amplitudes = np.asanyarray(amp_spectrum)
        if amplitudes.ndim != 2 or amplitudes.shape[1] != frequency.size:
            raise ValueError('Spectra must have shape (acquisitions, frequencies).')
        stop = int(np.searchsorted(frequency, frequency[-1] if freq_max is None else freq_max, side='right'))
        frequency, amplitudes = frequency[:stop], amplitudes[:, :stop]
        if frequency.size < 2 or not amplitudes.shape[0]:
            raise ValueError('Require at least two frequencies and one spectrum.')
        budget = float(self.settings.get('processing_memory_mb', 32)) * 1024 ** 2
        rows = max(1, min(128, int(budget // max(1, frequency.size * 128))))
        maximum = 0.0
        for start in range(0, amplitudes.shape[0], rows):
            block = amplitudes[start:start + rows]
            if not np.isfinite(block).all() or np.any(block < 0):
                raise ValueError('Require finite non-negative spectra.')
            maximum = max(maximum, float(block.max()))
            self._release_mapped_pages(amplitudes)
        maximum = maximum or 1.0
        f_edges = np.linspace(frequency[0], frequency[-1], min(200, frequency.size) + 1)
        a_edges = np.linspace(0, maximum * 1.001, 161)
        histogram = np.zeros((len(f_edges) - 1, len(a_edges) - 1))
        for start in range(0, amplitudes.shape[0], rows):
            block = amplitudes[start:start + rows]
            # Only this small block has an expanded frequency vector.
            counts, _, _ = np.histogram2d(
                np.broadcast_to(frequency, block.shape).ravel(), block.ravel(),
                bins=[f_edges, a_edges])
            histogram += counts
            self._release_mapped_pages(amplitudes)

        width, height = self.settings['figure_size']
        fig, axes = plt.subplots(3, 1, figsize=(width, max(14, 1.6 * height)), layout='constrained')
        indices = np.unique(np.linspace(0, amplitudes.shape[0] - 1,
                                         min(12, amplitudes.shape[0]), dtype=int))
        for index, color in zip(indices, plt.get_cmap('viridis')(np.linspace(0, 1, len(indices)))):
            axes[0].plot(frequency, np.array(amplitudes[index], copy=True), color=color,
                         linewidth=self.settings['line_width'])
        self._release_mapped_pages(amplitudes)
        axes[0].set_ylim(bottom=0)
        units = metadata.get('amplitude_unit', 'a.u.')
        ylabel = f'Spectral amplitude [{units}]'
        self._style_data_axes(axes[0], 'Selected direct-arrival spectra', 'Frequency [MHz]', ylabel)
        for ax, logarithmic in zip(axes[1:], (True, False)):
            norm = mcolors.LogNorm(1, max(2, histogram.max())) if logarithmic else None
            image = ax.pcolormesh(f_edges, a_edges, np.ma.masked_equal(histogram.T, 0) if logarithmic else histogram.T,
                                  shading='flat', cmap='magma', norm=norm, rasterized=True)
            self._style_data_axes(ax, 'Spectral amplitude distribution', 'Frequency [MHz]', ylabel)
            self._style_data_colorbar(fig, image, ax, 'Count [log scale]' if logarithmic else 'Count')
        for ax in axes:
            ax.set_xlim(frequency[0], frequency[-1])
        self.output_path_choice(fig=fig, outfile_path=outfile_path)


    def filtered_amp_and_phase_spectrum_plot(
        self, signal_freqs, amp_spectrum, phase_spectrum=None,
        filtered_amp_spectrum=None, lowpass_filter=None, freq_max=None,
        outfile_path=None, *, show_phase=False, show_filter=False,
        amplitude_ylim=None, amplitude_unit='a.u.', title='Direct-arrival amplitude spectrum',
    ):
        """Plot unnormalized spectral amplitudes on a linear y axis.

        Phase is opt-in. An optional filter gain is drawn on its own
        dimensionless axis, without rescaling the data spectra. Supply |H|^2
        for the forward/backward Butterworth filter used by bandpass().
        """
        frequency, amplitude = np.asarray(signal_freqs), np.asarray(amp_spectrum)
        if frequency.ndim != 1 or amplitude.shape != frequency.shape:
            raise ValueError('Frequencies and amplitudes must be matching 1D arrays.')
        if not np.isfinite(amplitude).all() or np.any(amplitude < 0):
            raise ValueError('Spectral amplitudes must be finite and non-negative.')
        show_phase = bool(show_phase and phase_spectrum is not None)
        fig, axes = plt.subplots(2 if show_phase else 1, 1,
                                 figsize=self.settings['figure_size'], squeeze=False,
                                 layout='constrained')
        ax = axes[0, 0]
        ax.plot(frequency, amplitude, color=self.get_color('observed'),
                label='Unfiltered', linewidth=self.settings['line_width'])
        if filtered_amp_spectrum is not None:
            filtered = np.asarray(filtered_amp_spectrum)
            if filtered.shape != frequency.shape:
                raise ValueError('Filtered spectrum must match the frequency axis.')
            ax.plot(frequency, filtered, color=self.get_color('synthetic'),
                    label='Filtered', linewidth=self.settings['line_width'])
        self._style_data_axes(ax, title, 'Frequency [MHz]', f'Spectral amplitude [{amplitude_unit}]')
        ax.set_xlim(0, min(freq_max, frequency[-1]) if freq_max is not None else frequency[-1])
        ax.set_ylim(amplitude_ylim if amplitude_ylim is not None else (0, None))
        lines, labels = ax.get_legend_handles_labels()
        if show_filter and lowpass_filter is not None:
            gain = np.asarray(lowpass_filter)
            if gain.shape != frequency.shape:
                raise ValueError('Filter response must match the frequency axis.')
            gain_ax = ax.twinx()
            gain_ax.plot(frequency, gain, color=self.get_color('accent'), linestyle='--',
                         linewidth=2, label='Filter response')
            gain_ax.set_ylim(0, 1.05)
            gain_ax.set_ylabel('Filter gain [1]', fontsize=self.settings['fontsize_labels'], labelpad=12)
            gain_ax.tick_params(labelsize=self.settings['fontsize_ticks'])
            extra_lines, extra_labels = gain_ax.get_legend_handles_labels()
            lines += extra_lines
            labels += extra_labels
        ax.legend(lines, labels, fontsize=self.settings.get('fontsize_legend', 20), loc='upper right')
        if show_phase:
            phase_ax = axes[1, 0]
            phase_ax.plot(frequency, phase_spectrum, color=self.get_color('accent'))
            phase_ax.set_xlim(ax.get_xlim())
            phase_ax.set_ylim(-np.pi, np.pi)
            phase_ax.set_yticks([-np.pi, 0, np.pi], [r'$-\pi$', '0', r'$\pi$'])
            self._style_data_axes(phase_ax, 'Phase spectrum', 'Frequency [MHz]', 'Phase [rad]')
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def signal_vs_filtered_signal_plot(
        self, single_waveform, single_waveform_filtered, metadata,
        freq_max=None, outfile_path=None, *, title='Direct-arrival waveform',
        highlight_window=None, amplitude_ylim=None,
    ):
        """Compare traces with absolute sample times and optional direct window."""
        time = np.asarray(metadata['time_ax_waveform'])
        original, filtered = np.asarray(single_waveform), np.asarray(single_waveform_filtered)
        if original.ndim != 1 or original.shape != filtered.shape or original.shape != time.shape:
            raise ValueError('Both traces must be 1D and match time_ax_waveform.')
        fig, ax = plt.subplots(figsize=self.settings['figure_size'], layout='constrained')
        ax.plot(time, original, color=self.get_color('observed'), label='Unfiltered',
                linewidth=self.settings['line_width'], alpha=0.8)
        ax.plot(time, filtered, color=self.get_color('synthetic'), label='Filtered',
                linewidth=self.settings['line_width'])
        if highlight_window is not None:
            ax.axvspan(*highlight_window, color=self.get_color('highlight'), alpha=0.22,
                       label='Direct-arrival window')
        self._style_data_axes(ax, title, r'Time after trigger [$\mu$s]',
                              f"Amplitude [{metadata.get('amplitude_unit', 'a.u.')}]")
        ax.set_xlim(time[0], time[-1])
        if amplitude_ylim is not None:
            ax.set_ylim(amplitude_ylim)
        ax.legend(fontsize=self.settings.get('fontsize_legend', 20), loc='upper right')
        self.output_path_choice(fig=fig, outfile_path=outfile_path)


    def wavelet_selection_plot(self,
                               time: np.ndarray,
                               waveform: np.ndarray,
                               ratio: np.ndarray,
                               index_max_list: List[int],
                               index_min_before_list: List[int],
                               index_min_after_list: List[int],
                               outfile_path: Optional[str] = None) -> None:
        """
        Plot waveform data along with STA/LTA analysis results.
        """
        if not all(arr.ndim == 1 for arr in [time, waveform, ratio]):
            raise ValueError("time, waveform, and ratio must be 1D numpy arrays.")
        if not (len(time) == len(waveform) == len(ratio)):
            raise ValueError("time, waveform, and ratio must have the same length.")

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # Recorded waveform
        ax.plot(
            time,
            waveform,
            label='Recorded Waveform',
            color=self.get_color('observed'),
            alpha=0.5,
            linewidth=self.settings['line_width']
        )

        # STA/LTA ratio scaled to waveform amplitude
        norm = np.amax(waveform) / np.amax(ratio)
        ax.plot(
            time,
            norm * ratio,
            label='STA/LTA Ratio',
            color=self.get_color('accent'),
            linewidth=self.settings['line_width']
        )

        # Mark maxima and minima
        ax.plot(
            time[index_max_list],
            norm * ratio[index_max_list],
            marker='o',
            linestyle='',
            label='Maxima',
            color=self.get_color('synthetic')
        )
        ax.plot(
            time[index_min_before_list],
            norm * ratio[index_min_before_list],
            marker='o',
            linestyle='',
            label='Minima Before',
            color=self.get_color('highlight')
        )
        ax.plot(
            time[index_min_after_list],
            norm * ratio[index_min_after_list],
            marker='o',
            linestyle='',
            label='Minima After',
            color=self.get_color('shadow')
        )

        ax.set_title(
            "Wavelet Selection Using STA/LTA",
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            'Time [$\\mu s$]',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            'Amplitude [a.u.]',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )

        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.3, color=self.get_color('shadow'))

        ax.legend(
            loc="lower left",
            fontsize=self.settings['fontsize_ticks'],
            facecolor=self.get_color('background'),
            edgecolor=self.get_color('shadow')
        )

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_simulation_waveform(self,
                                 t: np.ndarray,
                                 sp_simulated: np.ndarray,
                                 sp_recorded: np.ndarray,
                                 misfit_interval: np.ndarray,
                                 outfile_path: Optional[str] = None) -> None:
        """
        Plot the simulated waveform against the recorded waveform,
        with alternating misfit shading and simulation drawn last.
        """
        if not all(arr.ndim == 1 for arr in [t, sp_simulated, sp_recorded]):
            raise ValueError("t, sp_simulated, and sp_recorded must be 1D numpy arrays.")
        if not (len(t) == len(sp_simulated) == len(sp_recorded)):
            raise ValueError("t, sp_simulated, and sp_recorded must have the same length.")

        # bump up title/fonts for emphasis
        # PLOT_MISFIT_MASK_COMPAT_V1
        # The optimizer may supply sample indices or a boolean selection mask.
        # Conversion is local to this plot; the objective's interval is unchanged.
        misfit_interval = np.asarray(misfit_interval)
        if misfit_interval.ndim != 1:
            raise ValueError("misfit_interval must be a 1D mask or index array.")
        if misfit_interval.dtype.kind == 'b':
            if misfit_interval.size != t.size:
                raise ValueError("A boolean misfit mask must match the time axis.")
            misfit_interval = np.flatnonzero(misfit_interval)
        elif misfit_interval.size:
            if misfit_interval.dtype.kind not in 'iu':
                raise ValueError("Misfit sample indices must be integers.")
            if np.any(misfit_interval < 0) or np.any(misfit_interval >= t.size):
                raise ValueError("Misfit sample indices are outside the time axis.")

        title_fs = int(1.2 * self.settings['fontsize_title'])
        label_fs = int(1.1 * self.settings['fontsize_labels'])

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # colors for shading
        light = self.get_color('misfit_light')
        dark  = self.get_color('misfit_dark')

        # --- alternating misfit shading ---
        if misfit_interval.size:
            idx     = np.sort(np.unique(misfit_interval))
            gaps    = np.where(np.diff(idx) > 1)[0]
            blocks  = np.split(idx, gaps + 1)

            # region before first block
            ax.axvspan(t[0], t[blocks[0][0]], color=light, alpha=0.3, zorder=1)

            # shade each misfit block (dark) and gap (light)
            for i, blk in enumerate(blocks):
                ax.axvspan(
                    t[blk[0]],
                    t[blk[-1]],
                    color=dark,
                    alpha=0.5,
                    zorder=2
                )
                if i + 1 < len(blocks):
                    nxt = blocks[i+1]
                    ax.axvspan(
                        t[blk[-1]],
                        t[nxt[0]],
                        color=light,
                        alpha=0.3,
                        zorder=1
                    )

            # region after last block
            ax.axvspan(
                t[blocks[-1][-1]],
                t[-1],
                color=light,
                alpha=0.3,
                zorder=1
            )

            # label the first misfit block
            y0 = np.min(sp_recorded)
            ax.text(
                t[blocks[0][0]],
                y0,
                'Misfit Evaluation Interval',
                ha='left',
                fontsize=label_fs,
                color=self.get_color('shadow'),
                zorder=3
            )
        else:
            print("Misfit interval is empty; cannot shade region.")

        # --- plot the data, simulation on top ---
        ax.plot(
            t,
            sp_recorded,
            label="Recorded Waveform",
            color=self.get_color('observed'),
            linewidth=self.settings['line_width'],
            zorder=2
        )
        ax.plot(
            t,
            sp_simulated,
            label="Simulated Waveform",
            color=self.get_color('synthetic'),
            linewidth=2 * self.settings['line_width'],
            alpha=0.8,
            zorder=4
        )

        # --- formatting ---
        ax.set_title(
            "Ultrasonic Wave Simulation",
            fontsize=title_fs,
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            "Time [$\\mu s$]",
            fontsize=label_fs,
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            "Amplitude [a.u.]",
            fontsize=label_fs,
            color=self.get_color('primary')
        )
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.2, color=self.get_color('shadow'))
        
        ax.set_xlim(left=t[100],right=t[-1])

        ax.legend(
            fontsize=self.settings['fontsize_ticks'],
            facecolor=self.get_color('background'),
            edgecolor=self.get_color('shadow'),
            framealpha=0.8
        )

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)


    def make_movie_from_simulation(self,
                                    outfile_path: str,
                                    x: np.ndarray,
                                    t: np.ndarray,
                                    sp_field: np.ndarray,
                                    sp_recorded: np.ndarray,
                                    sample_dimensions: Tuple[float, float],
                                    idx_dict: Dict[str, np.ndarray]) -> None:
        """
        Create an animation of the wavefield simulation.
        """
        # Input validation unchanged...

        movie_sampling = 10  # Downsampling of the snapshot to speed up movie

        fig, (ax, ax2) = plt.subplots(
            1, 2,
            figsize=self.settings['figure_size'],
            gridspec_kw={'width_ratios': [10, 1]}
        )
        ylim = 1.3 * np.amax(np.abs(sp_field))

        # Axis limits and labels
        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([-ylim, ylim])
        ax.set_title(
            "Ultrasonic Wavefield in DDS Experiment",
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            "Sample Length [cm]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            "Relative Shear Wave Amplitude",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.2, color=self.get_color('shadow'))

        # Shading layers
        is_simple = len(sample_dimensions) == 1
        layers = []
        if is_simple:
            layers = [
                ('PZT Layer 1', 'pzt_1', 'accent'),
                ('PZT Layer 2', 'pzt_2', 'accent'),
                ('Steel Blocks', 'steel_block', 'misfit_light'),
            ]
        else:
            layers = [
                ('PZT Layer 1', 'pzt_1', 'accent'),
                ('PZT Layer 2', 'pzt_2', 'accent'),
                ('Steel Blocks', ['side_block_1','central_block','side_block_2'], 'misfit_light'),
                ('Grooves',       ['groove_sb1','groove_cb1','groove_cb2','groove_sb2'], 'shadow'),
                ('Gouge Layer 1', 'gouge_1', 'highlight'),
                ('Gouge Layer 2', 'gouge_2', 'highlight'),
            ]

        for name, key, role in layers:
            idxs = idx_dict.get(key) if isinstance(key, str) else np.concatenate([idx_dict[k] for k in key if idx_dict.get(k) is not None])
            if idxs is not None and len(idxs):
                ax.axvspan(
                    x[idxs[0]],
                    x[idxs[-1]],
                    color=self.get_color(role),
                    alpha=0.3,
                    label=name
                )

        # Transmitter & Receiver patches
        p1, p2 = idx_dict['pzt_1'], idx_dict['pzt_2']
        pzt_w = x[p1[-1]] - x[p1[0]]
        pzt_h = 4 * pzt_w

        # transmitter
        x_tr = x[p1[-1]]
        ax.add_patch(Rectangle(
            (x_tr - pzt_w, -pzt_h/2),
            pzt_w, pzt_h,
            color=self.get_color('accent')
        ))
        ax.text(
            x_tr - pzt_w/2, -pzt_h,
            'Transmitter',
            ha='center',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('shadow')
        )

        # receiver
        x_rc = x[p2[0]]
        ax.add_patch(Rectangle(
            (x_rc, -pzt_h/2),
            pzt_w, pzt_h,
            color=self.get_color('accent')
        ))
        ax.text(
            x_rc + pzt_w/2, -pzt_h,
            'Receiver',
            ha='center',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('shadow')
        )

        # side panel for recorded signal
        ax2.set_ylim([t[0], t[-1]])
        ax2.set_xlim([-1, 1])
        ax2.set_ylabel(
            "Recorded Signal",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax2.axis('off')
        ax2.invert_yaxis()

        fig.tight_layout()

        # animation setup
        line_wf, = ax.plot([], [], color=self.get_color('shadow'), lw=self.settings['line_width'])
        line_rc, = ax2.plot([], [], color=self.get_color('shadow'), lw=self.settings['line_width'])

        sp_movie = sp_field[::movie_sampling]
        sp_rec   = sp_recorded[::movie_sampling] / np.amax(np.abs(sp_recorded))
        t_rec    = t[::movie_sampling]
        n_frames = len(sp_movie)

        def update(frame):
            line_wf.set_data(x, sp_movie[frame])
            line_rc.set_data(sp_rec[:frame], t_rec[:frame])
            return line_wf, line_rc

        ani = animation.FuncAnimation(
            fig, update, frames=n_frames,
            blit=True, interval=20
        )
        ani.save(outfile_path, fps=30, extra_args=['-vcodec', 'libx264'])
        plt.close(fig)


    def plot_velocity_model(
        self,
        x: np.ndarray,
        c: np.ndarray,
        idx_dict: Dict[str, np.ndarray],
        dx: float,
        pzt_layer_width: float,
        pla_layer_width: float,
        outfile_path: Optional[str] = None,
        quantity: str = "velocity",
    ) -> None:
        """
        Plot a velocity or Kelvin--Voigt damping profile and annotate the DDS
        geometry using the actual grid indices.
        """
        if x.ndim != 1 or c.ndim != 1:
            raise ValueError("x and c must be 1D numpy arrays.")
        if len(x) != len(c):
            raise ValueError("x and c must have the same length.")
        plot_labels = {
            "velocity": {
                "title": "Velocity model",
                "ylabel": r"Velocity $c(x)$ [cm/$\mu$s]",
                "curve": r"$c(x)$",
            },
            "damping": {
                "title": "Kelvin–Voigt damping model",
                "ylabel": r"Damping $\nu(x)$ [cm$^2$/$\mu$s]",
                "curve": r"$\nu(x)$",
            },
        }
        if quantity not in plot_labels:
            raise ValueError("quantity must be either 'velocity' or 'damping'")
        labels = plot_labels[quantity]

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        ax.plot(
            x,
            c,
            label=labels["curve"],
            color=self.get_color('synthetic'),
            linewidth=2 * self.settings['line_width'],
            zorder=5,
        )

        layer_defs = [
            ('pla_1', 'shadow'),
            ('pzt_1', 'synthetic'),
            ('side_block_1', 'misfit_light'),
            ('groove_sb1', 'shadow'),
            ('gouge_1', 'highlight'),
            ('groove_cb1', 'shadow'),
            ('central_block', 'misfit_light'),
            ('groove_cb2', 'shadow'),
            ('gouge_2', 'highlight'),
            ('groove_sb2', 'shadow'),
            ('side_block_2', 'misfit_light'),
            ('pzt_2', 'synthetic'),
            ('pla_2', 'shadow'),
        ]

        region_extents = {}
        for key, role in layer_defs:
            if key not in idx_dict or len(idx_dict[key]) == 0:
                continue

            idx = np.asarray(idx_dict[key], dtype=int)
            start = x[idx[0]] - 0.5 * dx
            end   = x[idx[-1]] + 0.5 * dx
            region_extents[key] = (start, end)

            ax.axvspan(
                start,
                end,
                color=self.get_color(role),
                alpha=0.3,
                zorder=0,
            )

        axis_transform = ax.get_xaxis_transform()

        # Wide regions can carry their own labels without occupying a legend.
        block_labels = {
            "side_block_1": "Side block 1",
            "central_block": "Central block",
            "side_block_2": "Side block 2",
        }
        for key, text in block_labels.items():
            if key not in region_extents:
                continue
            start, end = region_extents[key]
            ax.text(
                0.5 * (start + end),
                0.91,
                text,
                transform=axis_transform,
                ha="center",
                va="top",
                fontsize=self.settings['fontsize_ticks'],
                color=self.get_color('primary'),
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": self.get_color('background'),
                    "edgecolor": "none",
                    "alpha": 0.72,
                },
                zorder=7,
            )

        domain_width = float(x[-1] - x[0])

        def region_center(key: str) -> Optional[float]:
            if key not in region_extents:
                return None
            start, end = region_extents[key]
            return 0.5 * (start + end)

        def annotate_region(
            text: str,
            key: str,
            *,
            text_x: float,
            text_y: float,
        ) -> None:
            center = region_center(key)
            if center is None:
                return
            ax.annotate(
                text,
                xy=(center, 0.13),
                xycoords=axis_transform,
                xytext=(text_x, text_y),
                textcoords=axis_transform,
                ha="center",
                va="bottom",
                fontsize=self.settings['fontsize_ticks'],
                color=self.get_color('primary'),
                arrowprops={
                    "arrowstyle": "->",
                    "color": self.get_color('primary'),
                    "linewidth": 0.9,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
                zorder=8,
            )

        # Gouge layers and grooves are too thin for internal text.  Each gouge
        # is identified separately.  The label shared by each groove pair is
        # placed inside the central block, away from the gouge annotation, and
        # points back to the two corresponding thin groove regions.
        central_extent = region_extents.get("central_block")
        for layer_number, gouge_key, groove_keys, direction in (
            (1, "gouge_1", ("groove_sb1", "groove_cb1"), -1.0),
            (2, "gouge_2", ("groove_cb2", "groove_sb2"), 1.0),
        ):
            gouge_center = region_center(gouge_key)
            available_grooves = [
                region_center(key) for key in groove_keys
                if region_center(key) is not None
            ]
            if gouge_center is not None:
                annotate_region(
                    f"Gouge layer {layer_number}",
                    gouge_key,
                    text_x=gouge_center + direction * 0.055 * domain_width,
                    text_y=0.68,
                )
            if available_grooves:
                if central_extent is not None:
                    central_start, central_end = central_extent
                    central_width = central_end - central_start
                    groove_text_x = (
                        central_start + 0.18 * central_width
                        if layer_number == 1
                        else central_end - 0.18 * central_width
                    )
                else:
                    groove_text_x = float(np.mean(available_grooves))
                annotate_region(
                    "Grooves",
                    groove_keys[0],
                    text_x=groove_text_x,
                    text_y=0.48,
                )
                second_center = region_center(groove_keys[1])
                if second_center is not None:
                    ax.annotate(
                        "",
                        xy=(second_center, 0.13),
                        xycoords=axis_transform,
                        xytext=(groove_text_x, 0.48),
                        textcoords=axis_transform,
                        arrowprops={
                            "arrowstyle": "->",
                            "color": self.get_color('primary'),
                            "linewidth": 0.9,
                            "shrinkA": 2,
                            "shrinkB": 2,
                        },
                        zorder=8,
                    )

        transmitter_pos = pzt_layer_width + pla_layer_width
        receiver_pos = x[-1] - pzt_layer_width - pla_layer_width

        # Compact transducer markers replace the former full-height lines.
        marker_width = max(4.0 * float(dx), 0.012 * domain_width)
        transducer_markers = (
            (transmitter_pos, "Source", self.get_color('highlight')),
            (receiver_pos, "Receiver", self.get_color('accent')),
        )
        for position, text, color in transducer_markers:
            center = float(np.clip(
                position,
                x[0] + 0.5 * marker_width,
                x[-1] - 0.5 * marker_width,
            ))
            ax.add_patch(Rectangle(
                (center - 0.5 * marker_width, 0.02),
                marker_width,
                0.075,
                transform=axis_transform,
                facecolor=color,
                edgecolor=self.get_color('primary'),
                linewidth=0.8,
                clip_on=False,
                zorder=9,
            ))
            ax.text(
                center,
                0.11,
                text,
                transform=axis_transform,
                ha="center",
                va="bottom",
                fontsize=self.settings['fontsize_ticks'],
                color=self.get_color('primary'),
                zorder=9,
            )

        ax.set_title(
            labels["title"],
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary'),
            pad=34,
        )
        ax.set_xlabel(
            "Position (cm)",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            labels["ylabel"],
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.grid(alpha=0.2, color=self.get_color('shadow'))
        ax.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, 1.005),
            fontsize=self.settings['fontsize_ticks'],
            facecolor=self.get_color('background'),
            edgecolor=self.get_color('shadow'),
            framealpha=0.85,
        )
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)
        
    def plot_synthetic_spatial_function(self,
                                        x: np.ndarray,
                                        spatial_function: np.ndarray,
                                        outfile_path: Optional[str] = None) -> None:
        """
        Plot the synthetic spatial function.

        Args:
            x (np.ndarray): Spatial axis.
            spatial_function (np.ndarray): The synthetic spatial function values.
            outfile_path (str, optional): Path to save the plot.

        Raises:
            ValueError: If x and spatial_function are not 1D arrays of the same length.
        """
        if x.ndim != 1 or spatial_function.ndim != 1:
            raise ValueError("x and spatial_function must be 1D numpy arrays.")
        if len(x) != len(spatial_function):
            raise ValueError("x and spatial_function must have the same length.")

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # Plot using the 'synthetic' role color
        ax.plot(
            x,
            spatial_function,
            color=self.get_color('synthetic'),
            linewidth=self.settings['line_width']
        )

        ax.set_title(
            "Synthetic Spatial Function",
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            "Position (cm)",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            "Amplitude",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )

        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.3, color=self.get_color('shadow'))

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_velocity_and_stresses(self,
                                   x_values: np.ndarray,
                                   velocities: np.ndarray,
                                   normal_stress: np.ndarray,
                                   shear_stress: np.ndarray,
                                   x_label: str,
                                   velocity_label: str,
                                   stress_labels: Tuple[str, str],
                                   title: str,
                                   outfile_path: Optional[str] = None) -> None:
        """
        Plot estimated velocities and two stresses vs a common x-axis variable.
        """
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # -- Velocities on left y-axis --
        ax.set_xlabel(
            x_label,
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            velocity_label,
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('observed')
        )
        ax.plot(
            x_values,
            velocities,
            color=self.get_color('observed'),
            label=velocity_label,
            linewidth=self.settings['line_width']
        )
        ax.tick_params(
            axis='x',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.tick_params(
            axis='y',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('observed')
        )

        # -- Normal stress on second y-axis --
        ax2 = ax.twinx()
        ax2.set_ylabel(
            stress_labels[0],
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('accent')
        )
        ax2.plot(
            x_values,
            normal_stress,
            color=self.get_color('accent'),
            linestyle='--',
            label=stress_labels[0],
            linewidth=self.settings['line_width']
        )
        ax2.tick_params(
            axis='y',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('accent')
        )

        # -- Shear stress on third y-axis --
        ax2.spines['right'].set_position(('axes', 1.0))
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('axes', 1.15))
        ax3.set_ylabel(
            stress_labels[1],
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('synthetic')
        )
        ax3.plot(
            x_values,
            shear_stress,
            color=self.get_color('synthetic'),
            label=stress_labels[1],
            linewidth=self.settings['line_width']
        )
        ax3.tick_params(
            axis='y',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('synthetic')
        )

        # -- Legend combining all --
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(
            lines + lines2 + lines3,
            labels + labels2 + labels3,
            loc='upper left',
            fontsize=self.settings['fontsize_ticks'],
            facecolor=self.get_color('background'),
            edgecolor=self.get_color('shadow')
        )

        # -- Title, grid, and layout --
        ax.set_title(
            title,
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        ax.grid(alpha=0.3, color=self.get_color('shadow'))

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_l2_norm_vs_velocity(self,
                                 velocity: np.ndarray,
                                 L2norm: np.ndarray,
                                 outfile_path: Optional[str] = None) -> None:
        """
        Plot the L2 norm vs velocity.
        """
        if velocity.ndim != 1 or L2norm.ndim != 1:
            raise ValueError("velocity and L2norm must be 1D numpy arrays.")
        if len(velocity) != len(L2norm):
            raise ValueError("velocity and L2norm must have the same length.")

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # Plot using semantic colors
        ax.plot(
            velocity,
            L2norm,
            color=self.get_color('synthetic'),
            linewidth=self.settings['line_width']
        )

        ax.set_xlabel(
            'Velocity (cm/$\\mu$s)',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            'L2 Norm of Residuals',
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_title(
            'L2 Norm vs Velocity',
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.3, color=self.get_color('shadow'))

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)
       
    def plot_direct_and_reflections(
        self,
        direct_wave_time: np.ndarray,
        direct_wave_data: np.ndarray,
        reflection_info_list: list[dict],
        outfile_path: Optional[str] = None
    ) -> None:
        """
        Plots the direct wave snippet and each reflection wave snippet,
        showing correlation in the title.
        """
        n_reflections = len(reflection_info_list)
        fig, axs = plt.subplots(
            nrows=n_reflections,
            figsize=self.settings['figure_size']
        )
        if n_reflections == 1:
            axs = [axs]

        for ax, info in zip(axs, reflection_info_list):
            reflection_time = info['reflection_time']
            reflection_data = info['reflection_data']
            arr_time = info['arrival_time']
            corr_coeff = info['corr_coeff']

            overlay_time = reflection_time - reflection_time[0]
            max_ref = np.max(reflection_data) if reflection_data.size else 1.0
            scale = np.max(direct_wave_data) / max_ref if max_ref != 0 else 1.0

            ax.plot(
                overlay_time,
                direct_wave_data,
                label="Direct",
                color=self.get_color('observed'),
                linewidth=self.settings['line_width']
            )
            ax.plot(
                overlay_time,
                scale * reflection_data,
                label="Reflection (scaled)",
                color=self.get_color('synthetic'),
                alpha=0.7,
                linewidth=self.settings['line_width']
            )

            ax.set_title(
                f"Direct vs Reflection at {arr_time:.2f} μs, corr={corr_coeff:.3f}",
                fontsize=self.settings['fontsize_subplot_title'],
                fontname=self.FONT_TYPE,
                color=self.get_color('primary')
            )
            ax.set_xlabel(
                "Time [$\\mu s$]",
                fontsize=self.settings['fontsize_labels'],
                color=self.get_color('primary')
            )
            ax.set_ylabel(
                "Amplitude [a.u.]",
                fontsize=self.settings['fontsize_labels'],
                color=self.get_color('primary')
            )

            ax.tick_params(
                axis='both',
                which='major',
                labelsize=self.settings['fontsize_ticks'],
                colors=self.get_color('shadow')
            )
            ax.grid(alpha=0.3, color=self.get_color('shadow'))

        handles, labels = [], []
        for ax in axs:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        by_label = dict(zip(labels, handles))

        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            ncol=len(by_label),
            fontsize=self.settings['fontsize_ticks'],
            facecolor=self.get_color('background'),
            edgecolor=self.get_color('shadow')
        )

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_reflection_windows(
        self,
        observed_time: np.ndarray,
        waveform: np.ndarray,
        t_start_direct: float,
        t_end_direct: float,
        reflection_info_list: list[dict],
        idx_Dstart: int,
        idx_Dend: int,
        title: str = "Reflections highlighted",
        x_label: str = "Time (μs)",
        y_label: str = "Amplitude",
        outfile_path: Optional[str] = None
    ) -> None:
        """
        Plots the main waveform and highlights the direct wave arrival window,
        plus each reflection window.
        """
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # -- plot main waveform --
        ax.plot(
            observed_time,
            waveform,
            label="Waveform",
            color=self.get_color('observed'),
            linewidth=self.settings['line_width']
        )

        # -- highlight direct arrival --
        ax.axvspan(
            t_start_direct,
            t_end_direct,
            color=self.get_color('highlight'),
            alpha=0.2,
            label="Direct Arrival"
        )

        # -- labels, title, grid, ticks --
        ax.set_xlabel(
            x_label,
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            y_label,
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_title(
            title,
            fontsize=self.settings['fontsize_title'],
            fontname=self.FONT_TYPE,
            color=self.get_color('primary')
        )
        ax.grid(color=self.get_color('shadow'), alpha=0.2)
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )

        # -- highlight each reflection window --
        direct_span = idx_Dend - idx_Dstart
        for i, info in enumerate(reflection_info_list, start=1):
            arr = info['arrival_time']
            start_idx = np.searchsorted(observed_time, arr)
            end_idx = start_idx + direct_span
            if end_idx > len(observed_time):
                break
            ax.axvspan(
                observed_time[start_idx],
                observed_time[end_idx - 1],
                color=self.get_color('misfit_light'),
                alpha=0.2,
                label="Reflection windows" if i == 1 else None
            )

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            ncol=len(by_label),
            fontsize=self.settings['fontsize_ticks'],
            facecolor=self.get_color('background'),
            edgecolor=self.get_color('shadow')
        )

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        self.output_path_choice(fig=fig, outfile_path=outfile_path)
        
    def plot_original_vs_updated_stf(self,
                                 t: np.ndarray,
                                 stf_updated: np.ndarray,
                                 stf_original: np.ndarray,
                                 min_time: float,
                                 max_time : float,
                                 outfile_path: Optional[str] = None) -> None:
        """
        Plot the simulated waveform against the recorded waveform.
        """
        if not all(arr.ndim == 1 for arr in [t, stf_updated, stf_original]):
            raise ValueError("t, stf_updated, and stf_original must be 1D numpy arrays.")
        if not (len(t) == len(stf_updated) == len(stf_original)):
            raise ValueError("t, stf_updated, and stf_original must have the same length.")

        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # Plot original and updated STF using semantic colors
        ax.plot(
            t,
            stf_original,
            label="Original STF",
            color=self.get_color('observed'),
            linewidth=self.settings['line_width'],
            zorder=2
        )
        ax.plot(
            t,
            stf_updated,
            label="Updated STF",
            color=self.get_color('synthetic'),
            linewidth=2 * self.settings['line_width'],
            alpha=1.0,
            zorder=4
        )

        # Title and labels
        ax.set_title(
            "Updating Source Time Function with FWI",
            fontsize=self.settings['fontsize_title'],
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            "Time [$\\mu s$]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            "Amplitude [a.u.]",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )

        # Axis limits, grid, ticks, legend
        ax.set_xlim(left=min_time, right=max_time)
        ax.grid(alpha=0.3, color=self.get_color('shadow'))
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.legend(
            fontsize=self.settings['fontsize_ticks'],
            facecolor=self.get_color('background'),
            edgecolor=self.get_color('shadow')
        )

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_boxplot_parameters(
        self,
        param_matrix: np.ndarray,
        param_labels: List[str],
        title: str,
        ylabel: str,
        outfile_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Creates a box plot for the given parameter matrix.
        """
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        box = ax.boxplot(
            param_matrix,
            patch_artist=True,
            showmeans=True
        )

        # Customize box colors
        for patch in box['boxes']:
            patch.set(facecolor=self.get_color('highlight'), alpha=0.5)
        for median in box['medians']:
            median.set(color=self.get_color('accent'), linewidth=2)
        for mean_line in box['means']:
            mean_line.set(
                marker='o',
                markerfacecolor=self.get_color('synthetic'),
                markeredgecolor=self.get_color('synthetic'),
                markersize=6
            )

        # X‑axis labels
        ax.set_xticks(np.arange(1, len(param_labels) + 1))
        ax.set_xticklabels(
            param_labels,
            rotation=0,
            fontsize=self.settings['fontsize_ticks'],
            color=self.get_color('shadow')
        )

        # Title and y‑label
        ax.set_title(
            title,
            fontsize=self.settings['fontsize_title'],
            fontname=self.FONT_TYPE,
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            ylabel,
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )

        ax.tick_params(
            axis='y',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.1, color=self.get_color('shadow'))

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_histogram_l2_distribution(
        self,
        data: np.ndarray,
        bins: int,
        title: str,
        xlabel: str,
        ylabel: str,
        outfile_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Creates a histogram for the given data.
        """
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # Histogram bars with semantic colors
        ax.hist(
            data,
            bins=bins,
            color=self.get_color('highlight'),
            edgecolor=self.get_color('shadow'),
            alpha=0.7
        )

        # Title and axis labels
        ax.set_title(
            title,
            fontsize=self.settings['fontsize_title'],
            fontname=self.FONT_TYPE,
            color=self.get_color('primary')
        )
        ax.set_xlabel(
            xlabel,
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            ylabel,
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )

        # Tick styling and grid
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.1, color=self.get_color('shadow'))

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def plot_scatter_l2_vs_parameters(
        self,
        param_list: List[Tuple[str, np.ndarray]],
        l2_values: np.ndarray,
        title: str,
        best_index: int = 0,
        outfile_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Creates scatter plots of L2 misfit vs. each parameter in subplots.
        """
        n_params = len(param_list)
        n_cols = 4
        n_rows = int(np.ceil(n_params / n_cols))

        fig, axs = plt.subplots(
            n_rows, n_cols,
            figsize=self.settings['figure_size'],
            tight_layout=True
        )
        axs = axs.flatten()

        for i, (label, param_data) in enumerate(param_list):
            ax = axs[i]
            # scatter all points
            ax.scatter(
                param_data,
                l2_values,
                s=30,
                c=self.get_color('observed'),
                alpha=0.7,
                edgecolors=self.get_color('shadow')
            )
            ax.set_xlabel(
                label,
                fontsize=self.settings['fontsize_labels'],
                color=self.get_color('primary')
            )
            ax.set_ylabel(
                "L2 Misfit",
                fontsize=self.settings['fontsize_labels'],
                color=self.get_color('primary')
            )

            # highlight best point
            best_val = param_data[best_index]
            best_l2 = l2_values[best_index]
            ax.scatter(
                best_val,
                best_l2,
                s=100,
                c=self.get_color('highlight'),
                marker="*",
                zorder=5
            )

            ax.tick_params(
                axis='both',
                which='major',
                labelsize=self.settings['fontsize_ticks'],
                colors=self.get_color('shadow')
            )
            ax.grid(alpha=0.1, color=self.get_color('shadow'))

            min_l2 = np.nanmin(l2_values)
            max_l2 = np.nanmax(l2_values)
            max_ylim = max_l2 if max_l2 < 1.5*min_l2 else 1.5*min_l2
            ax.set_ylim([min_l2, max_ylim])

        # hide unused subplots
        for j in range(n_params, n_rows * n_cols):
            axs[j].axis('off')

        fig.suptitle(
            title,
            fontsize=self.settings['fontsize_title'],
            fontname=self.FONT_TYPE,
            color=self.get_color('primary')
        )
        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

    def misfit_map(self,
                   misfit_grid: np.ndarray,
                   unique_damps: np.ndarray,
                   unique_vels: np.ndarray,
                   outfile_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plots a 2D misfit surface given a misfit grid.
        """
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])

        # Display misfit surface
        im = ax.imshow(
            misfit_grid,
            origin='lower',
            extent=[
                min(unique_damps), max(unique_damps),
                min(unique_vels),  max(unique_vels)
            ],
            aspect='auto',
            cmap=self.settings.get('cmap_misfit', 'viridis')
        )

        # Colorbar with semantic styling
        cbar = fig.colorbar(im, ax=ax, pad=0.04)
        cbar.set_label(
            "Misfit (L2 norm)",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        cbar.ax.yaxis.set_tick_params(
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        plt.setp(cbar.ax.get_yticklabels(), color=self.get_color('shadow'))

        # Axes labels and title
        ax.set_xlabel(
            "Damping",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_ylabel(
            "Velocity",
            fontsize=self.settings['fontsize_labels'],
            color=self.get_color('primary')
        )
        ax.set_title(
            "2D Misfit Surface",
            fontsize=self.settings['fontsize_title'],
            fontname=self.FONT_TYPE,
            color=self.get_color('primary')
        )

        # Tick and grid styling
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.settings['fontsize_ticks'],
            colors=self.get_color('shadow')
        )
        ax.grid(alpha=0.2, color=self.get_color('shadow'))

        fig.tight_layout()
        self.output_path_choice(fig=fig, outfile_path=outfile_path)

def _get_pyplot():
    """
    Ensure an interactive backend if possible, then import and return pyplot.
    Respect MPLBACKEND if the user set it in the environment.
    """
    if "MPLBACKEND" not in os.environ:
        # Try TkAgg first (requires tkinter), then Qt-based backends if available
        for be in ("TkAgg", "Qt5Agg", "QtAgg"):
            try:
                matplotlib.use(be, force=True)
                break
            except Exception:
                continue
    # Now import pyplot under the chosen backend
    import matplotlib.pyplot as plt
    return plt

def _get_button_widget():
    # Import the widget only after backend is set and pyplot is importable
    from matplotlib.widgets import Button as MplButton
    return MplButton


class InteractivePlotter(Plotter):
    def manual_pick_arrival_times(
        self,
        observed_time: np.ndarray,
        observed_waveform: np.ndarray,
        start_time: float = 0.0,
        outfile_path: Optional[Path] = None,
    ) -> List[float]:
        # --- import after backend selection ---
        plt = _get_pyplot()
        MplButton = _get_button_widget()

        # --- input checks ---
        if observed_time.ndim != 1 or observed_waveform.ndim != 1:
            raise ValueError("observed_time and observed_waveform must be 1D numpy arrays.")
        if len(observed_time) != len(observed_waveform):
            raise ValueError("observed_time and observed_waveform must have the same length.")

        picked_times: List[float] = []
        picking_mode = [False]  # mutable closure

        def onclick(event):
            if picking_mode[0] and (event.button == 1) and event.inaxes is ax:
                picking_mode[0] = False
                t = float(event.xdata)
                picked_times.append(t)
                event.inaxes.axvline(x=t, color="r", linestyle="--")
                plt.draw()

        def start_picking_callback(_event):
            picking_mode[0] = True
            print("Picking mode enabled. Left-click to pick arrival times.")

        # 1) figure
        fig, ax = plt.subplots(figsize=self.settings["figure_size"])
        plt.subplots_adjust(bottom=0.2)

        # 2) plot
        ax.plot(observed_time, observed_waveform, label="Waveform")
        ax.set_xlabel('Time [$\\mu s$]', fontsize=self.settings["fontsize_labels"])
        ax.set_ylabel("Amplitude", fontsize=self.settings["fontsize_labels"])
        ax.set_title("Pick Arrival Times", fontsize=self.settings["fontsize_title"])
        ax.set_xlim(observed_time[0], observed_time[-1])
        ax.set_ylim(np.min(observed_waveform), np.max(observed_waveform))
        ax.legend()
        ax.grid(True)

        ax.axvspan(0, start_time, facecolor="0.2", alpha=0.3)
        ax.vlines(x=start_time, ymin=np.min(observed_waveform), ymax=np.max(observed_waveform), colors="k")

        # 3) button
        ax_button = plt.axes([0.70, 0.05, 0.25, 0.08])
        pick_button = MplButton(ax_button, "Click to Allow Picking")
        pick_button.on_clicked(start_picking_callback)

        # 4) connect + show
        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        try:
            plt.show()
        finally:
            fig.canvas.mpl_disconnect(cid)

        # 5) optional save
        if outfile_path:
            outfile_path.parent.mkdir(parents=True, exist_ok=True)
            with open(outfile_path, "wb") as f:
                pickle.dump(picked_times, f)

        return picked_times
