# lab_uw/forward_modeling.py
import pathlib
import sys
import matplotlib.pyplot as plt
from math import floor, ceil
import numpy as np
from numpy import linalg as LA
from typing import Union, Tuple, Optional, Dict, Any, Sequence
from scipy.sparse import csr_matrix
from scipy.signal.windows import tukey

from lab_uw.simulation_setup import (
    Grid1D,
    VelocityModel1D_DDS,
    VelocityModel1D_SingleBlock,
    Source1D,
    Receiver1D,
    SimulationTime
)
from lab_uw.plotting import Plotter
try:
    from lab_uw.data_io.data_io import UltrasonicDataHandler
except ModuleNotFoundError as exc:
    # The update archive can be exercised by the self-contained analytic
    # synthetic validator before it is overlaid on the complete project.  The
    # production project always supplies this class; forward modeling itself
    # only requires an object exposing ``waveform_data`` and ``metadata``.
    if exc.name not in {"lab_uw.data_io", "lab_uw.data_io.data_io"}:
        raise

    class UltrasonicDataHandler:  # pragma: no cover - overlay-only fallback
        pass
from lab_uw.fd1d import (
    discrete_model_gradient_1d_kelvin_voigt,
    finite_difference_1d_kelvin_voigt,
    finite_difference_1d_receiver_trace,
    interpolation_adjoint,
    receiver_adjoint,
    sample_receiver,
)
from lab_uw.inversion_windows import sample_interval_mask
from lab_uw.spectral_diagnostics import (
    compare_spectral_shapes,
    plot_spectral_comparison,
)


def _local_inversion_plot_paths(
    output_directory: pathlib.Path,
    label: str,
    name_prefix: Optional[str] = None,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Return local-result stems inside the record-specific directory."""
    output_directory = pathlib.Path(output_directory)
    prefix = output_directory.name if name_prefix is None else str(name_prefix)
    result_stem = output_directory / f"{prefix}{label}"
    return (
        result_stem,
        output_directory / f"{result_stem.name}_velocity_model",
        output_directory / f"{result_stem.name}_damping_model",
    )


def _filename_decimal(value: float, places: int = 4) -> str:
    """Format a decimal without a dot that pathlib can mistake for a suffix."""
    return f"{float(value):.{int(places)}f}".replace(".", ",")


def _waveform_interval_indices(interval, size: int) -> np.ndarray:
    """Return validated integer indices for a waveform interval."""

    if isinstance(interval, slice):
        indices = np.arange(size, dtype=int)[interval]
    else:
        values = np.asarray(interval)
        if values.dtype == np.bool_:
            mask = values.reshape(-1)
            if mask.size != size:
                raise ValueError("a boolean interval must match the waveform length")
            indices = np.flatnonzero(mask)
        else:
            indices = values.astype(int, copy=False).reshape(-1)
    if indices.size == 0:
        raise ValueError("waveform interval is empty")
    if np.any(indices < 0) or np.any(indices >= size):
        raise IndexError("waveform interval contains out-of-range indices")
    return indices


def relative_waveform_objective(
    observed_waveform: np.ndarray,
    predicted_waveform: np.ndarray,
    interval,
    *,
    spectral_gamma: float = 1.0,
    taper_fraction: float = 0.0,
    return_gradient: bool = False,
    eps: float = 1.0e-15,
) -> Dict[str, Any]:
    r"""Dimensionless amplitude--spectrum--phase waveform objective.

    For the windowed observed and predicted traces ``d`` and ``p`` this
    evaluates

    .. math::

        J_\gamma = \frac{\|d-p\|_2^2}{\|d\|_2^2}
        + 2 r (\gamma-1)D_f^2,

    where ``r=||p||/||d||`` and ``D_f^2`` is the squared Hellinger distance
    between their normalised spectral-energy distributions.  Equivalently,

    .. math::

        J_\gamma=(1-r)^2+2r(\gamma D_f^2+J_\phi).

    Thus ``gamma=1`` is exactly the ordinary relative L2 objective.  The
    returned gradient is that of ``0.5 * J_gamma`` with respect to the full
    predicted trace; this convention matches the adjoint residual used by the
    inversion.
    """

    observed = np.asarray(observed_waveform, dtype=float).reshape(-1)
    predicted = np.asarray(predicted_waveform, dtype=float).reshape(-1)
    if observed.size != predicted.size:
        raise ValueError("observed and predicted waveforms must have equal size")
    gamma = float(spectral_gamma)
    taper = float(taper_fraction)
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError("spectral_gamma must be finite and non-negative")
    if not np.isfinite(taper) or not 0.0 <= taper <= 1.0:
        raise ValueError("taper_fraction must lie in [0, 1]")

    indices = _waveform_interval_indices(interval, observed.size)
    weights = tukey(indices.size, alpha=taper) if taper > 0.0 else np.ones(indices.size)
    data = weights * observed[indices]
    prediction = weights * predicted[indices]

    data_norm = float(np.linalg.norm(data))
    prediction_norm = float(np.linalg.norm(prediction))
    if data_norm <= eps:
        # A zero observed trace is useful in forward-only calls and unit tests,
        # although a relative data objective is then undefined.  Fall back to
        # absolute prediction energy rather than preventing the forward run.
        total = float(np.dot(prediction, prediction))
        result = {
            "total": total,
            "relative_l2": total,
            "amplitude_misfit": total,
            "spectral_shape_misfit": 0.0 if prediction_norm <= eps else 1.0,
            "phase_misfit": 0.0,
            "amplitude_ratio": 0.0 if prediction_norm <= eps else np.inf,
            "correlation": 0.0,
            "spectral_gamma": gamma,
        }
        if return_gradient:
            gradient = np.zeros_like(predicted)
            gradient[indices] = weights * prediction
            result["gradient_predicted"] = gradient
        return result

    residual = prediction - data
    relative_l2 = float(np.dot(residual, residual) / (data_norm * data_norm))
    amplitude_ratio = prediction_norm / data_norm

    if prediction_norm <= eps:
        spectral_distance_squared = 1.0
        correlation = 0.0
        phase_misfit = 0.0
        spectral_gradient_window = np.zeros_like(prediction)
    else:
        data_fft = np.fft.fft(data, norm="ortho")
        prediction_fft = np.fft.fft(prediction, norm="ortho")
        data_spectral_shape = np.abs(data_fft) / data_norm
        prediction_spectral_shape = np.abs(prediction_fft) / prediction_norm
        spectral_affinity = float(
            np.dot(data_spectral_shape, prediction_spectral_shape)
        )
        spectral_affinity = float(np.clip(spectral_affinity, 0.0, 1.0))
        spectral_distance_squared = 1.0 - spectral_affinity
        correlation = float(
            np.dot(data, prediction) / (data_norm * prediction_norm)
        )
        correlation = float(np.clip(correlation, -1.0, 1.0))
        phase_misfit = max(
            0.0,
            1.0 - correlation - spectral_distance_squared,
        )

        unit_phase = np.zeros_like(prediction_fft, dtype=complex)
        nonzero = np.abs(prediction_fft) > eps
        unit_phase[nonzero] = (
            prediction_fft[nonzero] / np.abs(prediction_fft[nonzero])
        )
        affinity_gradient = np.fft.ifft(
            data_spectral_shape * unit_phase,
            norm="ortho",
        ).real
        spectral_gradient_window = (
            prediction / prediction_norm - affinity_gradient
        ) / data_norm

    amplitude_misfit = (1.0 - amplitude_ratio) ** 2
    total = relative_l2 + 2.0 * amplitude_ratio * (
        gamma - 1.0
    ) * spectral_distance_squared

    result: Dict[str, Any] = {
        "total": float(total),
        "relative_l2": relative_l2,
        "amplitude_misfit": float(amplitude_misfit),
        "spectral_shape_misfit": float(spectral_distance_squared),
        "phase_misfit": float(phase_misfit),
        "amplitude_ratio": float(amplitude_ratio),
        "correlation": float(correlation),
        "spectral_gamma": gamma,
    }
    if return_gradient:
        # Gradient of 0.5*J_gamma with respect to the tapered prediction.
        gradient_window = residual / (data_norm * data_norm)
        gradient_window += (gamma - 1.0) * spectral_gradient_window
        gradient = np.zeros_like(predicted)
        gradient[indices] = weights * gradient_window
        result["gradient_predicted"] = gradient
    return result


def select_cross_talk_candidate(
    current_misfit: float,
    velocity_only_misfit: Optional[float],
    damping_only_misfit: Optional[float],
    joint_misfit: Optional[float],
) -> Dict[str, Any]:
    """Select the best multiparameter trial and quantify its interaction.

    The interaction term is the finite-difference mixed contribution

    ``J(c+dc,a+da) - J(c+dc,a) - J(c,a+da) + J(c,a)``.

    A positive value means that the joint perturbation performs worse than
    the sum of the two separate perturbations; a negative value means that
    the two perturbations cooperate.  Missing candidates are represented by
    ``None`` and are excluded from selection.
    """

    candidates = {"current": float(current_misfit)}
    optional_candidates = {
        "velocity": velocity_only_misfit,
        "damping": damping_only_misfit,
        "joint": joint_misfit,
    }
    for label, value in optional_candidates.items():
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{label} candidate misfit must be finite")
        candidates[label] = value

    selected_label = min(candidates, key=candidates.get)
    selected_misfit = candidates[selected_label]
    if selected_misfit >= float(current_misfit):
        selected_label = "current"
        selected_misfit = float(current_misfit)

    interaction = None
    if all(
        value is not None
        for value in (
            velocity_only_misfit,
            damping_only_misfit,
            joint_misfit,
        )
    ):
        interaction = float(
            joint_misfit
            - velocity_only_misfit
            - damping_only_misfit
            + current_misfit
        )

    return {
        "selected": selected_label,
        "selected_misfit": selected_misfit,
        "current_misfit": float(current_misfit),
        "velocity_only_misfit": (
            None
            if velocity_only_misfit is None
            else float(velocity_only_misfit)
        ),
        "damping_only_misfit": (
            None
            if damping_only_misfit is None
            else float(damping_only_misfit)
        ),
        "joint_misfit": (
            None if joint_misfit is None else float(joint_misfit)
        ),
        "interaction": interaction,
    }

class UltrasonicModeler:
    """
    Class for simulating ultrasonic wave propagation under different 1D geometries:
     - "dds": double-direct-shear geometry (two gouge layers)
     - "block": single-block geometry
    """

    def __init__(self, plotter: Optional["Plotter"] = None):
        """
        Initialize the UltrasonicModeler.

        Args:
            plotter (Plotter, optional): For plotting results.
        """
        self.plotter = plotter or Plotter()

    def forward_simulation(
        self,
        geometry_type      : str,                        
        observed_time      : np.ndarray,
        observed_waveform  : np.ndarray,
        stf_handler        : UltrasonicDataHandler,
        frequency_cutoff   : float,
        minimum_velocity   : float,
        maximum_velocity   : float,
        assembly_dict      : Dict[str, Any],
        misfit_interval    : np.ndarray,
        absorbing          : bool = False,
        maximum_damping    : float = None,
        montecarlo         : Optional[Dict[str, Any]] = None,
        source_receiver_config: Optional[Dict[str, Any]] = None,
        embedding_mode     : str = "legacy_stack",
        boundary_condition : Optional[str] = None,
        boundary_impedance : Optional[float | Sequence[float]] = None,
        source_injection   : Optional[str] = None,
        receiver_quantity  : Optional[str] = None,
        points_per_wavelength: Optional[int] = None,
        store_wavefield     : bool = True,
        normalize_waveform : bool = True,
        normalization_interval: Optional[np.ndarray] = None,
        enable_plotting    : bool = False,
        make_movie         : bool = False,
        plot_output_path   : Optional[str] = None,
        movie_output_path  : Optional[str] = "simulation_movie.mp4",
        plot_name_prefix   : str = "",
    ) -> Tuple[
        np.ndarray,  # synthetic_waveform
        Optional[np.ndarray],  # wavefield_forward; None in receiver-only mode
        Union[VelocityModel1D_DDS, VelocityModel1D_SingleBlock],  # velocity_model_handler
        SimulationTime,
        Grid1D,
        Source1D,
        Receiver1D
    ]:
        """
        Unifies forward modeling under different 1D geometries ("dds" or "block").
        The method unpacks geometry-specific parameters, builds the velocity model,
        runs the finite-difference simulation, and stores/returns the results.

        ``embedding_mode='legacy_stack'`` reproduces the historical domain,
        including explicit PLA and PZT layers.  ``'effective_surface'`` is the
        calibrated effective-STF convention: the domain contains only the DDS
        mechanical assembly and the source/receiver operators act on its two
        external steel faces.

        The legacy numerical options reproduce the historical implementation:
        fixed external nodes, displacement sampling and a source added directly
        as a discrete displacement increment.  Reference-block validation must
        pass these options explicitly so that the physical source, receiver and
        boundary conventions are never implicit.

        When ``store_wavefield=False``, the receiver trace is sampled during
        propagation and the complete ``(num_t, num_x)`` wavefield is never
        allocated.  This produces the same synthetic trace and misfit, but
        local inversion and movies still require ``store_wavefield=True``.
        """

        # Explicitly assign each argument to `self`
        self.geometry_type       = geometry_type
        self.observed_time       = observed_time
        self.observed_waveform   = observed_waveform
        self.stf_handler         = stf_handler
        self.frequency_cutoff    = frequency_cutoff
        self.minimum_velocity    = minimum_velocity
        self.maximum_velocity    = maximum_velocity
        self.assembly_dict       = assembly_dict
        self.misfit_interval     = misfit_interval
        self.absorbing           = absorbing
        self.maximum_damping     = maximum_damping
        self.montecarlo          = montecarlo
        self.source_receiver_config = source_receiver_config
        if embedding_mode not in {"legacy_stack", "effective_surface"}:
            raise ValueError(
                "embedding_mode must be 'legacy_stack' or 'effective_surface'"
            )
        if embedding_mode == "effective_surface" and geometry_type.lower() != "dds":
            raise ValueError("effective_surface is currently defined only for DDS geometry")
        self.embedding_mode = embedding_mode
        if boundary_condition is None:
            if embedding_mode == "effective_surface":
                boundary_condition = "free"
            else:
                boundary_condition = "absorbing" if absorbing else "legacy_fixed"
        if source_injection is None:
            source_injection = (
                "body_force" if embedding_mode == "effective_surface"
                else "legacy_increment"
            )
        if receiver_quantity is None:
            receiver_quantity = (
                "velocity" if embedding_mode == "effective_surface"
                else "displacement"
            )
        if points_per_wavelength is None:
            points_per_wavelength = 40 if embedding_mode == "effective_surface" else 5
        self.boundary_condition = boundary_condition
        self.boundary_impedance = boundary_impedance
        self.source_injection   = source_injection
        self.receiver_quantity  = receiver_quantity
        self.points_per_wavelength = int(points_per_wavelength)
        self.store_wavefield     = bool(store_wavefield)
        self.normalize_waveform  = normalize_waveform
        self.normalization_interval = normalization_interval
        self.enable_plotting     = enable_plotting
        self.make_movie          = make_movie
        self.plot_output_path    = plot_output_path
        self.movie_output_path   = movie_output_path
        if make_movie and not self.store_wavefield:
            raise ValueError("make_movie requires store_wavefield=True")

        # LEGACY MONTE CARLO OR DEFAULT PARAMS
        if montecarlo:
            spreading_factor_transmitter = montecarlo.get("spreading_factor_transmitter", 0.1)
            spreading_factor_receiver    = montecarlo.get("spreading_factor_receiver", 0.1)
            position2edge_transmitter    = montecarlo.get("position2edge_transmitter", 0.0)
            position2edge_receiver       = montecarlo.get("position2edge_receiver", 0.0)
            radius_factor_transmitter    = montecarlo.get("radius_factor_transmitter", 1.0)
            radius_factor_receiver       = montecarlo.get("radius_factor_receiver", 1.0)
            multiplier_STF               = montecarlo.get("multiplier_STF", 1.0)

        else:
            spreading_factor_transmitter = 0.1
            spreading_factor_receiver    = 0.1
            position2edge_transmitter    = 0. # -0.5
            position2edge_receiver       = 0. # -0.5
            radius_factor_transmitter    = 1
            radius_factor_receiver       = 1
            multiplier_STF               = 1

        source_receiver_config = {} if source_receiver_config is None else dict(source_receiver_config)
        if embedding_mode == "effective_surface":
            default_source_normalization = (
                "integral" if source_injection == "body_force" else "sum"
            )
            default_receiver_normalization = "sum"
            default_boundary_mode = "truncate"
        else:
            default_source_normalization = "legacy_peak"
            default_receiver_normalization = "legacy_peak"
            default_boundary_mode = "legacy_odd"
        legacy_spatial_normalization = source_receiver_config.get(
            "spatial_normalization", default_source_normalization
        )
        source_normalization = source_receiver_config.get(
            "source_normalization", legacy_spatial_normalization
        )
        receiver_normalization = source_receiver_config.get(
            "receiver_normalization", default_receiver_normalization
        )
        spatial_boundary_mode = source_receiver_config.get(
            "spatial_boundary_mode", default_boundary_mode
        )
        source_amplitude = float(source_receiver_config.get("source_amplitude", multiplier_STF))

        # ASSEMBLY DICT MUST HAVE AT LEAST:
        self.wave_type            = assembly_dict["wave_type"]
        self.sample_dimensions    = assembly_dict["sample_dimensions"]
        self.transmitter_position = assembly_dict.get("transmitter_position", 0.0)
        self.receiver_position    = assembly_dict.get("receiver_position", 0.0)

        # For either "dds" or "block", we retrieve pzt and pla widths:
        if self.geometry_type.lower() == "dds":
            # Double-Direct-Shear geometry
            side1_params      = assembly_dict["side1_params"] 
            side2_params      = assembly_dict["side2_params"] 
            central_params    = assembly_dict["central_params"]
            if self.embedding_mode == "effective_surface":
                pzt_layer_width = 0.0
                pla_layer_width = 0.0
            else:
                pzt_layer_width = side1_params["pzt_layer_width"]
                pla_layer_width = side1_params["pla_layer_width"]

        elif self.geometry_type.lower() == "block":
            # Single-block geometry
            pzt_layer_width      = assembly_dict["pzt_layer_width"]     
            pla_layer_width      = assembly_dict["pla_layer_width"] 
            maximum_damping      = 0.

        else:
            raise ValueError(f"Unknown geometry_type: {self.geometry_type}.")

        # CREATE THE 1D GRID
        total_length = float(np.sum(self.sample_dimensions))
        if self.embedding_mode == "legacy_stack":
            total_length += 2 * pla_layer_width + 2 * pzt_layer_width
        grid_handler = Grid1D(
            cmin=minimum_velocity,
            fmax=frequency_cutoff,
            grid_len=total_length,
            ppt=self.points_per_wavelength
        )
        spatial_axis = grid_handler.spatial_axis
        dx           = grid_handler.dx
        num_x        = grid_handler.total_grid_points

        # DEFINE TIME AXIS
        sim_time_handler = SimulationTime(
            observed_time = observed_time,
            dx=dx,
            max_velocity  = maximum_velocity,
            max_alpha     = maximum_damping
        )
        simulation_time = sim_time_handler.simulation_time
        dt              = sim_time_handler.dt
        num_t           = sim_time_handler.num_t

        # -------------------------------------------------------
        # GEOMETRY-SPECIFIC: BUILD MODEL
        # -------------------------------------------------------
        if self.geometry_type.lower() == "dds":
            try:
                gouge_velocity_1 = assembly_dict["gouge_velocity_1"]
                gouge_velocity_2 = assembly_dict["gouge_velocity_2"]
                gouge_damping_1  = assembly_dict["gouge_damping_1"]
                gouge_damping_2  = assembly_dict["gouge_damping_2"]
            except KeyError:
                raise ValueError("`gouge_velocity_1` and `gouge_velocity_2` are required for DDS geometry.")
            
            central_params   = assembly_dict["central_params"]
            side1_params     = assembly_dict["side1_params"] 
            h_groove_central = central_params["h_grooves"]
            h_groove_side    = side1_params["h_grooves"]

            block_velocity = {
                "side1": side1_params["velocity" + self.wave_type],
                "central": central_params["velocity" + self.wave_type],
                "side2": side2_params["velocity" + self.wave_type],
            }
            density_enabled = bool(assembly_dict.get("density_model_enabled", False))
            if density_enabled:
                block_density = {
                    "side1": float(assembly_dict["side1_density"]),
                    "central": float(assembly_dict["central_density"]),
                    "side2": float(assembly_dict["side2_density"]),
                }
                gouge_density = (
                    float(assembly_dict["gouge_density_1"]),
                    float(assembly_dict["gouge_density_2"]),
                )
            else:
                # rho=1 reproduces the former density-normalised operator.
                block_density = {"side1": 1.0, "central": 1.0, "side2": 1.0}
                gouge_density = (1.0, 1.0)

            if self.embedding_mode == "effective_surface":
                # These values are constructor placeholders only; the
                # corresponding regions do not exist in the effective domain.
                pzt_velocity = block_velocity["side1"]
                pla_velocity = block_velocity["side1"]
            else:
                pzt_velocity = side1_params["pzt_velocity" + self.wave_type]
                pla_velocity = side1_params["pla_velocity" + self.wave_type]

            velocity_model_handler = VelocityModel1D_DDS(
                x=spatial_axis,  
                sample_dimensions=self.sample_dimensions,
                x_transmitter=self.transmitter_position,
                x_receiver=self.receiver_position,
                pzt_layer_width=pzt_layer_width,
                pla_layer_width=pla_layer_width,
                h_groove_side=h_groove_side,
                h_groove_central=h_groove_central,
                block_velocity=block_velocity,
                gouge_velocity=(gouge_velocity_1, gouge_velocity_2),
                gouge_damping=(gouge_damping_1, gouge_damping_2),
                pzt_velocity=pzt_velocity,
                pla_velocity=pla_velocity,
                include_transducer_layers=(self.embedding_mode == "legacy_stack"),
                block_density=block_density,
                gouge_density=gouge_density,
                pzt_density=float(assembly_dict.get("pzt_density", 1.0)),
                pla_density=float(assembly_dict.get("pla_density", 1.0)),
            )

        elif self.geometry_type.lower() == "block":
            steel_velocity = assembly_dict["velocity" + self.wave_type]
            pzt_velocity   = assembly_dict["pzt_velocity" + self.wave_type]
            pla_velocity   = assembly_dict["pla_velocity" + self.wave_type]
            gouge_velocity_1 = 0.
            gouge_velocity_2 = 0.
            gouge_damping_1  = 0.
            gouge_damping_2  = 0.

            velocity_model_handler = VelocityModel1D_SingleBlock(
                x=spatial_axis,
                sample_dimensions=self.sample_dimensions,
                x_transmitter=self.transmitter_position,
                x_receiver=self.receiver_position,
                pzt_layer_width=pzt_layer_width,
                pla_layer_width=pla_layer_width,
                block_velocity=steel_velocity,
                pzt_velocity=pzt_velocity,
                pla_velocity=pla_velocity,
                block_density=float(assembly_dict.get("density", 1.0)),
                pzt_density=float(assembly_dict.get("pzt_density", 1.0)),
                pla_density=float(assembly_dict.get("pla_density", 1.0)),
            )

        velocity_model = velocity_model_handler.velocity_array
        damping_model = velocity_model_handler.damping_array
        density_model = velocity_model_handler.density_array

        # BUILD SOURCE
        if self.embedding_mode == "effective_surface":
            default_source_position = 0.0
        else:
            default_source_position = (
                pzt_layer_width
                + position2edge_transmitter * pzt_layer_width
                + pla_layer_width
            )
        self.transmitter_position_relative = float(
            source_receiver_config.get("source_position_cm", default_source_position)
        )
        default_radius = 4 if self.embedding_mode == "effective_surface" else ceil(
            radius_factor_transmitter * (pzt_layer_width / 2) / dx
        )
        radius_transmitter = int(source_receiver_config.get(
            "source_radius_cells",
            source_receiver_config.get("interpolation_radius_cells", default_radius),
        ))
        extension_transmitter = float(source_receiver_config.get(
            "source_extension_cm",
            0.0 if self.embedding_mode == "effective_surface"
            else spreading_factor_transmitter * pzt_layer_width,
        ))

        source_handler = Source1D(
            stf_time=stf_handler.metadata["time_ax_waveform"],
            stf_waveform=stf_handler.waveform_data * source_amplitude,
            position=self.transmitter_position_relative,
            radius=radius_transmitter,
            extension=extension_transmitter,
            pzt_layer_width=pzt_layer_width,
            normalization=source_normalization,
            boundary_mode=spatial_boundary_mode,
        )
        
        source_handler.interpolate_time_function(dt=dt, simulation_time=simulation_time)
        source_handler.create_spatial_function(spatial_axis=spatial_axis, dx=dx)

        # BUILD RECEIVER
        if self.embedding_mode == "effective_surface":
            default_receiver_position = total_length
        else:
            default_receiver_position = (
                total_length
                - pzt_layer_width
                - position2edge_receiver * pzt_layer_width
                - pla_layer_width
            )
        self.receiver_position_relative = float(
            source_receiver_config.get("receiver_position_cm", default_receiver_position)
        )
        default_receiver_radius = (
            4 if self.embedding_mode == "effective_surface"
            else ceil(radius_factor_receiver * (pzt_layer_width / 2) / dx)
        )
        radius_receiver = int(source_receiver_config.get(
            "receiver_radius_cells",
            source_receiver_config.get(
                "interpolation_radius_cells", default_receiver_radius
            ),
        ))
        extension_receiver = float(source_receiver_config.get(
            "receiver_extension_cm",
            0.0 if self.embedding_mode == "effective_surface"
            else spreading_factor_receiver * pzt_layer_width,
        ))

        receiver_handler = Receiver1D(
            position=self.receiver_position_relative,
            radius=radius_receiver,
            extension=extension_receiver,
            pzt_layer_width=pzt_layer_width,
            normalization=receiver_normalization,
            boundary_mode=spatial_boundary_mode,
        )
        receiver_handler.create_spatial_function(spatial_axis=spatial_axis, dx=dx)

        # plt.plot(receiver_handler.spatial_function)
        # plt.plot(source_handler.spatial_function)
        # plt.show()

        # FORWARD MODEL: SECOND-ORDER FINITE DIFFERENCES
        if self.store_wavefield:
            wavefield_forward = finite_difference_1d_kelvin_voigt(
                num_x=num_x,
                delta_x=dx,
                num_t=num_t,
                delta_t=dt,
                source_spatial_function=source_handler.spatial_function,
                source_time_function=source_handler.time_function,
                velocity_model=velocity_model,
                damping_model=damping_model,
                density_model=density_model,
                boundary_condition=self.boundary_condition,
                boundary_impedance=self.boundary_impedance,
                source_injection=self.source_injection,
            )

            # EXTRACT THE DECLARED PHYSICAL OBSERVABLE AT THE RECEIVER
            simulated_waveform = sample_receiver(
                wavefield_forward,
                receiver_handler.spatial_function,
                quantity=self.receiver_quantity,
                delta_t=dt,
                delta_x=dx,
                velocity_model=velocity_model,
                damping_model=damping_model,
                density_model=density_model,
            )
        else:
            wavefield_forward = None
            simulated_waveform = finite_difference_1d_receiver_trace(
                num_x=num_x,
                delta_x=dx,
                num_t=num_t,
                delta_t=dt,
                source_spatial_function=source_handler.spatial_function,
                source_time_function=source_handler.time_function,
                receiver_spatial_function=receiver_handler.spatial_function,
                receiver_quantity=self.receiver_quantity,
                velocity_model=velocity_model,
                damping_model=damping_model,
                density_model=density_model,
                boundary_condition=self.boundary_condition,
                boundary_impedance=self.boundary_impedance,
                source_injection=self.source_injection,
            )
        # INTERPOLATE ONTO OBSERVED TIME AXIS
        synthetic_waveform = np.interp(observed_time, simulation_time, simulated_waveform)
        raw_synthetic_waveform = synthetic_waveform.copy()

        # # NORMALIZE IF REQUESTED
        scale = 1.0
        if normalize_waveform:
            scale_interval = (
                misfit_interval
                if normalization_interval is None
                else normalization_interval
            )
            scale = fit_trace_scale(
                observed_waveform=observed_waveform,
                synthetic_waveform=synthetic_waveform,
                misfit_interval=scale_interval,
            )
            synthetic_waveform *= scale

        #-------------------------------------------
        # COMPUTE MISFIT
        #-------------------------------------------
        misfit = self.compute_misfit(
            observed_waveform   = observed_waveform,
            synthetic_waveform  = synthetic_waveform,
            misfit_interval     = misfit_interval
        )


        # Store results in the instance so we can reuse them
        self.synthetic_waveform     = synthetic_waveform
        self.raw_synthetic_waveform = raw_synthetic_waveform
        self.trace_scale            = float(scale)
        self.objective_components   = relative_waveform_objective(
            observed_waveform,
            synthetic_waveform,
            misfit_interval,
            spectral_gamma=1.0,
        )
        self.wavefield_forward      = wavefield_forward
        self.velocity_model_handler = velocity_model_handler
        self.sim_time_handler       = sim_time_handler
        self.grid_handler           = grid_handler
        self.source_handler         = source_handler
        self.receiver_handler       = receiver_handler
        self.misfit                 = misfit

        if enable_plotting:
            if self.geometry_type == "dds":
                damping_label    = str(round(gouge_damping_1,5)).replace(".",",")
                velocity_label_1 = str(round(1e4*gouge_velocity_1)).replace(".",",")
                velocity_label_2 = str(round(1e4*gouge_velocity_2)).replace(".",",")
                misfit_label = _filename_decimal(misfit, 4)
                thickness_1_mm = 10.0 * float(assembly_dict["gouge_thickness_1"])
                thickness_2_mm = 10.0 * float(assembly_dict["gouge_thickness_2"])
                if np.isclose(gouge_velocity_1, gouge_velocity_2):
                    velocity_part = f"vel_{velocity_label_1}"
                else:
                    velocity_part = (
                        f"vel1_{velocity_label_1}_vel2_{velocity_label_2}"
                    )
                if np.isclose(thickness_1_mm, thickness_2_mm):
                    thickness_part = ""
                else:
                    h1_label = f"{thickness_1_mm:.3f}".replace(".", ",")
                    h2_label = f"{thickness_2_mm:.3f}".replace(".", ",")
                    thickness_part = f"_h1_{h1_label}mm_h2_{h2_label}mm"
                label = (
                    f"{velocity_part}{thickness_part}_damping_{damping_label}"
                    f"_misfit_{misfit_label}"
                )
                plot_output_name = f"{plot_name_prefix}{label}"
                plot_output_path = plot_output_path / plot_output_name        


            self.plotter.plot_simulation_waveform(
            t=observed_time,
            sp_simulated=synthetic_waveform,
            sp_recorded=observed_waveform,
            misfit_interval=misfit_interval,
            outfile_path=plot_output_path
        )

            model_output_name = plot_output_path.name + "_velocity_model"
            model_output_path = plot_output_path.parent / model_output_name
            # self.velocity_model_handler.plot(model=self.velocity_model_handler.velocity_array, outfile_path=model_output_path)

            # model_output_name = plot_output_path.name + "_damping_model"
            # model_output_path = plot_output_path.parent / model_output_name
            # self.velocity_model_handler.plot(model=self.velocity_model_handler.damping_array, outfile_path=model_output_path)

        # if make_movie:
        #     self.plotter.make_movie_from_simulation(
        #         outfile_path=movie_output_path,
        #         x=spatial_axis,
        #         t=simulation_time,
        #         sp_field=wavefield_forward,
        #         sp_recorded=simulated_waveform,
        #         sample_dimensions=self.sample_dimensions,
        #         idx_dict=idx_dict,
        #     )

        return (
            synthetic_waveform,
            wavefield_forward,
            velocity_model_handler,
            sim_time_handler,
            grid_handler,
            source_handler,
            receiver_handler,
        )

    def write_direct_arrival_spectral_diagnostics(
        self,
        *,
        output_directory: Optional[pathlib.Path] = None,
        frequency_min_mhz: float = 0.5,
        frequency_max_mhz: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Write the two spectral comparisons for the final direct arrival.

        The first compares the effective STF with the recorded direct packet.
        The second compares recorded and simulated direct packets on exactly
        the same time samples used by the waveform misfit.
        """
        observed_time = np.asarray(self.observed_time, dtype=float).reshape(-1)
        observed = np.asarray(self.observed_waveform, dtype=float).reshape(-1)
        synthetic = np.asarray(self.synthetic_waveform, dtype=float).reshape(-1)
        direct_mask = sample_interval_mask(self.misfit_interval, observed_time.size)
        if np.count_nonzero(direct_mask) < 4:
            raise ValueError("the direct-arrival interval needs at least four samples")

        direct_time = observed_time[direct_mask]
        recorded_direct = observed[direct_mask]
        simulated_direct = synthetic[direct_mask]
        stf_time = np.asarray(
            self.stf_handler.metadata["time_ax_waveform"], dtype=float
        ).reshape(-1)
        stf_waveform = np.asarray(self.stf_handler.waveform_data, dtype=float).reshape(-1)

        observed_nyquist = 0.5 / float(np.median(np.diff(direct_time)))
        stf_nyquist = 0.5 / float(np.median(np.diff(stf_time)))
        maximum = min(observed_nyquist, stf_nyquist)
        if frequency_max_mhz is not None:
            maximum = min(maximum, float(frequency_max_mhz))
        minimum = float(frequency_min_mhz)
        if not minimum < maximum:
            raise ValueError(
                "the requested spectral band is empty after the Nyquist limit"
            )

        stf_vs_recorded = compare_spectral_shapes(
            stf_time,
            stf_waveform,
            direct_time,
            recorded_direct,
            frequency_min_mhz=minimum,
            frequency_max_mhz=maximum,
        )
        recorded_vs_simulated = compare_spectral_shapes(
            direct_time,
            recorded_direct,
            direct_time,
            simulated_direct,
            frequency_min_mhz=minimum,
            frequency_max_mhz=maximum,
        )

        if output_directory is not None:
            output_directory = pathlib.Path(output_directory)
            plot_spectral_comparison(
                stf_vs_recorded,
                first_label="Effective wavelet",
                second_label="Recorded direct arrival",
                title="Effective wavelet vs recorded direct-arrival spectrum",
                output_path=(
                    output_directory
                    / "spectrum_effective_wavelet_vs_recorded_direct.png"
                ),
                show_distance=False,
            )
            plot_spectral_comparison(
                recorded_vs_simulated,
                first_label="Recorded direct arrival",
                second_label="Simulated direct arrival",
                title="Recorded vs simulated direct-arrival spectrum",
                output_path=(
                    output_directory
                    / "spectrum_recorded_vs_simulated_direct.png"
                ),
                show_distance=True,
            )

        self.spectral_shape_distance = recorded_vs_simulated.shape_distance
        self.stf_recorded_spectral_shape_distance = stf_vs_recorded.shape_distance
        self.spectral_diagnostics = {
            "frequency_min_mhz": minimum,
            "frequency_max_mhz": maximum,
            "recorded_vs_simulated_distance": recorded_vs_simulated.shape_distance,
            "stf_vs_recorded_distance": stf_vs_recorded.shape_distance,
        }
        return dict(self.spectral_diagnostics)


    def compute_misfit(self,
            observed_waveform : np.ndarray,
            synthetic_waveform: np.ndarray,
            misfit_interval   : slice | np.ndarray,
            *,
            equal_segment_weight: bool = False,
            eps: float = 1e-12
    ) -> float:
        """
        Squared relative-L2 misfit on the selected waveform interval.

        The returned quantity is ``||d-s||² / ||d||²``.  Unlike the former
        unnormalised norm, it is dimensionless and can be compared between
        records with different amplitudes.  When the synthetic trace has first
        been least-squares scaled, this is the normalised shape objective
        ``1-rho²``; with a fixed experiment-wide scale it retains amplitude
        evolution as part of the objective.

        Parameters
        ----------
        observed_waveform, synthetic_waveform : 1‑D ndarray
        misfit_interval : slice | 1‑D ndarray of indices
            Either a simple slice (contiguous window) or the array of indices returned
            by `compute_misfit_interval`, which may contain several disjoint segments.
        equal_segment_weight : bool, optional
            *False* (default) – the classic behaviour: every individual **sample**
            inside `misfit_interval` is equally weighted.
            *True*  – every **segment** (direct arrival, reflection, …) is given the
            same total weight, so a low‑amplitude reflection can influence the
            inversion as much as the stronger direct wave.
        eps : float, optional
            Small constant to protect against divisions by zero.
        """
        # ------------------------------------------------------------------ helpers
        def _split_into_segments(idxs: np.ndarray) -> list[np.ndarray]:
            """Split an index array into a list of contiguous blocks."""
            idxs = np.sort(np.unique(idxs))
            gaps = np.where(np.diff(idxs) > 1)[0]
            return np.split(idxs, gaps + 1)

        # ------------------------------------------------------------------ guards
        if synthetic_waveform.ndim != 1 or observed_waveform.ndim != 1:
            raise ValueError("Waveforms must be 1‑D arrays")
        if synthetic_waveform.size != observed_waveform.size:
            raise ValueError("Synthetic and observed waveforms must have same length")
        idx = _waveform_interval_indices(
            misfit_interval,
            observed_waveform.size,
        )

        if equal_segment_weight:
            values = []
            for segment in _split_into_segments(idx):
                values.append(
                    relative_waveform_objective(
                        observed_waveform,
                        synthetic_waveform,
                        segment,
                        spectral_gamma=1.0,
                        eps=eps,
                    )["total"]
                )
            return float(np.mean(values))

        return float(
            relative_waveform_objective(
                observed_waveform,
                synthetic_waveform,
                idx,
                spectral_gamma=1.0,
                eps=eps,
            )["total"]
        )
      

    def compute_amplitude_and_phase_spectrum(self, 
                                    observed_time     : np.ndarray,
                                    synthetic_waveform: np.ndarray,
                                    ) -> tuple:
            
            n_samples = len(observed_time)
            dt = observed_time[1]-observed_time[0]
            self.frequencies = np.fft.rfftfreq(n_samples, d=dt)

            # Compute the FFT along the sample axis
            fft_data = np.fft.rfft(synthetic_waveform)

            # Compute amplitude and phase
            self.amplitude_spectrum = np.abs(fft_data)
            self.phase_spectrum = np.angle(fft_data)

            return self.frequencies, self.amplitude_spectrum, self.phase_spectrum

    def _require_legacy_adjoint_configuration(self) -> None:
        """Prevent use of the old adjoint with a different forward operator."""
        incompatible = []
        if self.boundary_condition != "legacy_fixed":
            incompatible.append(f"boundary_condition={self.boundary_condition!r}")
        if self.source_injection != "legacy_increment":
            incompatible.append(f"source_injection={self.source_injection!r}")
        if self.receiver_quantity != "displacement":
            incompatible.append(f"receiver_quantity={self.receiver_quantity!r}")
        if incompatible:
            details = ", ".join(incompatible)
            raise NotImplementedError(
                "The current local adjoint is defined only for the historical "
                "fixed-end, displacement-increment/displacement-sampling operator. "
                f"The forward run used {details}. Validate or calibrate this "
                "configuration with the deterministic reference-block workflow "
                "instead of invoking the legacy adjoint."
            )

    def _require_local_model_adjoint_configuration(self) -> None:
        """Accept only model inversions whose discrete adjoint is verified."""
        if self.embedding_mode == "legacy_stack":
            self._require_legacy_adjoint_configuration()
            return

        incompatible = []
        if self.boundary_condition != "free":
            incompatible.append(f"boundary_condition={self.boundary_condition!r}")
        if self.source_injection != "body_force":
            incompatible.append(f"source_injection={self.source_injection!r}")
        if self.receiver_quantity != "velocity":
            incompatible.append(f"receiver_quantity={self.receiver_quantity!r}")
        if self.absorbing:
            incompatible.append("absorbing=True")
        if incompatible:
            raise NotImplementedError(
                "effective_surface model inversion is verified only for the "
                "calibrated free/body_force/velocity operator; got "
                + ", ".join(incompatible)
            )
        
    def run_local_inversion(
        self,
        n_iterations: int,
        dc_max_start: float = 0,
        dc_threshold: float = 0,
        da_max_start: float = 0,
        da_threshold: float = 0,
        dw_max_start: float = 0,   # ignored in this fixed-source version
        dw_threshold: float = 0,   # ignored in this fixed-source version
        stf_duration: float = 4,   # unused here, kept for signature compatibility
        ds_max_start: float = 0,   # ignored in this fixed-source version
        ds_threshold: float = 0,   # ignored in this fixed-source version
        reduce_factor: float = 2,
        misfit_threshold: float = 1.0e-8,
        normalize_waveform: bool = True,
        amplitude_scale_mode: Optional[str] = None,
        fixed_trace_scale: Optional[float] = None,
        spectral_gamma: float = 1.0,
        objective_taper_fraction: float = 0.0,
        update_strategy: str = "joint",
        enable_plotting: bool = True,
        make_movie: bool = False,
        plot_output_path: Optional[str] = None,
        movie_output_path: Optional[str] = "simulation_movie.mp4",
        retain_wavefield: bool = True,
        plot_name_prefix: Optional[str] = None,
    ) -> None:
        """
        Local waveform inversion for nodal velocity and damping in the selected update region,
        using a discrete adjoint matched to the Kelvin–Voigt finite-difference stencil.

        In the current workflow source and receiver functions are fixed.
        ``amplitude_scale_mode='fixed'`` preserves one reference calibration
        across the waveform sequence; ``'profiled'`` retains the former
        per-model shape-only behaviour.  Set ``retain_wavefield=False`` when
        only the accepted model and synthetic trace are needed: the complete
        final wavefield is then released before returning.  This does not
        change the inversion result and is appropriate for sequential or
        parallel waveform processing without movie generation.

        ``update_strategy='joint'`` preserves the historical simultaneous
        velocity/damping update.  ``'alternating'`` changes one parameter
        block per iteration and accepts it independently.  ``'cross_talk'``
        evaluates velocity-only, damping-only and joint trials, stores their
        mixed interaction, and accepts only the best improving candidate.
        """

        if self.synthetic_waveform is None:
            raise RuntimeError("No forward-simulation results. Call `forward_simulation()` first.")
        self._require_local_model_adjoint_configuration()
        use_effective_adjoint = self.embedding_mode == "effective_surface"
        density_before_shrink = self.velocity_model_handler.density_array.copy()

        # A local inversion recomputes its initial forward field from the
        # carried model.  The complete field retained by the preceding record
        # is therefore stale and would otherwise occupy one extra (nt, nx)
        # array throughout the new inversion.
        self.wavefield_forward = None

        # ------------------------------------------------------------------
        # 0) OPTIONAL: SHRINK GRID IF GOUGE SHRUNK
        # ------------------------------------------------------------------
        (
            self.assembly_dict,
            new_idx_dict,
            new_x,
            new_v,
            new_a,
            new_src_sp,
            new_rcv_sp,
            removed_idx,
        ) = shrink_grid_if_gouge_shrinks(
            assembly_dict=self.assembly_dict,
            idx_dict=self.velocity_model_handler.idx_dict,
            spatial_axis=self.grid_handler.spatial_axis,
            dx=self.grid_handler.dx,
            velocity_model=self.velocity_model_handler.velocity_array,
            damping_model=self.velocity_model_handler.damping_array,
            source_spatial_function=self.source_handler.spatial_function,
            receiver_spatial_function=self.receiver_handler.spatial_function,
        )

        if removed_idx.size > 0:
            self.grid_handler.spatial_axis = new_x
            self.grid_handler.total_grid_points = new_x.size

            self.velocity_model_handler.velocity_array = new_v
            self.velocity_model_handler.damping_array = new_a
            self.velocity_model_handler.density_array = np.delete(
                density_before_shrink, removed_idx
            )
            self.velocity_model_handler.idx_dict = new_idx_dict
            self.velocity_model_handler.x = new_x
            self.source_handler.spatial_function = new_src_sp
            self.receiver_handler.spatial_function = new_rcv_sp

        # ------------------------------------------------------------------
        # 1) RE-BIND ALL locals AFTER SHRINK
        # ------------------------------------------------------------------
        absorbing = self.absorbing

        observed_time = self.observed_time
        observed_waveform = self.observed_waveform
        misfit_interval = self.misfit_interval
        num_x = self.grid_handler.total_grid_points
        delta_x = self.grid_handler.dx
        delta_t = self.sim_time_handler.dt
        num_t = self.sim_time_handler.num_t
        simulation_time = self.sim_time_handler.simulation_time
        spatial_axis = self.grid_handler.spatial_axis

        initial_source_time_function = self.source_handler.time_function.copy()
        initial_source_spatial_function = self.source_handler.spatial_function.copy()
        initial_receiver_spatial_function = self.receiver_handler.spatial_function.copy()
        initial_velocity_model = self.velocity_model_handler.velocity_array.copy()
        initial_damping_model = self.velocity_model_handler.damping_array.copy()
        # Density is part of the forward operator, but is deliberately fixed
        # during this local inversion.  It comes from the global material
        # search and is differentiated through, not updated.
        initial_density_model = self.velocity_model_handler.density_array.copy()
        idx_dict = self.velocity_model_handler.idx_dict

        if amplitude_scale_mode is None:
            amplitude_scale_mode = "profiled" if normalize_waveform else "none"
        if amplitude_scale_mode not in {"profiled", "fixed", "none"}:
            raise ValueError(
                "amplitude_scale_mode must be 'profiled', 'fixed', or 'none'"
            )
        if amplitude_scale_mode == "profiled" and float(spectral_gamma) != 1.0:
            raise ValueError(
                "profiled amplitude is compatible only with spectral_gamma=1; "
                "use a fixed experiment-wide trace scale for spectral reweighting"
            )
        update_strategy = str(update_strategy).strip().lower()
        if update_strategy not in {"joint", "alternating", "cross_talk"}:
            raise ValueError(
                "update_strategy must be 'joint', 'alternating', or 'cross_talk'"
            )

        # ------------------------------------------------------------------
        # 2) REGIONS TO UPDATE
        # ------------------------------------------------------------------
        if self.geometry_type == "dds":
            regions_to_update = np.concatenate([
                idx_dict["groove_sb1"],
                idx_dict["gouge_1"],
                idx_dict["groove_cb1"],
                idx_dict["groove_cb2"],
                idx_dict["gouge_2"],
                idx_dict["groove_sb2"],
            ])
        elif self.geometry_type == "block":
            regions_to_update = np.concatenate([
                idx_dict["pzt_1"],
                idx_dict["steel_block"][:len(idx_dict["pzt_1"])],
                idx_dict["steel_block"][-len(idx_dict["pzt_1"]):],
                idx_dict["pzt_2"]
            ])
        else:
            raise ValueError(f"Unknown geometry_type: {self.geometry_type}")

        # ------------------------------------------------------------------
        # 3) INITIAL FORWARD + MISFIT
        # ------------------------------------------------------------------
        initial_wavefield_forward = pseudospectral_1D_damped(
            num_x=num_x,
            delta_x=delta_x,
            num_t=num_t,
            delta_t=delta_t,
            source_spatial_function=initial_source_spatial_function,
            source_time_function=initial_source_time_function,
            velocity_model=initial_velocity_model,
            damping_model=initial_damping_model,
            density_model=initial_density_model,
            absorbing=absorbing,
            boundary_condition=self.boundary_condition,
            boundary_impedance=self.boundary_impedance,
            source_injection=self.source_injection,
        )

        initial_simulated_waveform = sample_receiver(
            initial_wavefield_forward,
            initial_receiver_spatial_function,
            quantity=self.receiver_quantity,
            delta_t=delta_t,
            delta_x=delta_x,
            velocity_model=initial_velocity_model,
            damping_model=initial_damping_model,
            density_model=initial_density_model,
        )
        initial_synthetic_waveform = np.interp(
            observed_time,
            simulation_time,
            initial_simulated_waveform
        )

        if amplitude_scale_mode == "profiled":
            best_trace_scale = fit_trace_scale(
                observed_waveform=observed_waveform,
                synthetic_waveform=initial_synthetic_waveform,
                misfit_interval=misfit_interval,
            )
        elif amplitude_scale_mode == "fixed":
            if fixed_trace_scale is None:
                fixed_trace_scale = getattr(
                    self,
                    "reference_trace_scale",
                    getattr(self, "trace_scale", None),
                )
            if fixed_trace_scale is None:
                raise ValueError(
                    "fixed amplitude scaling requires fixed_trace_scale or "
                    "self.reference_trace_scale"
                )
            best_trace_scale = float(fixed_trace_scale)
            if not np.isfinite(best_trace_scale) or best_trace_scale <= 0.0:
                raise ValueError("fixed_trace_scale must be positive and finite")
        else:
            best_trace_scale = 1.0
        initial_synthetic_waveform *= best_trace_scale

        best_objective = relative_waveform_objective(
            observed_waveform,
            initial_synthetic_waveform,
            misfit_interval,
            spectral_gamma=spectral_gamma,
            taper_fraction=objective_taper_fraction,
        )
        initial_misfit = float(best_objective["total"])

        # ------------------------------------------------------------------
        # STORE "BEST" STATE
        # ------------------------------------------------------------------
        best_velocity_model = initial_velocity_model.copy()
        best_damping_model = initial_damping_model.copy()

        # Fixed source/receiver in this version
        best_source_time_function = initial_source_time_function.copy()
        best_source_spatial_function = initial_source_spatial_function.copy()
        best_receiver_spatial_function = initial_receiver_spatial_function.copy()

        # The propagator returns a new array and the inversion never mutates a
        # stored forward field, so an additional full-wavefield copy is not
        # needed here.
        best_wavefield_forward = initial_wavefield_forward
        best_synthetic_waveform = initial_synthetic_waveform.copy()
        best_misfit = initial_misfit
        previous_misfit = initial_misfit
        local_iteration_history = [{
            "iteration": 0,
            "trial_misfit": initial_misfit,
            "best_misfit": initial_misfit,
            "accepted": True,
            "update_strategy": update_strategy,
            "parameter_block": "initial",
            "velocity_only_misfit": None,
            "damping_only_misfit": None,
            "joint_misfit": None,
            "cross_talk_interaction": None,
            "velocity_step_limit": float(dc_max_start),
            "damping_step_limit": float(da_max_start),
        }]
        del initial_wavefield_forward

        dc_max = dc_max_start
        da_max = da_max_start

        # ------------------------------------------------------------------
        # PRECOMPUTE MASK
        # ------------------------------------------------------------------
        mask = np.zeros_like(observed_waveform, dtype=bool)
        mask[misfit_interval] = True

        updating = True
        activate_velocity = dc_max > 0
        activate_damping = da_max > 0

        # ----------------------------------------------------------
        # NODEWISE VELOCITY BOUNDS
        # Groove regions are capped by the adjacent host-block velocity
        # ----------------------------------------------------------
        v_lower = np.full(num_x, -np.inf, dtype=float)
        v_upper = np.full(num_x,  np.inf, dtype=float)

        if self.geometry_type == "dds":
            side1_params   = self.assembly_dict["side1_params"]
            side2_params   = self.assembly_dict["side2_params"]
            central_params = self.assembly_dict["central_params"]

            v_side1   = side1_params["velocity" + self.wave_type]
            v_side2   = side2_params["velocity" + self.wave_type]
            v_central = central_params["velocity" + self.wave_type]

            # optional lower bound; keep your current logic or replace with a physical one
            vmin_update = max(1e-12, 0.25 * np.min(initial_velocity_model[regions_to_update]))

            # lower bound on all updated regions
            v_lower[regions_to_update] = vmin_update

            # pure gouge layers: choose what upper bound you want there
            # here I keep the same cap as the neighboring side blocks
            v_upper[idx_dict["gouge_1"]] = v_side1
            v_upper[idx_dict["gouge_2"]] = v_side2

            # grooves capped by the relative host block
            v_upper[idx_dict["groove_sb1"]] = v_side1
            v_upper[idx_dict["groove_cb1"]] = v_central
            v_upper[idx_dict["groove_cb2"]] = v_central
            v_upper[idx_dict["groove_sb2"]] = v_side2

        elif self.geometry_type == "block":
            vmin_update = max(1e-12, 0.25 * np.min(initial_velocity_model[regions_to_update]))
            vmax_update = np.max(initial_velocity_model[regions_to_update])

            v_lower[regions_to_update] = vmin_update
            v_upper[regions_to_update] = vmax_update

        def _scaled_candidate_objective(simulated_waveform):
            synthetic_waveform = np.interp(
                observed_time,
                simulation_time,
                simulated_waveform,
            )
            if amplitude_scale_mode == "profiled":
                trace_scale = fit_trace_scale(
                    observed_waveform=observed_waveform,
                    synthetic_waveform=synthetic_waveform,
                    misfit_interval=misfit_interval,
                )
            elif amplitude_scale_mode == "fixed":
                trace_scale = best_trace_scale
            else:
                trace_scale = 1.0
            synthetic_waveform *= trace_scale
            objective = relative_waveform_objective(
                observed_waveform,
                synthetic_waveform,
                misfit_interval,
                spectral_gamma=spectral_gamma,
                taper_fraction=objective_taper_fraction,
            )
            return synthetic_waveform, float(trace_scale), objective

        def _evaluate_receiver_only_candidate(velocity_model, damping_model):
            simulated_waveform = finite_difference_1d_receiver_trace(
                num_x=num_x,
                delta_x=delta_x,
                num_t=num_t,
                delta_t=delta_t,
                source_spatial_function=best_source_spatial_function,
                source_time_function=best_source_time_function,
                receiver_spatial_function=best_receiver_spatial_function,
                receiver_quantity=self.receiver_quantity,
                velocity_model=velocity_model,
                damping_model=damping_model,
                density_model=initial_density_model,
                boundary_condition=self.boundary_condition,
                boundary_impedance=self.boundary_impedance,
                source_injection=self.source_injection,
            )
            return _scaled_candidate_objective(simulated_waveform)

        def _evaluate_full_candidate(velocity_model, damping_model):
            wavefield = pseudospectral_1D_damped(
                num_x=num_x,
                delta_x=delta_x,
                num_t=num_t,
                delta_t=delta_t,
                source_spatial_function=best_source_spatial_function,
                source_time_function=best_source_time_function,
                velocity_model=velocity_model,
                damping_model=damping_model,
                density_model=initial_density_model,
                absorbing=absorbing,
                boundary_condition=self.boundary_condition,
                boundary_impedance=self.boundary_impedance,
                source_injection=self.source_injection,
            )
            simulated_waveform = sample_receiver(
                wavefield,
                best_receiver_spatial_function,
                quantity=self.receiver_quantity,
                delta_t=delta_t,
                delta_x=delta_x,
                velocity_model=velocity_model,
                damping_model=damping_model,
                density_model=initial_density_model,
            )
            synthetic_waveform, trace_scale, objective = (
                _scaled_candidate_objective(simulated_waveform)
            )
            return wavefield, synthetic_waveform, trace_scale, objective

        for iteration in range(n_iterations):
            velocity_available = bool(
                activate_velocity and dc_max >= dc_threshold
            )
            damping_available = bool(
                activate_damping and da_max >= da_threshold
            )
            if not velocity_available and not damping_available:
                print("Stopping: active step sizes below thresholds.")
                break

            if update_strategy == "alternating":
                parameter_block = "velocity" if iteration % 2 == 0 else "damping"
                if parameter_block == "velocity" and not velocity_available:
                    parameter_block = "damping"
                elif parameter_block == "damping" and not damping_available:
                    parameter_block = "velocity"
            else:
                parameter_block = "joint"

            if updating:
                # ----------------------------------------------------------
                # DATA RESIDUAL ON OBSERVED GRID
                # ----------------------------------------------------------
                objective_with_gradient = relative_waveform_objective(
                    observed_waveform,
                    best_synthetic_waveform,
                    misfit_interval,
                    spectral_gamma=spectral_gamma,
                    taper_fraction=objective_taper_fraction,
                    return_gradient=True,
                )
                trace_gradient_obs = (
                    best_trace_scale
                    * objective_with_gradient["gradient_predicted"]
                )

                # ----------------------------------------------------------
                # MAP RESIDUAL TO SIMULATION-TIME GRID
                # ----------------------------------------------------------
                if use_effective_adjoint:
                    # Exact transpose of both interpolation and velocity
                    # receiver sampling, followed by reverse differentiation
                    # of the free-boundary leapfrog recurrence.
                    trace_gradient_sim = interpolation_adjoint(
                        simulation_time,
                        observed_time,
                        trace_gradient_obs,
                    )
                    dJdu = receiver_adjoint(
                        trace_gradient_sim,
                        best_receiver_spatial_function,
                        quantity=self.receiver_quantity,
                        delta_t=delta_t,
                    )
                    gradient_vel, gradient_damp, adjoint_wavefield = (
                        discrete_model_gradient_1d_kelvin_voigt(
                            wavefield=best_wavefield_forward,
                            wavefield_gradient=dJdu,
                            velocity_model=best_velocity_model,
                            damping_model=best_damping_model,
                            density_model=initial_density_model,
                            delta_x=delta_x,
                            delta_t=delta_t,
                            boundary_condition=self.boundary_condition,
                            consume_wavefield_gradient=True,
                        )
                    )
                else:
                    residual_sim = residual_to_simulation_time(
                        observed_time=observed_time,
                        simulation_time=simulation_time,
                        observed_residual=trace_gradient_obs,
                    )
                    dJdu = build_adjoint_data_gradient(
                        residual_sim=residual_sim,
                        receiver_spatial_function=best_receiver_spatial_function,
                    )
                    adjoint_wavefield = discrete_adjoint_1D_kelvin_voigt(
                        dJdu=dJdu,
                        velocity_model=best_velocity_model,
                        damping_model=best_damping_model,
                        delta_x=delta_x,
                        delta_t=delta_t,
                    )
                    gradient_vel, gradient_damp = compute_discrete_gradients_velocity_damping(
                        wavefield_forward=best_wavefield_forward,
                        adjoint_wavefield=adjoint_wavefield,
                        velocity_model=best_velocity_model,
                        delta_x=delta_x,
                        delta_t=delta_t,
                    )

                if activate_velocity:
                    max_vel_grad = np.abs(gradient_vel[regions_to_update]).max()

                if activate_damping:
                    max_damp_grad = np.abs(gradient_damp[regions_to_update]).max()

                # The adjoint field is not used by the line-search update.
                # Release it before allocating the trial forward wavefield.
                del adjoint_wavefield
                if use_effective_adjoint:
                    del dJdu, trace_gradient_sim

            # ------------------------------------------------------------------
            # FORM THE TWO PARAMETER-BLOCK PROPOSALS INDEPENDENTLY
            # ------------------------------------------------------------------
            velocity_proposal = best_velocity_model.copy()
            damping_proposal = best_damping_model.copy()

            if velocity_available:
                step_size_vel = dc_max / (max_vel_grad + 1e-15)
                direction = -1.0 if use_effective_adjoint else 1.0
                velocity_proposal[regions_to_update] += (
                    direction * step_size_vel * gradient_vel[regions_to_update]
                )

                velocity_proposal[regions_to_update] = np.clip(
                    velocity_proposal[regions_to_update],
                    v_lower[regions_to_update],
                    v_upper[regions_to_update],
                )

            if damping_available:
                step_size_damp = da_max / (max_damp_grad + 1e-15)
                direction = -1.0 if use_effective_adjoint else 1.0
                damping_proposal[regions_to_update] += (
                    direction * step_size_damp * gradient_damp[regions_to_update]
                )

                damping_proposal[regions_to_update] = np.clip(
                    damping_proposal[regions_to_update],
                    0.0,
                    None,
                )

            velocity_only_misfit = None
            damping_only_misfit = None
            joint_misfit = None
            cross_talk_interaction = None
            selected_candidate = parameter_block
            wavefield_forward_updated = None

            if update_strategy == "joint":
                updated_velocity_model = (
                    velocity_proposal if velocity_available
                    else best_velocity_model
                )
                updated_damping_model = (
                    damping_proposal if damping_available
                    else best_damping_model
                )
                (
                    wavefield_forward_updated,
                    updated_synthetic_waveform,
                    updated_trace_scale,
                    updated_objective,
                ) = _evaluate_full_candidate(
                    updated_velocity_model,
                    updated_damping_model,
                )
                updated_misfit = float(updated_objective["total"])
                joint_misfit = updated_misfit

            elif update_strategy == "alternating":
                if parameter_block == "velocity":
                    updated_velocity_model = velocity_proposal
                    updated_damping_model = best_damping_model
                else:
                    updated_velocity_model = best_velocity_model
                    updated_damping_model = damping_proposal
                (
                    wavefield_forward_updated,
                    updated_synthetic_waveform,
                    updated_trace_scale,
                    updated_objective,
                ) = _evaluate_full_candidate(
                    updated_velocity_model,
                    updated_damping_model,
                )
                updated_misfit = float(updated_objective["total"])
                if parameter_block == "velocity":
                    velocity_only_misfit = updated_misfit
                else:
                    damping_only_misfit = updated_misfit

            else:
                candidate_results = {}
                if velocity_available:
                    candidate_results["velocity"] = (
                        velocity_proposal,
                        best_damping_model,
                        _evaluate_receiver_only_candidate(
                            velocity_proposal,
                            best_damping_model,
                        ),
                    )
                    velocity_only_misfit = float(
                        candidate_results["velocity"][2][2]["total"]
                    )
                if damping_available:
                    candidate_results["damping"] = (
                        best_velocity_model,
                        damping_proposal,
                        _evaluate_receiver_only_candidate(
                            best_velocity_model,
                            damping_proposal,
                        ),
                    )
                    damping_only_misfit = float(
                        candidate_results["damping"][2][2]["total"]
                    )
                if velocity_available and damping_available:
                    candidate_results["joint"] = (
                        velocity_proposal,
                        damping_proposal,
                        _evaluate_receiver_only_candidate(
                            velocity_proposal,
                            damping_proposal,
                        ),
                    )
                    joint_misfit = float(
                        candidate_results["joint"][2][2]["total"]
                    )

                selection = select_cross_talk_candidate(
                    current_misfit=best_misfit,
                    velocity_only_misfit=velocity_only_misfit,
                    damping_only_misfit=damping_only_misfit,
                    joint_misfit=joint_misfit,
                )
                selected_candidate = selection["selected"]
                cross_talk_interaction = selection["interaction"]
                finite_trial_misfits = [
                    value for value in (
                        velocity_only_misfit,
                        damping_only_misfit,
                        joint_misfit,
                    )
                    if value is not None and np.isfinite(value)
                ]
                updated_misfit = (
                    min(finite_trial_misfits)
                    if finite_trial_misfits else best_misfit
                )

                if selected_candidate != "current":
                    (
                        updated_velocity_model,
                        updated_damping_model,
                        receiver_only_result,
                    ) = candidate_results[selected_candidate]
                    (
                        wavefield_forward_updated,
                        updated_synthetic_waveform,
                        updated_trace_scale,
                        updated_objective,
                    ) = _evaluate_full_candidate(
                        updated_velocity_model,
                        updated_damping_model,
                    )
                    updated_misfit = float(updated_objective["total"])
                    receiver_only_misfit = float(
                        receiver_only_result[2]["total"]
                    )
                    if not np.isclose(
                        updated_misfit,
                        receiver_only_misfit,
                        rtol=1e-9,
                        atol=1e-12,
                    ):
                        raise RuntimeError(
                            "receiver-only and full-wavefield forwards gave "
                            "different candidate objectives"
                        )

            trial_was_accepted = bool(
                selected_candidate != "current"
                and updated_misfit < best_misfit
            )
            old_best_misfit = best_misfit

            # ------------------------------------------------------------------
            # ACCEPT / REJECT AND ADAPT EACH BLOCK STEP INDEPENDENTLY
            # ------------------------------------------------------------------
            if trial_was_accepted:
                updating = True
                print(f"    Accepted [{selected_candidate}]")

                best_velocity_model = updated_velocity_model
                best_damping_model = updated_damping_model
                best_wavefield_forward = wavefield_forward_updated
                best_synthetic_waveform = updated_synthetic_waveform
                best_trace_scale = updated_trace_scale
                best_misfit = updated_misfit
                best_objective = updated_objective
            else:
                updating = False

            if update_strategy == "joint":
                if not trial_was_accepted:
                    if velocity_available:
                        dc_max /= reduce_factor
                    if damping_available:
                        da_max /= reduce_factor
            elif update_strategy == "alternating":
                if not trial_was_accepted:
                    if parameter_block == "velocity":
                        dc_max /= reduce_factor
                    else:
                        da_max /= reduce_factor
            else:
                if velocity_available and (
                    velocity_only_misfit is None
                    or velocity_only_misfit >= old_best_misfit
                ):
                    dc_max /= reduce_factor
                if damping_available and (
                    damping_only_misfit is None
                    or damping_only_misfit >= old_best_misfit
                ):
                    da_max /= reduce_factor

            misfit_gap = updated_misfit - old_best_misfit
            local_iteration_history.append({
                "iteration": int(iteration + 1),
                "trial_misfit": float(updated_misfit),
                "best_misfit": float(best_misfit),
                "accepted": bool(trial_was_accepted),
                "update_strategy": update_strategy,
                "parameter_block": selected_candidate,
                "velocity_only_misfit": velocity_only_misfit,
                "damping_only_misfit": damping_only_misfit,
                "joint_misfit": joint_misfit,
                "cross_talk_interaction": cross_talk_interaction,
                "velocity_step_limit": float(dc_max),
                "damping_step_limit": float(da_max),
            })

            # Drop the trial binding before the next allocation.  When the
            # trial was accepted, best_wavefield_forward retains the array.
            if wavefield_forward_updated is not None:
                del wavefield_forward_updated

            if (
                update_strategy == "joint"
                and not trial_was_accepted
                and 0.0 <= misfit_gap < misfit_threshold
            ):
                print("Stopping: rejected step is too close to current best misfit.")
                break

        # ------------------------------------------------------------------
        # WRITE BACK BEST RESULT
        # ------------------------------------------------------------------
        self.synthetic_waveform = best_synthetic_waveform
        self.trace_scale = float(best_trace_scale)
        if amplitude_scale_mode == "fixed":
            self.reference_trace_scale = float(best_trace_scale)
        self.amplitude_scale_mode = amplitude_scale_mode
        self.spectral_gamma = float(spectral_gamma)
        self.objective_taper_fraction = float(objective_taper_fraction)
        self.local_update_strategy = update_strategy
        self.objective_components = dict(best_objective)
        self.local_inversion_history = local_iteration_history
        self.source_handler.time_function = best_source_time_function
        self.source_handler.spatial_function = best_source_spatial_function
        self.receiver_handler.spatial_function = best_receiver_spatial_function
        self.misfit = best_misfit
        self.wavefield_forward = (
            best_wavefield_forward if retain_wavefield else None
        )

        if self.geometry_type == "dds":
            self.velocity_model_handler.velocity_array = best_velocity_model
            self.velocity_model_handler.damping_array = best_damping_model

            gouge_indices = np.concatenate([
                idx_dict["gouge_1"],
                idx_dict["gouge_2"],
            ])

            self.average_gouge_damping = np.mean(
                self.velocity_model_handler.damping_array[gouge_indices]
            )
            self.average_gouge_velocity = np.mean(
                self.velocity_model_handler.velocity_array[gouge_indices]
            )

            damping_label = str(round(self.average_gouge_damping, 5)).replace(".", ",")
            velocity_label = str(round(1e4 * self.average_gouge_velocity)).replace(".", ",")
            misfit_label = _filename_decimal(best_misfit, 4)
            label = (
                f"_vel_{velocity_label}_damping_{damping_label}"
                f"_FWI_misfit_{misfit_label}"
            )
        else:
            self.velocity_model_handler.velocity_array = best_velocity_model
            self.velocity_model_handler.damping_array = best_damping_model
            label = f"FWI_{best_misfit:.4f}_waveform"

        if enable_plotting and plot_output_path:
            record_output_directory = pathlib.Path(plot_output_path)
            record_output_directory.mkdir(parents=True, exist_ok=True)
            (
                waveform_output_path,
                velocity_model_output_path,
                damping_model_output_path,
            ) = _local_inversion_plot_paths(
                record_output_directory,
                label,
                name_prefix=plot_name_prefix,
            )

            self.plotter.plot_simulation_waveform(
                t=observed_time,
                sp_simulated=best_synthetic_waveform,
                sp_recorded=observed_waveform,
                misfit_interval=misfit_interval,
                outfile_path=waveform_output_path,
            )

            if dc_max_start:
                self.velocity_model_handler.plot(
                    model=self.velocity_model_handler.velocity_array,
                    outfile_path=velocity_model_output_path,
                    quantity="velocity",
                )

            if da_max_start:
                self.velocity_model_handler.plot(
                    model=self.velocity_model_handler.damping_array,
                    outfile_path=damping_model_output_path,
                    quantity="damping",
                )

            plt.close('all')

        if not retain_wavefield:
            # The trace, objective and accepted material profiles have already
            # been copied into the modeler.  No later continuation step needs
            # the space-time field itself.
            del best_wavefield_forward

        # if (make_movie) and (initial_misfit != best_misfit):
        #     movie_output_name = movie_output_path.name + "_local_inversion.mp4"
        #     movie_output_path = movie_output_path.parent / movie_output_name

        #     self.plotter.make_movie_from_simulation(
        #         outfile_path=movie_output_path,
        #         x=spatial_axis,
        #         t=simulation_time,
        #         sp_field=best_wavefield_forward,
        #         sp_recorded = best_simulated_waveform,
        #         sample_dimensions=self.sample_dimensions,
        #         idx_dict=idx_dict,
        #     )
    
    # ------------------------------------------------------------------
    def run_local_source_receiver_inversion(
        self,
        n_iterations: int,
        dw_max_start: float = 0,
        dw_threshold: float = 0,
        ds_max_start: float = 0,
        ds_threshold: float = 0,
        stf_duration: float = 4.0,          # [µs]
        reduce_factor: float = 2.0,
        misfit_threshold: float = 1.0,
        normalize_waveform: bool = False,
        enable_plotting: bool = True,
        make_movie: bool = False,
        plot_output_path: Optional[str] = None,
        movie_output_path: Optional[str] = "simulation_movie.mp4",
    ) -> None:
        """
        Local waveform inversion for:
        1) source time function
        2) source spatial function
        3) receiver spatial function

        Velocity and damping are kept fixed.
        """

        if self.synthetic_waveform is None:
            raise RuntimeError("No forward-simulation results. Call `forward_simulation()` first.")
        self._require_legacy_adjoint_configuration()

        # ------------------------------------------------------------------
        # 0) OPTIONAL: SHRINK GRID IF GOUGE SHRUNK
        # ------------------------------------------------------------------
        try: 
            (
                self.assembly_dict,
                new_idx_dict,
                new_x,
                new_v,
                new_a,
                new_src_sp,
                new_rcv_sp,
                removed_idx,
            ) = shrink_grid_if_gouge_shrinks(
                assembly_dict=self.assembly_dict,
                idx_dict=self.velocity_model_handler.idx_dict,
                spatial_axis=self.grid_handler.spatial_axis,
                dx=self.grid_handler.dx,
                velocity_model=self.velocity_model_handler.velocity_array,
                damping_model=self.velocity_model_handler.damping_array,
                source_spatial_function=self.source_handler.spatial_function,
                receiver_spatial_function=self.receiver_handler.spatial_function,
            )

            if removed_idx.size > 0:
                self.grid_handler.spatial_axis = new_x
                self.grid_handler.total_grid_points = new_x.size

                self.velocity_model_handler.velocity_array = new_v
                self.velocity_model_handler.damping_array = new_a
                self.velocity_model_handler.idx_dict = new_idx_dict
                self.velocity_model_handler.x = new_x
                self.source_handler.spatial_function = new_src_sp
                self.receiver_handler.spatial_function = new_rcv_sp

        except:
            pass
        # ------------------------------------------------------------------
        # 1) RE-BIND ALL locals AFTER SHRINK
        # ------------------------------------------------------------------
        absorbing = self.absorbing

        observed_time = self.observed_time
        observed_waveform = self.observed_waveform
        misfit_interval = self.misfit_interval
        num_x = self.grid_handler.total_grid_points
        delta_x = self.grid_handler.dx
        delta_t = self.sim_time_handler.dt
        num_t = self.sim_time_handler.num_t
        simulation_time = self.sim_time_handler.simulation_time
        spatial_axis = self.grid_handler.spatial_axis

        initial_source_time_function = self.source_handler.time_function.copy()
        initial_source_spatial_function = self.source_handler.spatial_function.copy()
        initial_receiver_spatial_function = self.receiver_handler.spatial_function.copy()
        fixed_velocity_model = self.velocity_model_handler.velocity_array.copy()
        fixed_damping_model = self.velocity_model_handler.damping_array.copy()

        # ------------------------------------------------------------------
        # 2) SUPPORTS / UPDATE REGIONS FOR SOURCE AND RECEIVER
        # ------------------------------------------------------------------
        # STF active support
        stf_duration_idx = int(np.searchsorted(simulation_time - simulation_time[0], stf_duration, side="right"))
        stf_duration_idx = max(1, min(stf_duration_idx, num_t))

        # source support = where source is currently non-zero
        src_support_idx = np.flatnonzero(np.abs(initial_source_spatial_function) > 0)
        if src_support_idx.size == 0:
            raise ValueError("Initial source spatial function has empty support.")

        # receiver support = where receiver is currently non-zero
        rcv_support_idx = np.flatnonzero(np.abs(initial_receiver_spatial_function) > 0)
        if rcv_support_idx.size == 0:
            raise ValueError("Initial receiver spatial function has empty support.")

        # masks for updates
        stf_update_mask = np.zeros(num_t, dtype=bool)
        stf_update_mask[:stf_duration_idx] = True

        src_update_mask = np.zeros(num_x, dtype=bool)
        src_update_mask[src_support_idx] = True

        rcv_update_mask = np.zeros(num_x, dtype=bool)
        rcv_update_mask[rcv_support_idx] = True

        # ------------------------------------------------------------------
        # 3) INITIAL FORWARD + MISFIT
        # ------------------------------------------------------------------
        initial_wavefield_forward = pseudospectral_1D_damped(
            num_x=num_x,
            delta_x=delta_x,
            num_t=num_t,
            delta_t=delta_t,
            source_spatial_function=initial_source_spatial_function,
            source_time_function=initial_source_time_function,
            velocity_model=fixed_velocity_model,
            damping_model=fixed_damping_model,
            absorbing=absorbing,
        )

        initial_simulated_waveform = np.sum(
            initial_wavefield_forward * initial_receiver_spatial_function,
            axis=1
        )
        initial_synthetic_waveform = np.interp(
            observed_time,
            simulation_time,
            initial_simulated_waveform
        )

        # Important:
        # if you invert source/receiver amplitudes, normalization can fight the inversion.
        # Keep the option, but default should be False.
        if normalize_waveform:
            fixed_trace_scale = fit_trace_scale(
                observed_waveform=observed_waveform,
                synthetic_waveform=initial_synthetic_waveform,
                misfit_interval=misfit_interval,
            )
            initial_synthetic_waveform *= fixed_trace_scale
        else:
            fixed_trace_scale = 1.0

        initial_misfit = self.compute_misfit(
            observed_waveform=observed_waveform,
            synthetic_waveform=initial_synthetic_waveform,
            misfit_interval=misfit_interval
        )

        # ------------------------------------------------------------------
        # STORE "BEST" STATE
        # ------------------------------------------------------------------
        best_source_time_function = initial_source_time_function.copy()
        best_source_spatial_function = initial_source_spatial_function.copy()
        best_receiver_spatial_function = initial_receiver_spatial_function.copy()

        best_wavefield_forward = initial_wavefield_forward.copy()
        best_synthetic_waveform = initial_synthetic_waveform.copy()
        best_misfit = initial_misfit

        previous_misfit = initial_misfit

        dw_max = dw_max_start
        ds_max = ds_max_start

        # ------------------------------------------------------------------
        # PRECOMPUTE MASK
        # ------------------------------------------------------------------
        mask = np.zeros_like(observed_waveform, dtype=bool)
        mask[misfit_interval] = True

        updating = True
        activate_stf = dw_max > 0
        activate_source_spatial = ds_max > 0
        activate_receiver_spatial = ds_max > 0

        # ------------------------------------------------------------------
        # OPTIONAL BOUNDS / CONSTRAINTS
        # ------------------------------------------------------------------
        # keep support fixed and prevent wild sign flips if desired
        # Here only simple clipping / renormalization.
        src_norm0 = np.linalg.norm(initial_source_spatial_function[src_support_idx])
        rcv_norm0 = np.linalg.norm(initial_receiver_spatial_function[rcv_support_idx])

        for iteration in range(n_iterations):
            updated_source_time_function = best_source_time_function.copy()
            updated_source_spatial_function = best_source_spatial_function.copy()
            updated_receiver_spatial_function = best_receiver_spatial_function.copy()

            if updating:
                # ----------------------------------------------------------
                # DATA RESIDUAL ON OBSERVED GRID
                # ----------------------------------------------------------
                residual_obs = (best_synthetic_waveform - observed_waveform) * mask

                # ----------------------------------------------------------
                # MAP RESIDUAL TO SIMULATION-TIME GRID
                # ----------------------------------------------------------
                residual_sim = residual_to_simulation_time(
                    observed_time=observed_time,
                    simulation_time=simulation_time,
                    observed_residual=residual_obs,
                )

                # ----------------------------------------------------------
                # BUILD dJ/du USING CURRENT RECEIVER WEIGHTING
                # ----------------------------------------------------------
                dJdu = build_adjoint_data_gradient(
                    residual_sim=residual_sim,
                    receiver_spatial_function=best_receiver_spatial_function,
                )

                # ----------------------------------------------------------
                # DISCRETE ADJOINT
                # ----------------------------------------------------------
                adjoint_wavefield = discrete_adjoint_1D_kelvin_voigt(
                    dJdu=dJdu,
                    velocity_model=fixed_velocity_model,
                    damping_model=fixed_damping_model,
                    delta_x=delta_x,
                    delta_t=delta_t,
                )

                # ----------------------------------------------------------
                # SOURCE-TIME GRADIENT
                # J wrt w_n ~ <lambda_{n+1}, f>
                # ----------------------------------------------------------
                if activate_stf:
                    gradient_w = np.sum(
                        adjoint_wavefield * best_source_spatial_function[None, :],
                        axis=1
                    )
                    gradient_w[~stf_update_mask] = 0.0
                    max_w_grad = np.max(np.abs(gradient_w[stf_update_mask]))

                # ----------------------------------------------------------
                # SOURCE-SPATIAL GRADIENT
                # J wrt f_i ~ sum_n lambda_{n+1,i} * w_n
                # ----------------------------------------------------------
                if activate_source_spatial:
                    gradient_s = np.sum(
                        adjoint_wavefield * best_source_time_function[:, None],
                        axis=0
                    )
                    gradient_s[~src_update_mask] = 0.0
                    max_s_grad = np.max(np.abs(gradient_s[src_update_mask]))

                # ----------------------------------------------------------
                # RECEIVER-SPATIAL GRADIENT
                # y_n = <u_n, r> => dJ/dr_i = sum_n residual_n * u_n,i
                # Use simulation-time residual for consistency with u_n.
                # ----------------------------------------------------------
                if activate_receiver_spatial:
                    gradient_r = np.sum(
                        best_wavefield_forward * residual_sim[:, None],
                        axis=0
                    )
                    gradient_r[~rcv_update_mask] = 0.0
                    max_r_grad = np.max(np.abs(gradient_r[rcv_update_mask]))

            # ------------------------------------------------------------------
            # STEP: STF
            # ------------------------------------------------------------------
            if activate_stf and max_w_grad > 0:
                step_size_w = dw_max / (max_w_grad + 1e-15)

                # keep same sign convention that improved your other loop
                updated_source_time_function[stf_update_mask] -= (
                    step_size_w * gradient_w[stf_update_mask]
                )

                # optional: keep STF zero after support
                updated_source_time_function[~stf_update_mask] = 0.0

            # ------------------------------------------------------------------
            # STEP: SOURCE SPATIAL FUNCTION
            # ------------------------------------------------------------------
            if activate_source_spatial and max_s_grad > 0:
                step_size_s = ds_max / (max_s_grad + 1e-15)

                updated_source_spatial_function[src_update_mask] += (
                    step_size_s * gradient_s[src_update_mask]
                )

                # keep support fixed
                updated_source_spatial_function[~src_update_mask] = 0.0

                # optional L2 renormalization to avoid amplitude blow-up
                src_norm = np.linalg.norm(updated_source_spatial_function[src_support_idx])
                if src_norm > 1e-15:
                    updated_source_spatial_function[src_support_idx] *= (src_norm0 / src_norm)

            # ------------------------------------------------------------------
            # STEP: RECEIVER SPATIAL FUNCTION
            # ------------------------------------------------------------------
            if activate_receiver_spatial and max_r_grad > 0:
                step_size_r = ds_max / (max_r_grad + 1e-15)

                updated_receiver_spatial_function[rcv_update_mask] += (
                    step_size_r * gradient_r[rcv_update_mask]
                )

                # keep support fixed
                updated_receiver_spatial_function[~rcv_update_mask] = 0.0

                # optional L2 renormalization
                rcv_norm = np.linalg.norm(updated_receiver_spatial_function[rcv_support_idx])
                if rcv_norm > 1e-15:
                    updated_receiver_spatial_function[rcv_support_idx] *= (rcv_norm0 / rcv_norm)

            # ------------------------------------------------------------------
            # FORWARD MODEL WITH UPDATED SOURCE / RECEIVER
            # ------------------------------------------------------------------
            wavefield_forward_updated = pseudospectral_1D_damped(
                num_x=num_x,
                delta_x=delta_x,
                num_t=num_t,
                delta_t=delta_t,
                source_spatial_function=updated_source_spatial_function,
                source_time_function=updated_source_time_function,
                velocity_model=fixed_velocity_model,
                damping_model=fixed_damping_model,
                absorbing=absorbing
            )

            updated_simulated_waveform = np.sum(
                wavefield_forward_updated * updated_receiver_spatial_function,
                axis=1
            )
            updated_synthetic_waveform = np.interp(
                observed_time,
                simulation_time,
                updated_simulated_waveform
            )

            if normalize_waveform:
                updated_synthetic_waveform *= fixed_trace_scale

            updated_misfit = self.compute_misfit(
                observed_waveform=observed_waveform,
                synthetic_waveform=updated_synthetic_waveform,
                misfit_interval=misfit_interval
            )

            # ------------------------------------------------------------------
            # ACCEPT / REJECT
            # ------------------------------------------------------------------
            if updated_misfit < best_misfit:
                updating = True
                print("    Accepted")

                best_source_time_function = updated_source_time_function
                best_source_spatial_function = updated_source_spatial_function
                best_receiver_spatial_function = updated_receiver_spatial_function

                best_wavefield_forward = wavefield_forward_updated
                best_synthetic_waveform = updated_synthetic_waveform
                best_misfit = updated_misfit

            else:
                updating = False
                misfit_gap = updated_misfit - best_misfit

                if activate_stf:
                    dw_max /= reduce_factor
                if activate_source_spatial or activate_receiver_spatial:
                    ds_max /= reduce_factor

                if misfit_gap < misfit_threshold:
                    print("Stopping: rejected step is too close to current best misfit.")
                    break

            stop_stf = (not activate_stf) or (dw_max < dw_threshold)
            stop_spatial = ((not activate_source_spatial) and (not activate_receiver_spatial)) or (ds_max < ds_threshold)

            if stop_stf and stop_spatial:
                print("Stopping: active step sizes below thresholds.")
                break

            previous_misfit = updated_misfit

        # ------------------------------------------------------------------
        # WRITE BACK BEST RESULT
        # ------------------------------------------------------------------
        self.synthetic_waveform = best_synthetic_waveform
        self.source_handler.time_function = best_source_time_function
        self.source_handler.spatial_function = best_source_spatial_function
        self.receiver_handler.spatial_function = best_receiver_spatial_function
        self.misfit = best_misfit

        # velocity and damping remain unchanged
        self.velocity_model_handler.velocity_array = fixed_velocity_model
        self.velocity_model_handler.damping_array = fixed_damping_model

        label = f"_SRCINV_misfit_{best_misfit:.0f}"

        if enable_plotting and plot_output_path:
            plot_output_name = plot_output_path.name + label
            plot_output_path = plot_output_path.parent / plot_output_name

            self.plotter.plot_simulation_waveform(
                t=observed_time,
                sp_simulated=best_synthetic_waveform,
                sp_recorded=observed_waveform,
                misfit_interval=misfit_interval,
                outfile_path=plot_output_path
            )

            if dw_max_start:
                stf_output_name = plot_output_path.name + "_best_STF"
                stf_output_path = plot_output_path.parent / stf_output_name
                self.plotter.plot_original_vs_updated_stf(
                    t=simulation_time,
                    stf_updated=best_source_time_function,
                    stf_original=initial_source_time_function,
                    min_time=simulation_time[0],
                    max_time=simulation_time[min(stf_duration_idx, len(simulation_time) - 1)],
                    outfile_path=stf_output_path,
                )

            plt.close("all")

    def run_local_stf_inversion(
        self,
        n_iterations: int,
        dw_max_start: float = 0,
        dw_threshold: float = 0,
        stf_duration: float = 4.0,   # [µs]
        reduce_factor: float = 2.0,
        misfit_threshold: float = 1.0,
        normalize_waveform: bool = False,
        smooth_gradient: bool = True,
        gradient_smoothing_halfwidth: int = 2,
        enable_plotting: bool = True,
        make_movie: bool = False,
        plot_output_path: Optional[str] = None,
        movie_output_path: Optional[str] = "simulation_movie.mp4",
    ) -> None:
        """
        Local waveform inversion for the source time function only.

        Velocity, damping, source spatial function and receiver spatial function
        are kept fixed.
        """

        if self.synthetic_waveform is None:
            raise RuntimeError("No forward-simulation results. Call `forward_simulation()` first.")
        self._require_legacy_adjoint_configuration()

        # ------------------------------------------------------------------
        # 0) OPTIONAL: SHRINK GRID IF GOUGE SHRUNK
        # ------------------------------------------------------------------
        try:
            (
                self.assembly_dict,
                new_idx_dict,
                new_x,
                new_v,
                new_a,
                new_src_sp,
                new_rcv_sp,
                removed_idx,
            ) = shrink_grid_if_gouge_shrinks(
                assembly_dict=self.assembly_dict,
                idx_dict=self.velocity_model_handler.idx_dict,
                spatial_axis=self.grid_handler.spatial_axis,
                dx=self.grid_handler.dx,
                velocity_model=self.velocity_model_handler.velocity_array,
                damping_model=self.velocity_model_handler.damping_array,
                source_spatial_function=self.source_handler.spatial_function,
                receiver_spatial_function=self.receiver_handler.spatial_function,
            )

            if removed_idx.size > 0:
                self.grid_handler.spatial_axis = new_x
                self.grid_handler.total_grid_points = new_x.size

                self.velocity_model_handler.velocity_array = new_v
                self.velocity_model_handler.damping_array = new_a
                self.velocity_model_handler.idx_dict = new_idx_dict
                self.velocity_model_handler.x = new_x
                self.source_handler.spatial_function = new_src_sp
                self.receiver_handler.spatial_function = new_rcv_sp
        except:
            pass

        # ------------------------------------------------------------------
        # 1) RE-BIND ALL locals AFTER SHRINK
        # ------------------------------------------------------------------
        absorbing = self.absorbing

        observed_time = self.observed_time
        observed_waveform = self.observed_waveform
        misfit_interval = self.misfit_interval
        num_x = self.grid_handler.total_grid_points
        delta_x = self.grid_handler.dx
        delta_t = self.sim_time_handler.dt
        num_t = self.sim_time_handler.num_t
        simulation_time = self.sim_time_handler.simulation_time

        fixed_velocity_model = self.velocity_model_handler.velocity_array.copy()
        fixed_damping_model = self.velocity_model_handler.damping_array.copy()
        fixed_source_spatial_function = self.source_handler.spatial_function.copy()
        fixed_receiver_spatial_function = self.receiver_handler.spatial_function.copy()

        initial_source_time_function = self.source_handler.time_function.copy()

        # ------------------------------------------------------------------
        # 2) STF SUPPORT
        # ------------------------------------------------------------------
        stf_duration_idx = int(np.searchsorted(
            simulation_time - simulation_time[0],
            stf_duration,
            side="right"
        ))
        stf_duration_idx = max(1, min(stf_duration_idx, num_t))

        stf_update_mask = np.zeros(num_t, dtype=bool)
        stf_update_mask[:stf_duration_idx] = True

        # ------------------------------------------------------------------
        # 3) INITIAL FORWARD + MISFIT
        # ------------------------------------------------------------------
        initial_wavefield_forward = pseudospectral_1D_damped(
            num_x=num_x,
            delta_x=delta_x,
            num_t=num_t,
            delta_t=delta_t,
            source_spatial_function=fixed_source_spatial_function,
            source_time_function=initial_source_time_function,
            velocity_model=fixed_velocity_model,
            damping_model=fixed_damping_model,
            absorbing=absorbing,
        )

        initial_simulated_waveform = np.sum(
            initial_wavefield_forward * fixed_receiver_spatial_function,
            axis=1
        )
        initial_synthetic_waveform = np.interp(
            observed_time,
            simulation_time,
            initial_simulated_waveform
        )

        if normalize_waveform:
            fixed_trace_scale = fit_trace_scale(
                observed_waveform=observed_waveform,
                synthetic_waveform=initial_synthetic_waveform,
                misfit_interval=misfit_interval,
            )
            initial_synthetic_waveform *= fixed_trace_scale
        else:
            fixed_trace_scale = 1.0

        initial_misfit = self.compute_misfit(
            observed_waveform=observed_waveform,
            synthetic_waveform=initial_synthetic_waveform,
            misfit_interval=misfit_interval,
        )

        # ------------------------------------------------------------------
        # STORE "BEST" STATE
        # ------------------------------------------------------------------
        best_source_time_function = initial_source_time_function.copy()
        best_wavefield_forward = initial_wavefield_forward.copy()
        best_synthetic_waveform = initial_synthetic_waveform.copy()
        best_misfit = initial_misfit

        dw_max = dw_max_start
        updating = True
        activate_stf = dw_max > 0

        # ------------------------------------------------------------------
        # PRECOMPUTE MASK
        # ------------------------------------------------------------------
        mask = np.zeros_like(observed_waveform, dtype=bool)
        mask[misfit_interval] = True

        # ------------------------------------------------------------------
        # HELPER: smooth 1D gradient
        # ------------------------------------------------------------------
        def _smooth_1d(arr: np.ndarray, halfwidth: int) -> np.ndarray:
            if halfwidth <= 0:
                return arr.copy()
            width = 2 * halfwidth + 1
            kernel = np.ones(width, dtype=float) / width
            return np.convolve(arr, kernel, mode="same")

        for iteration in range(n_iterations):
            updated_source_time_function = best_source_time_function.copy()

            if updating:
                # ----------------------------------------------------------
                # DATA RESIDUAL ON OBSERVED GRID
                # ----------------------------------------------------------
                residual_obs = (best_synthetic_waveform - observed_waveform) * mask

                # ----------------------------------------------------------
                # MAP RESIDUAL TO SIMULATION-TIME GRID
                # ----------------------------------------------------------
                residual_sim = residual_to_simulation_time(
                    observed_time=observed_time,
                    simulation_time=simulation_time,
                    observed_residual=residual_obs,
                )

                # ----------------------------------------------------------
                # BUILD dJ/du USING FIXED RECEIVER WEIGHTING
                # ----------------------------------------------------------
                dJdu = build_adjoint_data_gradient(
                    residual_sim=residual_sim,
                    receiver_spatial_function=fixed_receiver_spatial_function,
                )

                # ----------------------------------------------------------
                # DISCRETE ADJOINT
                # ----------------------------------------------------------
                adjoint_wavefield = discrete_adjoint_1D_kelvin_voigt(
                    dJdu=dJdu,
                    velocity_model=fixed_velocity_model,
                    damping_model=fixed_damping_model,
                    delta_x=delta_x,
                    delta_t=delta_t,
                )

                # ----------------------------------------------------------
                # SHIFTED ADJOINT FOR SOURCE INJECTION
                #
                # source_time_function[it] is injected into u^{n+1},
                # so the source gradient couples to lambda at the next step.
                # ----------------------------------------------------------
                adjoint_for_source = np.zeros_like(adjoint_wavefield)
                adjoint_for_source[:-1, :] = adjoint_wavefield[1:, :]
                adjoint_for_source[-1, :] = 0.0

                # ----------------------------------------------------------
                # STF GRADIENT
                # dJ/dw_n = <lambda_for_source[n], f>
                # ----------------------------------------------------------
                gradient_w = np.sum(
                    adjoint_for_source * fixed_source_spatial_function[None, :],
                    axis=1
                )

                gradient_w[~stf_update_mask] = 0.0

                if smooth_gradient:
                    gradient_w[stf_update_mask] = _smooth_1d(
                        gradient_w[stf_update_mask],
                        gradient_smoothing_halfwidth
                    )

                max_w_grad = np.max(np.abs(gradient_w[stf_update_mask]))

            # ------------------------------------------------------------------
            # STEP: STF
            # ------------------------------------------------------------------
            if activate_stf and max_w_grad > 0:
                step_size_w = dw_max / (max_w_grad + 1e-15)

                updated_source_time_function[stf_update_mask] -= (
                    step_size_w * gradient_w[stf_update_mask]
                )

                updated_source_time_function[~stf_update_mask] = 0.0

            # ------------------------------------------------------------------
            # FORWARD MODEL WITH UPDATED STF
            # ------------------------------------------------------------------
            wavefield_forward_updated = pseudospectral_1D_damped(
                num_x=num_x,
                delta_x=delta_x,
                num_t=num_t,
                delta_t=delta_t,
                source_spatial_function=fixed_source_spatial_function,
                source_time_function=updated_source_time_function,
                velocity_model=fixed_velocity_model,
                damping_model=fixed_damping_model,
                absorbing=absorbing,
            )

            updated_simulated_waveform = np.sum(
                wavefield_forward_updated * fixed_receiver_spatial_function,
                axis=1
            )
            updated_synthetic_waveform = np.interp(
                observed_time,
                simulation_time,
                updated_simulated_waveform
            )

            if normalize_waveform:
                updated_synthetic_waveform *= fixed_trace_scale

            updated_misfit = self.compute_misfit(
                observed_waveform=observed_waveform,
                synthetic_waveform=updated_synthetic_waveform,
                misfit_interval=misfit_interval,
            )

            # ------------------------------------------------------------------
            # ACCEPT / REJECT
            # ------------------------------------------------------------------
            if updated_misfit < best_misfit:
                updating = True
                print(f"Iter {iteration + 1:03d} | accepted | misfit={updated_misfit:.6e}")

                best_source_time_function = updated_source_time_function
                best_wavefield_forward = wavefield_forward_updated
                best_synthetic_waveform = updated_synthetic_waveform
                best_misfit = updated_misfit

            else:
                updating = False
                misfit_gap = updated_misfit - best_misfit

                if activate_stf:
                    dw_max /= reduce_factor

                print(f"Iter {iteration + 1:03d} | rejected | trial={updated_misfit:.6e} | best={best_misfit:.6e} | new dw_max={dw_max:.3e}")

                if misfit_gap < misfit_threshold:
                    print("Stopping: rejected step is too close to current best misfit.")
                    break

            stop_stf = (not activate_stf) or (dw_max < dw_threshold)
            if stop_stf:
                print("Stopping: STF step size below threshold.")
                break

        # ------------------------------------------------------------------
        # WRITE BACK BEST RESULT
        # ------------------------------------------------------------------
        self.synthetic_waveform = best_synthetic_waveform
        self.source_handler.time_function = best_source_time_function
        self.source_handler.spatial_function = fixed_source_spatial_function
        self.receiver_handler.spatial_function = fixed_receiver_spatial_function
        self.misfit = best_misfit

        self.velocity_model_handler.velocity_array = fixed_velocity_model
        self.velocity_model_handler.damping_array = fixed_damping_model

        label = f"local_misfit_{best_misfit:.0f}"
        print(label)

        if enable_plotting and plot_output_path:
            plot_output_name = plot_output_path.name.split("global")[0] + label
            plot_output_path = plot_output_path.parent / plot_output_name
            print(plot_output_path)
            self.plotter.plot_simulation_waveform(
                t=observed_time,
                sp_simulated=best_synthetic_waveform,
                sp_recorded=observed_waveform,
                misfit_interval=misfit_interval,
                outfile_path=plot_output_path
            )

            stf_output_name = plot_output_path.name + "_best_STF"
            stf_output_path = plot_output_path.parent / stf_output_name
            self.plotter.plot_original_vs_updated_stf(
                t=simulation_time,
                stf_updated=best_source_time_function,
                stf_original=initial_source_time_function,
                min_time=simulation_time[0],
                max_time=simulation_time[min(stf_duration_idx, len(simulation_time) - 1)],
                outfile_path=stf_output_path,
            )

            plt.close("all")

def pseudospectral_1D_damped(
    num_x: int,
    delta_x: float,
    num_t: int,
    delta_t: float,
    source_spatial_function: np.ndarray,
    source_time_function: np.ndarray,
    velocity_model: np.ndarray,
    damping_model: np.ndarray = None,
    density_model: np.ndarray = None,
    absorbing: bool = False,
    N_pad: int = 200,
    boundary_condition: Optional[str] = None,
    boundary_impedance: Optional[float | Sequence[float]] = None,
    source_injection: str = "legacy_increment",
):
    """
    Backward-compatible alias for the finite-difference propagator.

    The historical public name is retained because the local DDS inversion
    methods still call it.  It was never a pseudospectral implementation.
    Unless new options are supplied, this wrapper reproduces the old source
    increment and endpoint treatment exactly.
    """
    if boundary_condition is None:
        boundary_condition = "absorbing" if absorbing else "legacy_fixed"

    return finite_difference_1d_kelvin_voigt(
        num_x=num_x,
        delta_x=delta_x,
        num_t=num_t,
        delta_t=delta_t,
        source_spatial_function=source_spatial_function,
        source_time_function=source_time_function,
        velocity_model=velocity_model,
        damping_model=damping_model,
        density_model=density_model,
        boundary_condition=boundary_condition,
        boundary_impedance=boundary_impedance,
        source_injection=source_injection,
        n_pad=N_pad,
    )

#########################################################################
# ------------------------------------------------------------------
# Helper: 2‑nd derivative via *pre‑tabulated* spectral multiplier
# ------------------------------------------------------------------
def compute_time_derivatives(wavefield_out, delta_t):
    """
    Returns wavefield_tt of shape (num_t, num_x), 
    where wavefield_tt[i, j] ~ second derivative in time of the wavefield.
    """
    num_t, num_x = wavefield_out.shape
    wavefield_tt = np.zeros_like(wavefield_out)
    
    for ix in range(num_x):
        # 1) Take a first derivative in time
        # 2) Then a second derivative in time
        first_derivative = np.gradient(wavefield_out[:, ix], delta_t)
        second_derivative = np.gradient(first_derivative, delta_t)
        
        wavefield_tt[:, ix] = second_derivative
    
    return wavefield_tt

########################################
# JIT KERNEL FOR FASTER COMPUTATION
#######################################
try:
    from numba import njit, prange
except ImportError:  # Numba is an optional accelerator, not a runtime requirement.
    def njit(*decorator_args, **decorator_kwargs):
        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1:
            return decorator_args[0]

        def decorate(function):
            return function

        return decorate

    prange = range
# ------------------------------------------------------------
# Spatial Laplacian  ∂²u/∂x²   (shape = (num_t, num_x))
# ------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def compute_laplacian(u: np.ndarray, dx: float) -> np.ndarray:
    """
    Second spatial derivative for each time slice.
    """
    nt, nx = u.shape
    coeff  = 1.0 / (dx * dx)
    out    = np.empty_like(u)

    for it in prange(nt):                     # ← parallel over time
        # Dirichlet copy boundaries (cheap and OK for FWI)
        out[it, 0]      = 0.0
        out[it, nx - 1] = 0.0
        for ix in range(1, nx - 1):
            out[it, ix] = (u[it, ix + 1] - 2.0 * u[it, ix] + u[it, ix - 1]) * coeff
    return out


# ------------------------------------------------------------
# Time derivative of Laplacian  ∂/∂t (∂²u/∂x²)
# ------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def compute_dt_laplacian(lap: np.ndarray, dt: float) -> np.ndarray:
    """
    First‑order time derivative (central differences) of a (num_t, num_x) array.
    """
    nt, nx = lap.shape
    coeff  = 1.0 / (2.0 * dt)
    out    = np.empty_like(lap)

    for ix in prange(nx):                     # ← parallel over space
        out[0, ix]       = 0.0
        out[nt - 1, ix]  = 0.0
        for it in range(1, nt - 1):
            out[it, ix] = (lap[it + 1, ix] - lap[it - 1, ix]) * coeff
    return out

@njit(parallel=True, fastmath=True)
def compute_dx(u: np.ndarray, dx: float) -> np.ndarray:
    nt, nx = u.shape
    coeff = 1.0 / (2.0 * dx)
    out = np.empty_like(u)
    for it in prange(nt):
        out[it, 0] = 0.0
        out[it, nx-1] = 0.0
        for ix in range(1, nx-1):
            out[it, ix] = (u[it, ix+1] - u[it, ix-1]) * coeff
    return out

@njit(parallel=True, fastmath=True)
def compute_dt(u: np.ndarray, dt: float) -> np.ndarray:
    nt, nx = u.shape
    coeff = 1.0 / (2.0 * dt)
    out = np.empty_like(u)
    for ix in prange(nx):
        out[0, ix] = 0.0
        out[nt-1, ix] = 0.0
        for it in range(1, nt-1):
            out[it, ix] = (u[it+1, ix] - u[it-1, ix]) * coeff
    return out

##############################################
# BOUNDARY CONDITION
##############################################
def extend_model(original_model, N_pad):
    """
    Extend a 1D array 'original_model' by N_pad on each side
    by simply copying boundary values. 
    """
    N = len(original_model)
    N_ext = N + 2*N_pad
    extended = np.zeros(N_ext)
    
    # Fill interior
    extended[N_pad:N_pad+N] = original_model[:]
    
    # Left padding
    for i in range(N_pad):
        extended[N_pad - 1 - i] = original_model[0]  # copy left boundary value
    
    # Right padding
    for i in range(N_pad):
        extended[N_pad+N + i] = original_model[-1]   # copy right boundary value
    
    return extended

def sigmoid(z):
    return 1/(1 + np.exp(-z))

#####################################################
# HELPER FOR RESHAPING GRID IF LAYER THICKNESS SHRINK
####################################################
def shrink_grid_if_gouge_shrinks(
    *,
    assembly_dict: Dict[str, Any],
    idx_dict: Dict[str, np.ndarray],
    spatial_axis: np.ndarray,
    dx: float,
    velocity_model: np.ndarray,
    damping_model: Optional[np.ndarray],
    source_spatial_function: np.ndarray,
    receiver_spatial_function: np.ndarray,
    snap_thickness_to_grid: bool = True,
) -> Tuple[
    Dict[str, Any],              # updated assembly_dict (sample_dimensions may be snapped)
    Dict[str, np.ndarray],       # updated idx_dict
    np.ndarray,                  # new spatial_axis
    np.ndarray,                  # new velocity_model
    Optional[np.ndarray],        # new damping_model
    np.ndarray,                  # new source_spatial_function
    np.ndarray,                  # new receiver_spatial_function
    np.ndarray,                  # removed_indices (old-grid indices removed)
]:
    """
    Shrink the 1D grid ONLY if gouge thickness decreased, keeping dx constant.

    Mechanism:
      - new N_gouge_i = round(thickness_i / dx)
      - if new < old:
          * remove (old-new) points from gouge_1 at its RIGHT end (adjacent to central block)
          * remove (old-new) points from gouge_2 at its LEFT end (adjacent to central block)

    Then delete those indices from all arrays and update idx_dict accordingly.

    IMPORTANT:
      - This preserves dx and dt assumptions; you must ensure your time axis uses fixed dt elsewhere.
      - After shrinking, you should run ONE forward solve to refresh wavefield/synthetic before the inversion loop.
    """

    # --------------------------- guards ---------------------------
    if "gouge_1" not in idx_dict or "gouge_2" not in idx_dict:
        raise ValueError("idx_dict must contain 'gouge_1' and 'gouge_2' keys.")

    n_old = spatial_axis.size
    if velocity_model.size != n_old:
        raise ValueError("velocity_model length must match spatial_axis length.")
    if damping_model is not None and damping_model.size != n_old:
        raise ValueError("damping_model length must match spatial_axis length.")
    if source_spatial_function.size != n_old:
        raise ValueError("source_spatial_function length must match spatial_axis length.")
    if receiver_spatial_function.size != n_old:
        raise ValueError("receiver_spatial_function length must match spatial_axis length.")

    sd = list(assembly_dict["sample_dimensions"])  # [side1, gouge1, central, gouge2, side2]
    h1 = float(sd[1])
    h2 = float(sd[3])

    requested_length = float(np.sum(sd))
    if assembly_dict.get("embedding_mode", "legacy_stack") == "legacy_stack":
        side = assembly_dict["side1_params"]
        requested_length += 2.0 * float(side["pzt_layer_width"])
        requested_length += 2.0 * float(side["pla_layer_width"])
    current_length = float(spatial_axis[-1] - spatial_axis[0])
    if np.isclose(requested_length, current_length, rtol=0.0, atol=0.51 * dx):
        return (
            assembly_dict, idx_dict, spatial_axis,
            velocity_model, damping_model,
            source_spatial_function, receiver_spatial_function,
            np.array([], dtype=int)
        )

    g1 = np.asarray(idx_dict["gouge_1"], dtype=int)
    g2 = np.asarray(idx_dict["gouge_2"], dtype=int)

    n1_old = g1.size
    n2_old = g2.size

    n1_new = max(1, int(np.rint(h1 / dx)))
    n2_new = max(1, int(np.rint(h2 / dx)))

    # Only act on shrink (your requirement)
    if n1_new >= n1_old and n2_new >= n2_old:
        return (
            assembly_dict, idx_dict, spatial_axis,
            velocity_model, damping_model,
            source_spatial_function, receiver_spatial_function,
            np.array([], dtype=int)
        )

    # Optionally snap thickness to exact grid multiple (recommended for consistency)
    if snap_thickness_to_grid:
        sd[1] = n1_new * dx
        sd[3] = n2_new * dx
        assembly_dict["sample_dimensions"] = sd

    # ---------------------- choose indices to remove ----------------------
    remove_idx = []

    # gouge_1: remove at RIGHT end (near central block boundary)
    r1 = max(0, n1_old - n1_new)
    if r1 > 0:
        remove_idx.append(g1[-r1:])

    # gouge_2: remove at LEFT end (near central block boundary)
    r2 = max(0, n2_old - n2_new)
    if r2 > 0:
        remove_idx.append(g2[:r2])

    removed = np.unique(np.concatenate(remove_idx)).astype(int)
    removed.sort()

    if removed.size == 0:
        return (
            assembly_dict, idx_dict, spatial_axis,
            velocity_model, damping_model,
            source_spatial_function, receiver_spatial_function,
            removed
        )

    # ---------------------- delete indices from arrays ----------------------
    n_new = n_old - removed.size
    x0 = spatial_axis[0]
    spatial_axis_new = x0 + np.arange(n_new) * dx

    velocity_new     = np.delete(velocity_model, removed)
    damping_new      = None if damping_model is None else np.delete(damping_model, removed)
    src_sp_new       = np.delete(source_spatial_function, removed)
    rcv_sp_new       = np.delete(receiver_spatial_function, removed)

    # ---------------------- rebuild idx_dict via mapping ----------------------
    # Create a mapping old_index -> new_index using a mask
    keep_mask = np.ones(n_old, dtype=bool)
    keep_mask[removed] = False
    old_to_new = -np.ones(n_old, dtype=int)
    old_to_new[keep_mask] = np.arange(keep_mask.sum(), dtype=int)

    idx_new: Dict[str, np.ndarray] = {}
    for k, arr in idx_dict.items():
        a = np.asarray(arr, dtype=int)
        # keep only indices not removed
        a = a[keep_mask[a]]
        # remap
        a = old_to_new[a]
        # clean
        a = a[a >= 0]
        idx_new[k] = a

    print("shrinking")
    return (
        assembly_dict,
        idx_new,
        spatial_axis_new,
        velocity_new,
        damping_new,
        src_sp_new,
        rcv_sp_new,
        removed
    )

def compute_halfgrid_coefficients(
    velocity_model: np.ndarray,
    damping_model: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the exact half-grid coefficients used by the forward stencil.
    """
    mu = velocity_model ** 2
    nu = damping_model
    mu_half = 0.5 * (mu[1:] + mu[:-1])
    nu_half = 0.5 * (nu[1:] + nu[:-1])
    return mu_half, nu_half


def apply_halfgrid_operator(
    z: np.ndarray,
    coeff_half: np.ndarray,
    dx: float,
) -> np.ndarray:
    """
    Apply A(z) = D( coeff_half * G z ),
    where G is the forward difference to half-grid and D is the divergence
    back to the nodal grid, consistent with the forward solver.
    """
    gz = (z[1:] - z[:-1]) / dx
    q = coeff_half * gz

    out = np.zeros_like(z)
    out[0] = q[0] / dx
    out[1:-1] = (q[1:] - q[:-1]) / dx
    out[-1] = -q[-1] / dx
    return out


def residual_to_simulation_time(
    observed_time: np.ndarray,
    simulation_time: np.ndarray,
    observed_residual: np.ndarray,
) -> np.ndarray:
    """
    Map residual from observed-time grid to simulation-time grid.
    For now this uses standard linear interpolation.
    """
    return np.interp(simulation_time, observed_time, observed_residual)


def build_adjoint_data_gradient(
    residual_sim: np.ndarray,
    receiver_spatial_function: np.ndarray,
) -> np.ndarray:
    """
    Build dJ/du from trace residual and fixed receiver spatial weighting.

    If y(t_n) = sum_i u(n,i) * r(i), then dJ/du(n,i) = residual(n) * r(i).
    """
    return residual_sim[:, None] * receiver_spatial_function[None, :]


def discrete_adjoint_1D_kelvin_voigt(
    dJdu: np.ndarray,
    velocity_model: np.ndarray,
    damping_model: np.ndarray,
    delta_x: float,
    delta_t: float,
) -> np.ndarray:
    """
    Discrete adjoint matched to the implemented forward recurrence:

        u^{n+1} = 2u^n - u^{n-1} + dt^2 A_mu u^n + dt A_nu(u^n-u^{n-1}) + source^n

    where A_mu(z) = D(mu_half * Gz) and A_nu(z) = D(nu_half * Gz).

    Boundary treatment mirrors the current forward implementation by zeroing
    the outer nodes when not using absorbing boundaries.
    """
    nt, nx = dJdu.shape
    lam = np.zeros((nt, nx), dtype=np.float64)

    mu_half, nu_half = compute_halfgrid_coefficients(
        velocity_model=velocity_model,
        damping_model=damping_model,
    )

    dt = float(delta_t)

    # Terminal conditions lam[nt-1] = 0 and lam[nt-2] is filled by recurrence if needed.
    for n in range(nt - 2, 0, -1):
        A_mu_lam_n   = apply_halfgrid_operator(lam[n],   mu_half, delta_x)
        A_nu_lam_n   = apply_halfgrid_operator(lam[n],   nu_half, delta_x)
        A_nu_lam_np1 = apply_halfgrid_operator(lam[n + 1], nu_half, delta_x)

        lam[n - 1] = (
            2.0 * lam[n]
            - lam[n + 1]
            + (dt ** 2) * A_mu_lam_n
            + dt * A_nu_lam_n
            - dt * A_nu_lam_np1
            + dJdu[n]
        )

        # Mirror current non-absorbing endpoint behavior
        lam[n - 1, 0] = 0.0
        lam[n - 1, -1] = 0.0

    return lam


def compute_forward_halfgrid_terms(
    wavefield_forward: np.ndarray,
    delta_x: float,
    delta_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the exact half-grid forward quantities used in the forward stencil:
      ux_half  = (u[i+1]-u[i])/dx
      uxt_half = ((u^n[i+1]-u^n[i]) - (u^{n-1}[i+1]-u^{n-1}[i])) / (dx*dt)
    """
    du = wavefield_forward[:, 1:] - wavefield_forward[:, :-1]   # (nt, nx-1)
    ux_half = du / delta_x

    uxt_half = np.zeros_like(ux_half)
    uxt_half[1:, :] = (du[1:, :] - du[:-1, :]) / (delta_x * delta_t)

    return ux_half, uxt_half


def compute_adjoint_halfgrid_terms(
    adjoint_wavefield: np.ndarray,
    delta_x: float,
) -> np.ndarray:
    """
    Compute D^T lambda on the half-grid:
        (lambda_i - lambda_{i+1}) / dx
    """
    return (adjoint_wavefield[:, :-1] - adjoint_wavefield[:, 1:]) / delta_x


def halfgrid_to_nodal_gradient(grad_half: np.ndarray) -> np.ndarray:
    """
    Map a half-grid gradient back to nodal values using the arithmetic-average
    relation implied by coeff_half = 0.5*(coeff[i] + coeff[i+1]).
    """
    nx = grad_half.size + 1
    grad_node = np.zeros(nx, dtype=grad_half.dtype)
    grad_node[0] = 0.5 * grad_half[0]
    grad_node[1:-1] = 0.5 * (grad_half[:-1] + grad_half[1:])
    grad_node[-1] = 0.5 * grad_half[-1]
    return grad_node


def compute_discrete_gradients_velocity_damping(
    wavefield_forward: np.ndarray,
    adjoint_wavefield: np.ndarray,
    velocity_model: np.ndarray,
    delta_x: float,
    delta_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Discrete gradients consistent with the implemented stencil.

    Returns
    -------
    gradient_vel : nodal gradient with respect to velocity v
    gradient_damp: nodal gradient with respect to damping nu
    """
    ux_half, uxt_half = compute_forward_halfgrid_terms(
        wavefield_forward=wavefield_forward,
        delta_x=delta_x,
        delta_t=delta_t,
    )

    adj_half = compute_adjoint_halfgrid_terms(
        adjoint_wavefield=adjoint_wavefield,
        delta_x=delta_x,
    )

    grad_mu_half = -(delta_t ** 2) * np.sum(adj_half * ux_half, axis=0)
    grad_nu_half = -(delta_t ** 2) * np.sum(adj_half * uxt_half, axis=0)

    grad_mu_node = halfgrid_to_nodal_gradient(grad_mu_half)
    grad_nu_node = halfgrid_to_nodal_gradient(grad_nu_half)

    # mu = v^2  =>  dJ/dv = 2 v dJ/dmu
    grad_v_node = 2.0 * velocity_model * grad_mu_node

    return grad_v_node, grad_nu_node

def fit_trace_scale(
    observed_waveform: np.ndarray,
    synthetic_waveform: np.ndarray,
    misfit_interval,
    eps: float = 1e-12,
    nonnegative: bool = True,
) -> float:
    if isinstance(misfit_interval, slice):
        d = observed_waveform[misfit_interval]
        s = synthetic_waveform[misfit_interval]
    else:
        interval = np.asarray(misfit_interval)
        if interval.dtype == np.bool_:
            mask = interval.reshape(-1)
            if mask.size != np.asarray(observed_waveform).size:
                raise ValueError(
                    "a boolean misfit_interval must match the waveform length"
                )
            d = observed_waveform[mask]
            s = synthetic_waveform[mask]
        else:
            idx = interval.astype(int, copy=False).reshape(-1)
            d = observed_waveform[idx]
            s = synthetic_waveform[idx]

    scale = float(np.dot(d, s) / (np.dot(s, s) + eps))
    return max(0.0, scale) if nonnegative else scale

def fit_spectral_amplitude_scale(
    observed_waveform: np.ndarray,
    synthetic_waveform: np.ndarray,
    observed_time: np.ndarray,
    misfit_interval,
    *,
    fmin: float | None = None,
    fmax: float | None = None,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Least-squares optimal scalar a for the amplitude spectrum:
        min_a || |D(f)| - a |S(f)| ||_2^2

    Returns
    -------
    scale : float
        Optimal spectral amplitude scale.
    freqs : ndarray
        Frequencies kept in the selected band.
    amp_obs : ndarray
        Observed amplitude spectrum on that band.
    amp_syn : ndarray
        Synthetic amplitude spectrum on that band.
    """
    if isinstance(misfit_interval, slice):
        d = observed_waveform[misfit_interval]
        s = synthetic_waveform[misfit_interval]
        t = observed_time[misfit_interval]
    else:
        idx = np.asarray(misfit_interval, dtype=int)
        d = observed_waveform[idx]
        s = synthetic_waveform[idx]
        t = observed_time[idx]

    if d.size < 2:
        raise ValueError("Misfit interval too short for spectral analysis.")

    dt = float(np.mean(np.diff(t)))
    freqs = np.fft.rfftfreq(d.size, d=dt)

    D = np.fft.rfft(d)
    S = np.fft.rfft(s)

    amp_obs = np.abs(D)
    amp_syn = np.abs(S)

    band = np.ones_like(freqs, dtype=bool)
    if fmin is not None:
        band &= freqs >= fmin
    if fmax is not None:
        band &= freqs <= fmax

    freqs = freqs[band]
    amp_obs = amp_obs[band]
    amp_syn = amp_syn[band]

    scale = np.dot(amp_obs, amp_syn) / (np.dot(amp_syn, amp_syn) + eps)

    return scale, freqs, amp_obs, amp_syn


def compute_log_spectral_amplitude_misfit(
    observed_waveform: np.ndarray,
    synthetic_waveform: np.ndarray,
    observed_time: np.ndarray,
    misfit_interval,
    *,
    fmin: float | None = None,
    fmax: float | None = None,
    fit_scale: bool = False,
    eps: float = 1e-12,
) -> float:
    """
    Band-limited log-spectral-amplitude misfit.

    If fit_scale=True, first computes the optimal scalar a such that
        |D(f)| ≈ a |S(f)|
    in least-squares sense, then compares log-amplitudes.

    Misfit:
        sum_f [ log(|D(f)|+eps) - log(a|S(f)|+eps) ]^2
    """
    if fit_scale:
        scale, freqs, amp_obs, amp_syn = fit_spectral_amplitude_scale(
            observed_waveform=observed_waveform,
            synthetic_waveform=synthetic_waveform,
            observed_time=observed_time,
            misfit_interval=misfit_interval,
            fmin=fmin,
            fmax=fmax,
            eps=eps,
        )
    else:
        if isinstance(misfit_interval, slice):
            d = observed_waveform[misfit_interval]
            s = synthetic_waveform[misfit_interval]
            t = observed_time[misfit_interval]
        else:
            idx = np.asarray(misfit_interval, dtype=int)
            d = observed_waveform[idx]
            s = synthetic_waveform[idx]
            t = observed_time[idx]

        if d.size < 2:
            raise ValueError("Misfit interval too short for spectral analysis.")

        dt = float(np.mean(np.diff(t)))
        freqs = np.fft.rfftfreq(d.size, d=dt)

        D = np.fft.rfft(d)
        S = np.fft.rfft(s)

        amp_obs = np.abs(D)
        amp_syn = np.abs(S)

        band = np.ones_like(freqs, dtype=bool)
        if fmin is not None:
            band &= freqs >= fmin
        if fmax is not None:
            band &= freqs <= fmax

        amp_obs = amp_obs[band]
        amp_syn = amp_syn[band]
        scale = 1.0

    log_obs = np.log(amp_obs + eps)
    log_syn = np.log(scale * amp_syn + eps)

    return float(np.sum((log_obs - log_syn) ** 2))


def compute_combined_initial_model_misfit(
    observed_waveform: np.ndarray,
    synthetic_waveform: np.ndarray,
    observed_time: np.ndarray,
    misfit_interval,
    *,
    alpha_time: float = 1.0,
    beta_spec: float = 0.1,
    time_fit_scale: bool = True,
    spec_fit_scale: bool = False,
    fmin: float | None = None,
    fmax: float | None = None,
    eps: float = 1e-12,
) -> tuple[float, float, float, float]:
    """
    Combined misfit for global / initial homogeneous-model search.

    Returns
    -------
    total_misfit : float
    time_misfit  : float
    spec_misfit  : float
    time_scale   : float
    """
    syn_for_time = synthetic_waveform.copy()

    if time_fit_scale:
        time_scale = fit_trace_scale(
            observed_waveform=observed_waveform,
            synthetic_waveform=syn_for_time,
            misfit_interval=misfit_interval,
            eps=eps,
        )
        syn_for_time *= time_scale
    else:
        time_scale = 1.0

    if isinstance(misfit_interval, slice):
        time_misfit = float(np.linalg.norm(
            observed_waveform[misfit_interval] - syn_for_time[misfit_interval]
        ))
    else:
        idx = np.asarray(misfit_interval, dtype=int)
        time_misfit = float(np.linalg.norm(
            observed_waveform[idx] - syn_for_time[idx]
        ))

    spec_misfit = compute_log_spectral_amplitude_misfit(
        observed_waveform=observed_waveform,
        synthetic_waveform=synthetic_waveform,
        observed_time=observed_time,
        misfit_interval=misfit_interval,
        fmin=fmin,
        fmax=fmax,
        fit_scale=spec_fit_scale,
        eps=eps,
    )

    total_misfit = alpha_time * time_misfit + beta_spec * spec_misfit

    return total_misfit, time_misfit, spec_misfit, time_scale
