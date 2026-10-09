"""Forward construction for the homogeneous steel reference experiment.

Two one-dimensional embeddings are kept separate because they answer
different physical questions:

``free_steel``
    A homogeneous steel interval with free steel--air faces.  Effective source
    and receiver operators act at (or just inside) those faces.  This is the
    model consistent with direct, 3*t0, 5*t0, ... propagation paths.

``full_area_pzt_stack``
    PLA--PZT--steel--PZT--PLA layers spanning the complete 1D section.  It is a
    useful sensitivity hypothesis for the direct packet, but it must not be
    mistaken for the literal three-dimensional geometry of a small PZT bonded
    to only part of a free steel face.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from lab_uw.fd1d import (
    finite_difference_1d_kelvin_voigt,
    finite_difference_1d_receiver_trace,
    sample_receiver,
)
from lab_uw.validation.reference.case import ReferenceCase
from lab_uw.simulation_setup import Grid1D, Receiver1D, SimulationTime, Source1D


EmbeddingMode = Literal["free_steel", "full_area_pzt_stack"]


def canonical_wave_type(wave_type: str) -> str:
    value = str(wave_type).lower().strip()
    if value.startswith("_"):
        value = value[1:]
    if value not in {"p", "s"}:
        raise ValueError("wave_type must be 'p', 's', '_p' or '_s'")
    return value


@dataclass(frozen=True)
class ReferenceGeometry:
    block_id: str
    wave_type: str
    steel_thickness_cm: float
    steel_velocity_cm_per_us: float
    pzt_thickness_cm: float = 0.0
    pzt_velocity_cm_per_us: float = 0.0
    pla_thickness_cm: float = 0.0
    pla_velocity_cm_per_us: float = 0.27

    def validate(self) -> None:
        canonical_wave_type(self.wave_type)
        positive = {
            "steel_thickness_cm": self.steel_thickness_cm,
            "steel_velocity_cm_per_us": self.steel_velocity_cm_per_us,
            "pla_velocity_cm_per_us": self.pla_velocity_cm_per_us,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.pla_thickness_cm) or self.pla_thickness_cm < 0:
            raise ValueError("pla_thickness_cm must be finite and non-negative")
        if not np.isfinite(self.pzt_thickness_cm) or self.pzt_thickness_cm < 0:
            raise ValueError("pzt_thickness_cm must be finite and non-negative")
        if not np.isfinite(self.pzt_velocity_cm_per_us) or self.pzt_velocity_cm_per_us < 0:
            raise ValueError("pzt_velocity_cm_per_us must be finite and non-negative")


def load_reference_geometry(
    metadata_path: Path | str,
    *,
    block_id: str,
    wave_type: str,
    steel_thickness_cm: Optional[float] = None,
) -> ReferenceGeometry:
    """Load the small subset of block metadata used by the 1D reference."""
    wave = canonical_wave_type(wave_type)
    with Path(metadata_path).open("r", encoding="utf-8") as stream:
        all_blocks = json.load(stream)
    try:
        block = all_blocks[block_id]
    except KeyError as exc:
        raise KeyError(f"block_id={block_id!r} is absent from {metadata_path}") from exc

    geometry = ReferenceGeometry(
        block_id=block_id,
        wave_type=wave,
        steel_thickness_cm=float(
            block["z"] if steel_thickness_cm is None else steel_thickness_cm
        ),
        steel_velocity_cm_per_us=float(block[f"velocity_{wave}"]),
        pzt_thickness_cm=float(block.get("pzt_layer_width", 0.0)),
        pzt_velocity_cm_per_us=float(block.get(f"pzt_velocity_{wave}", 0.0)),
        pla_thickness_cm=float(block.get("pla_layer_width", 0.0)),
        pla_velocity_cm_per_us=float(block.get(f"pla_velocity_{wave}", 0.27)),
    )
    geometry.validate()
    return geometry


@dataclass(frozen=True)
class ReferenceForwardConfig:
    embedding_mode: EmbeddingMode = "free_steel"
    frequency_cutoff_mhz: float = 12.0
    points_per_wavelength: int = 40
    cfl_factor: float = 0.4
    boundary_condition: str = "free"
    boundary_impedance: Optional[float | tuple[float, float]] = None
    source_injection: str = "body_force"
    receiver_quantity: str = "velocity"
    source_depth_fraction: float = 1.0
    receiver_depth_fraction: float = 1.0
    source_inset_cm: float = 0.0
    receiver_inset_cm: float = 0.0
    source_extension_cm: float = 0.0
    receiver_extension_cm: float = 0.0
    interpolation_radius_cells: int = 4
    source_normalization: str = "auto"
    receiver_normalization: str = "sum"
    spatial_boundary_mode: str = "truncate"
    source_amplitude: float = 1.0

    def validate(self, geometry: ReferenceGeometry) -> None:
        if self.embedding_mode not in {"free_steel", "full_area_pzt_stack"}:
            raise ValueError(f"unknown embedding_mode={self.embedding_mode!r}")
        if self.frequency_cutoff_mhz <= 0 or self.points_per_wavelength < 2:
            raise ValueError("frequency cutoff must be positive and ppw at least 2")
        if not (0 < self.cfl_factor < 1):
            raise ValueError("cfl_factor must lie strictly between zero and one")
        if self.interpolation_radius_cells < 1:
            raise ValueError("interpolation_radius_cells must be positive")
        if self.source_extension_cm < 0 or self.receiver_extension_cm < 0:
            raise ValueError("source/receiver extensions cannot be negative")
        if self.embedding_mode == "free_steel":
            if not (0 <= self.source_inset_cm < geometry.steel_thickness_cm):
                raise ValueError("source_inset_cm lies outside the steel block")
            if not (0 <= self.receiver_inset_cm < geometry.steel_thickness_cm):
                raise ValueError("receiver_inset_cm lies outside the steel block")
            if self.source_inset_cm + self.receiver_inset_cm >= geometry.steel_thickness_cm:
                raise ValueError("source and receiver positions cross")
        else:
            if geometry.pzt_thickness_cm <= 0 or geometry.pzt_velocity_cm_per_us <= 0:
                raise ValueError(
                    "full_area_pzt_stack requires positive PZT thickness and velocity"
                )
            if not (0 <= self.source_depth_fraction <= 1):
                raise ValueError("source_depth_fraction must be in [0, 1]")
            if not (0 <= self.receiver_depth_fraction <= 1):
                raise ValueError("receiver_depth_fraction must be in [0, 1]")

    def resolved_source_normalization(self) -> str:
        if self.source_normalization != "auto":
            return self.source_normalization
        return "integral" if self.source_injection == "body_force" else "sum"


@dataclass
class ReferenceForwardResult:
    observed_time_us: np.ndarray
    synthetic_waveform: np.ndarray
    simulation_time_us: np.ndarray
    spatial_axis_cm: np.ndarray
    velocity_model_cm_per_us: np.ndarray
    source_spatial_function: np.ndarray
    receiver_spatial_function: np.ndarray
    source_position_cm: float
    receiver_position_cm: float
    config: ReferenceForwardConfig
    geometry: ReferenceGeometry
    wavefield: Optional[np.ndarray] = None

    def metadata(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "geometry": asdict(self.geometry),
            "realized_dx_cm": float(np.diff(self.spatial_axis_cm)[0]),
            "realized_dt_us": float(np.diff(self.simulation_time_us)[0]),
            "source_position_cm": self.source_position_cm,
            "receiver_position_cm": self.receiver_position_cm,
        }


def _layered_velocity(
    spatial_axis: np.ndarray,
    geometry: ReferenceGeometry,
) -> np.ndarray:
    layers = np.asarray(
        [
            geometry.pla_thickness_cm,
            geometry.pzt_thickness_cm,
            geometry.steel_thickness_cm,
            geometry.pzt_thickness_cm,
            geometry.pla_thickness_cm,
        ],
        dtype=float,
    )
    velocities = np.asarray(
        [
            geometry.pla_velocity_cm_per_us,
            geometry.pzt_velocity_cm_per_us,
            geometry.steel_velocity_cm_per_us,
            geometry.pzt_velocity_cm_per_us,
            geometry.pla_velocity_cm_per_us,
        ],
        dtype=float,
    )
    ends = np.cumsum(layers)
    # Right-continuous assignment gives every node exactly one material.
    layer_index = np.searchsorted(ends, spatial_axis, side="right")
    layer_index = np.minimum(layer_index, velocities.size - 1)
    return velocities[layer_index]


def _domain_and_positions(
    geometry: ReferenceGeometry,
    config: ReferenceForwardConfig,
) -> tuple[float, float, float]:
    if config.embedding_mode == "free_steel":
        total_length = geometry.steel_thickness_cm
        source_position = config.source_inset_cm
        receiver_position = total_length - config.receiver_inset_cm
    else:
        total_length = (
            geometry.steel_thickness_cm
            + 2.0 * geometry.pzt_thickness_cm
            + 2.0 * geometry.pla_thickness_cm
        )
        source_position = (
            geometry.pla_thickness_cm
            + config.source_depth_fraction * geometry.pzt_thickness_cm
        )
        receiver_position = (
            total_length
            - geometry.pla_thickness_cm
            - config.receiver_depth_fraction * geometry.pzt_thickness_cm
        )
    return total_length, source_position, receiver_position


def run_reference_forward(
    case: ReferenceCase,
    geometry: ReferenceGeometry,
    config: ReferenceForwardConfig,
    *,
    stop_time_us: Optional[float] = None,
    keep_wavefield: bool = False,
) -> ReferenceForwardResult:
    """Run one explicit 1D reference model using the effective wavelet."""
    case.validate()
    geometry.validate()
    config.validate(geometry)
    if not np.isclose(
        case.block_thickness_cm,
        geometry.steel_thickness_cm,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "reference JSON and block metadata disagree on steel thickness: "
            f"{case.block_thickness_cm:g} vs {geometry.steel_thickness_cm:g} cm"
        )

    total_length, source_position, receiver_position = _domain_and_positions(
        geometry, config
    )
    if config.embedding_mode == "full_area_pzt_stack":
        active_velocities = [
            geometry.steel_velocity_cm_per_us,
            geometry.pzt_velocity_cm_per_us,
        ]
        if geometry.pla_thickness_cm > 0:
            active_velocities.append(geometry.pla_velocity_cm_per_us)
        cmin = min(active_velocities)
        cmax = max(active_velocities)
    else:
        cmin = geometry.steel_velocity_cm_per_us
        cmax = geometry.steel_velocity_cm_per_us

    grid = Grid1D(
        cmin=cmin,
        fmax=config.frequency_cutoff_mhz,
        grid_len=total_length,
        ppt=config.points_per_wavelength,
    )
    x = grid.spatial_axis
    if config.embedding_mode == "free_steel":
        velocity = np.full(x.size, geometry.steel_velocity_cm_per_us, dtype=float)
    else:
        velocity = _layered_velocity(x, geometry)
    damping = np.zeros_like(velocity)

    observed_relative = case.observed_time_us - case.observed_time_us[0]
    if stop_time_us is not None:
        if stop_time_us <= observed_relative[0]:
            raise ValueError("stop_time_us must be after the first observed sample")
        last = int(np.searchsorted(observed_relative, stop_time_us, side="right"))
        observed_relative = observed_relative[:last]
    simulation_clock = SimulationTime(
        observed_time=observed_relative,
        dx=grid.dx,
        max_velocity=cmax,
        max_alpha=0.0,
        cfl_factor=config.cfl_factor,
    )

    source = Source1D(
        stf_time=case.wavelet_time_relative_us,
        stf_waveform=config.source_amplitude * case.wavelet_waveform,
        position=source_position,
        pzt_layer_width=geometry.pzt_thickness_cm,
        radius=config.interpolation_radius_cells,
        extension=config.source_extension_cm,
        normalization=config.resolved_source_normalization(),
        boundary_mode=config.spatial_boundary_mode,
    )
    source.interpolate_time_function(
        dt=simulation_clock.dt,
        simulation_time=simulation_clock.simulation_time,
    )
    source.create_spatial_function(x, grid.dx)

    receiver = Receiver1D(
        position=receiver_position,
        pzt_layer_width=geometry.pzt_thickness_cm,
        radius=config.interpolation_radius_cells,
        extension=config.receiver_extension_cm,
        normalization=config.receiver_normalization,
        boundary_mode=config.spatial_boundary_mode,
    )
    receiver.create_spatial_function(x, grid.dx)

    common_solver_args = {
        "num_x": x.size,
        "delta_x": grid.dx,
        "num_t": simulation_clock.num_t,
        "delta_t": simulation_clock.dt,
        "source_spatial_function": source.spatial_function,
        "source_time_function": source.time_function,
        "velocity_model": velocity,
        "damping_model": damping,
        "boundary_condition": config.boundary_condition,
        "boundary_impedance": config.boundary_impedance,
        "source_injection": config.source_injection,
    }
    if keep_wavefield:
        wavefield = finite_difference_1d_kelvin_voigt(**common_solver_args)
        simulated = sample_receiver(
            wavefield,
            receiver.spatial_function,
            quantity=config.receiver_quantity,
            delta_t=simulation_clock.dt,
            delta_x=grid.dx,
            velocity_model=velocity,
            damping_model=damping,
        )
    else:
        wavefield = None
        simulated = finite_difference_1d_receiver_trace(
            **common_solver_args,
            receiver_spatial_function=receiver.spatial_function,
            receiver_quantity=config.receiver_quantity,
        )
    synthetic = np.interp(
        observed_relative,
        simulation_clock.simulation_time,
        simulated,
    )

    result = ReferenceForwardResult(
        observed_time_us=case.observed_time_us[: observed_relative.size].copy(),
        synthetic_waveform=synthetic,
        simulation_time_us=simulation_clock.simulation_time.copy(),
        spatial_axis_cm=x.copy(),
        velocity_model_cm_per_us=velocity.copy(),
        source_spatial_function=source.spatial_function.copy(),
        receiver_spatial_function=receiver.spatial_function.copy(),
        source_position_cm=float(source_position),
        receiver_position_cm=float(receiver_position),
        config=config,
        geometry=geometry,
        wavefield=wavefield,
    )
    return result
