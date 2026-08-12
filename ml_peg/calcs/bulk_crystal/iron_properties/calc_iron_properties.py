"""Run the BCC iron properties benchmark."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from warnings import warn

from ase.build import bulk
from ase.constraints import FixedLine
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import BFGS
import numpy as np
import pandas as pd
import pytest

from ml_peg.calcs.bulk_crystal.iron_properties.iron_utils import (
    EV_PER_A2_TO_J_PER_M2,
    EV_PER_A3_TO_GPA,
    apply_voigt_strain,
    calculate_surface_energy,
    create_bain_cell,
    create_bcc_supercell,
    create_sfe_110_structure,
    create_sfe_112_structure,
    create_surface_100,
    create_surface_110,
    create_surface_111,
    create_surface_112,
    fit_eos,
    relax_volume_isotropic,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator

MODELS = load_models(current_models)
OUT_PATH = Path(__file__).parent / "outputs"

BENCHMARK_VERSION = 2
REFERENCE_LATTICE_PARAMETER = 2.834
EOS_NUM_POINTS = 30
BFGS_FMAX = 1e-5
BFGS_MAX_ITER = 100
ELASTIC_STRAIN = 1e-5
ELASTIC_SUPERCELL_SIZE = (4, 4, 4)
ELASTIC_ATOM_JIGGLE = 1e-5
BAIN_NUM_POINTS = 65
VACANCY_SUPERCELL_SIZE = (3, 3, 3)
SURFACE_VACUUM = 10.0
SFE_STEP_SIZE = 0.04
TS_MAX_SEPARATION = 5.0
TS_STEP_SIZE = 0.05


SURFACE_CONFIG = {
    "100": {
        "create_fn": create_surface_100,
        "layers": 10,
        "area_axes": (0, 1),
        "vacuum": SURFACE_VACUUM,
    },
    "110": {
        "create_fn": create_surface_110,
        "layers": 10,
        "area_axes": (0, 1),
        "vacuum": SURFACE_VACUUM,
    },
    "111": {
        "create_fn": create_surface_111,
        "size": (3, 15, 3),
        "area_axes": (0, 2),
        "vacuum": SURFACE_VACUUM,
    },
    "112": {
        "create_fn": create_surface_112,
        "layers": 15,
        "area_axes": (0, 1),
        "vacuum": 5.0,
    },
}
SFE_CONFIG = {
    "110": {"create_fn": create_sfe_110_structure, "axis": 1},
    "112": {"create_fn": create_sfe_112_structure, "axis": 2},
}
TS_CONFIG = {
    "100": lambda a: create_surface_100(a, layers=36, vacuum=0.0),
    "110": lambda a: create_surface_110(a, layers=10, vacuum=0.0),
}


def _set_iron_info(atoms: Atoms) -> None:
    """
    Annotate an iron structure with model metadata.

    Parameters
    ----------
    atoms
        Structure to annotate.
    """
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1


def _relax(target: Any, label: str) -> dict[str, Any]:
    """
    Run BFGS and return serializable convergence diagnostics.

    Parameters
    ----------
    target
        ASE atoms or filter to optimize.
    label
        Description used in warnings.

    Returns
    -------
    dict[str, Any]
        Convergence status, step count, final force, and error message.
    """
    status: dict[str, Any] = {
        "converged": False,
        "steps": 0,
        "max_force": None,
        "error": None,
    }
    try:
        optimizer = BFGS(target, logfile=None)
        status["converged"] = bool(optimizer.run(fmax=BFGS_FMAX, steps=BFGS_MAX_ITER))
        status["steps"] = int(optimizer.nsteps)
        forces = np.asarray(target.get_forces())
        if forces.size:
            status["max_force"] = float(
                np.max(np.linalg.norm(forces.reshape(-1, 3), axis=1))
            )
        if not status["converged"]:
            warn(f"{label} did not converge in {BFGS_MAX_ITER} steps", stacklevel=2)
    except Exception as exc:
        status["error"] = str(exc)
        warn(f"{label} failed: {exc}", stacklevel=2)
    return status


def _energy(atoms: Atoms, label: str) -> float:
    """
    Evaluate a potential energy, returning NaN on failure.

    Parameters
    ----------
    atoms
        Structure to evaluate.
    label
        Description used in warnings.

    Returns
    -------
    float
        Potential energy or NaN.
    """
    try:
        return float(atoms.get_potential_energy())
    except Exception as exc:
        warn(f"{label} failed: {exc}", stacklevel=2)
        return float("nan")


def _run_section(
    label: str, calculation: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    """
    Run one independent benchmark section without aborting later sections.

    Parameters
    ----------
    label
        Section name used in warnings.
    calculation
        Callable that performs the section.

    Returns
    -------
    dict[str, Any]
        Section results or a failure record.
    """
    try:
        return calculation()
    except Exception as exc:
        warn(f"{label} failed: {exc}", stacklevel=2)
        return {"status": {"converged": False, "error": str(exc)}}


def _json_safe(value: Any) -> Any:
    """
    Convert nested NumPy and nonfinite values to strict JSON values.

    Parameters
    ----------
    value
        Value to convert.

    Returns
    -------
    Any
        Strictly JSON-serializable value.
    """
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _finite(value: Any) -> bool:
    """
    Return whether a value is a finite scalar.

    Parameters
    ----------
    value
        Value to inspect.

    Returns
    -------
    bool
        Whether the value is finite.
    """
    try:
        return bool(np.isfinite(value))
    except TypeError:
        return False


def _save_curve(write_dir: Path, name: str, data: dict[str, list[Any]]) -> None:
    """
    Save one curve as CSV.

    Parameters
    ----------
    write_dir
        Output directory.
    name
        File stem.
    data
        Curve columns.
    """
    pd.DataFrame(_json_safe(data)).to_csv(write_dir / f"{name}.csv", index=False)


def run_eos_calculation(calc: Calculator) -> dict[str, Any]:
    """
    Calculate and fit the BCC energy-volume curve.

    Parameters
    ----------
    calc
        ASE calculator.

    Returns
    -------
    dict[str, Any]
        EOS fit, curve values, and convergence diagnostics.
    """
    lattice_params = np.array(
        [
            REFERENCE_LATTICE_PARAMETER - 0.05 + 0.1 / EOS_NUM_POINTS * index
            for index in range(1, EOS_NUM_POINTS + 1)
        ]
    )
    volumes: list[float] = []
    energies: list[float] = []
    statuses: list[dict[str, Any]] = []

    for lattice_parameter in lattice_params:
        atoms = bulk("Fe", "bcc", a=lattice_parameter, cubic=True)
        _set_iron_info(atoms)
        atoms.calc = calc
        statuses.append(_relax(atoms, f"EOS relaxation at a={lattice_parameter:.5f}"))
        volumes.append(float(atoms.get_volume() / len(atoms)))
        energies.append(_energy(atoms, "EOS energy") / len(atoms))

    eos = fit_eos(np.asarray(volumes), np.asarray(energies))
    return {
        "volumes": volumes,
        "energies": energies,
        "lattice_params": lattice_params.tolist(),
        **eos,
        "status": statuses,
    }


def run_elastic_calculation(
    calc: Calculator, lattice_parameter: float
) -> dict[str, Any]:
    """
    Calculate cubic elastic constants from central stress differences.

    Parameters
    ----------
    calc
        ASE calculator.
    lattice_parameter
        BCC lattice parameter in Angstrom.

    Returns
    -------
    dict[str, Any]
        Elastic constants, tensor, and convergence diagnostics.
    """
    atoms_ref = create_bcc_supercell(lattice_parameter, ELASTIC_SUPERCELL_SIZE)
    _set_iron_info(atoms_ref)
    atoms_ref.calc = calc
    status: dict[str, Any] = {
        "cell": _relax(FrechetCellFilter(atoms_ref), "elastic cell relaxation"),
        "strains": [],
    }

    rng = np.random.default_rng(seed=87287)
    atoms_ref.positions += rng.uniform(
        -ELASTIC_ATOM_JIGGLE, ELASTIC_ATOM_JIGGLE, atoms_ref.positions.shape
    )

    elastic_tensor = np.full((6, 6), np.nan)
    for direction in range(1, 7):
        stresses = []
        direction_status = {"direction": direction}
        for sign, name in ((1, "positive"), (-1, "negative")):
            strained = apply_voigt_strain(atoms_ref, direction, sign * ELASTIC_STRAIN)
            _set_iron_info(strained)
            strained.calc = calc
            direction_status[name] = _relax(
                strained, f"elastic direction {direction} {name} relaxation"
            )
            try:
                stresses.append(np.asarray(strained.get_stress(voigt=True)))
            except Exception as exc:
                warn(f"Elastic stress evaluation failed: {exc}", stacklevel=2)
                stresses.append(np.full(6, np.nan))
        elastic_tensor[:, direction - 1] = (
            (stresses[0] - stresses[1]) / (2 * ELASTIC_STRAIN) * EV_PER_A3_TO_GPA
        )
        status["strains"].append(direction_status)

    elastic_tensor = 0.5 * (elastic_tensor + elastic_tensor.T)
    c11 = float(np.mean(np.diag(elastic_tensor)[:3]))
    c12 = float(
        np.mean([elastic_tensor[0, 1], elastic_tensor[0, 2], elastic_tensor[1, 2]])
    )
    c44 = float(np.mean(np.diag(elastic_tensor)[3:]))
    return {
        "C11": c11,
        "C12": c12,
        "C44": c44,
        "bulk_modulus": (c11 + 2 * c12) / 3,
        "C_matrix": elastic_tensor.tolist(),
        "status": status,
    }


def run_bain_path_calculation(
    calc: Calculator, lattice_parameter: float
) -> dict[str, Any]:
    """
    Calculate the volume-relaxed Bain path, including exact FCC.

    Parameters
    ----------
    calc
        ASE calculator.
    lattice_parameter
        BCC lattice parameter in Angstrom.

    Returns
    -------
    dict[str, Any]
        Bain curve, endpoint energies, and convergence diagnostics.
    """
    regular_grid = np.array(
        [0.7 + 0.02 * index for index in range(1, BAIN_NUM_POINTS + 1)]
    )
    target_ratios = np.sort(np.append(regular_grid, np.sqrt(2)))
    actual_ratios: list[float] = []
    energies: list[float] = []
    statuses: list[dict[str, Any]] = []

    for ratio in target_ratios:
        atoms = create_bain_cell(lattice_parameter, ratio)
        _set_iron_info(atoms)
        atoms.calc = calc
        try:
            relaxed, volume_status = relax_volume_isotropic(atoms, calc)
        except Exception as exc:
            warn(
                f"Bain volume relaxation at c/a={ratio:.6f} failed: {exc}", stacklevel=2
            )
            relaxed = atoms
            volume_status = {
                "converged": False,
                "scale": 1.0,
                "boundary_hit": False,
                "pressure_GPa": None,
                "error": str(exc),
            }

        atomic_status = _relax(relaxed, f"Bain atomic relaxation at c/a={ratio:.6f}")
        actual_ratio = float(relaxed.cell[2, 2] / relaxed.cell[1, 1])
        actual_ratios.append(actual_ratio)
        energies.append(_energy(relaxed, "Bain energy") / len(relaxed))
        statuses.append(
            {
                "target_ca_ratio": float(ratio),
                "actual_ca_ratio": actual_ratio,
                "volume": volume_status,
                "atoms": atomic_status,
            }
        )

    energy_array = np.asarray(energies)
    ratio_array = np.asarray(actual_ratios)
    bcc_index = int(np.argmin(np.abs(ratio_array - 1.0)))
    fcc_index = int(np.argmin(np.abs(ratio_array - np.sqrt(2))))
    bcc_energy = energy_array[bcc_index]
    relative_energies = (energy_array - bcc_energy) * 1000
    return {
        "ca_ratios": actual_ratios,
        "energies": energies,
        "energies_meV": relative_energies.tolist(),
        "E_bcc_meV": float(relative_energies[bcc_index]),
        "E_fcc_meV": float(relative_energies[fcc_index]),
        "delta_E_meV": float((energy_array[fcc_index] - bcc_energy) * 1000),
        "status": statuses,
    }


def run_vacancy_calculation(
    calc: Calculator, lattice_parameter: float
) -> dict[str, Any]:
    """
    Calculate the fixed-cell 3x3x3 vacancy formation energy.

    Parameters
    ----------
    calc
        ASE calculator.
    lattice_parameter
        BCC lattice parameter in Angstrom.

    Returns
    -------
    dict[str, Any]
        Vacancy energies, atom counts, and convergence diagnostics.
    """
    perfect = create_bcc_supercell(lattice_parameter, VACANCY_SUPERCELL_SIZE)
    _set_iron_info(perfect)
    perfect.calc = calc
    atom_count = len(perfect)
    perfect_energy = _energy(perfect, "perfect vacancy reference energy")
    cohesive_energy = perfect_energy / atom_count

    defect = perfect.copy()
    del defect[0]
    _set_iron_info(defect)
    defect.calc = calc
    status = _relax(defect, "vacancy relaxation")
    defect_energy = _energy(defect, "vacancy defect energy")
    vacancy_energy = defect_energy - perfect_energy + cohesive_energy
    return {
        "E_vac": vacancy_energy,
        "E_coh": cohesive_energy,
        "E_perfect": perfect_energy,
        "E_defect": defect_energy,
        "perfect_atoms": atom_count,
        "defect_atoms": len(defect),
        "status": status,
    }


def run_surface_calculations(
    calc: Calculator, lattice_parameter: float
) -> dict[str, Any]:
    """
    Calculate relaxed surface energies for four low-index planes.

    Parameters
    ----------
    calc
        ASE calculator.
    lattice_parameter
        BCC lattice parameter in Angstrom.

    Returns
    -------
    dict[str, Any]
        Surface energies and convergence diagnostics by orientation.
    """
    surfaces: dict[str, float] = {}
    statuses: dict[str, Any] = {}
    for name, config in SURFACE_CONFIG.items():
        create_fn = config["create_fn"]
        if "size" in config:
            bulk_kwargs = {"size": config["size"], "vacuum": 0.0}
            slab_kwargs = {"size": config["size"], "vacuum": config["vacuum"]}
        else:
            bulk_kwargs = {"layers": config["layers"], "vacuum": 0.0}
            slab_kwargs = {
                "layers": config["layers"],
                "vacuum": config["vacuum"],
            }

        try:
            bulk_atoms = create_fn(lattice_parameter, **bulk_kwargs)
            slab = create_fn(lattice_parameter, **slab_kwargs)
        except Exception as exc:
            warn(f"Surface {name} construction failed: {exc}", stacklevel=2)
            surfaces[name] = float("nan")
            statuses[name] = {"converged": False, "error": str(exc)}
            continue

        _set_iron_info(bulk_atoms)
        bulk_atoms.calc = calc
        bulk_energy = _energy(bulk_atoms, f"surface {name} bulk energy")
        axes = config["area_axes"]
        area = float(
            np.linalg.norm(np.cross(bulk_atoms.cell[axes[0]], bulk_atoms.cell[axes[1]]))
        )

        _set_iron_info(slab)
        slab.calc = calc
        statuses[name] = _relax(slab, f"surface {name} relaxation")
        slab_energy = _energy(slab, f"surface {name} slab energy")
        surfaces[name] = calculate_surface_energy(slab_energy, bulk_energy, area)

    return {
        **{f"gamma_{name}": value for name, value in surfaces.items()},
        "status": statuses,
    }


def _sfe_displacements(lattice_parameter: float) -> np.ndarray:
    """
    Build a 0.04 Angstrom grid with an exact Burgers-vector endpoint.

    Parameters
    ----------
    lattice_parameter
        BCC lattice parameter in Angstrom.

    Returns
    -------
    numpy.ndarray
        Absolute displacement grid.
    """
    burgers_vector = lattice_parameter * np.sqrt(3) / 2
    regular = np.arange(0.0, burgers_vector, SFE_STEP_SIZE)
    return np.append(regular, burgers_vector)


def run_sfe_calculation(
    calc: Calculator, lattice_parameter: float, sfe_type: str
) -> dict[str, Any]:
    """
    Calculate one generalized stacking-fault energy curve.

    Parameters
    ----------
    calc
        ASE calculator.
    lattice_parameter
        BCC lattice parameter in Angstrom.
    sfe_type
        Slip-plane orientation, ``110`` or ``112``.

    Returns
    -------
    dict[str, Any]
        GSFE curve, maximum, and point-wise convergence diagnostics.
    """
    displacements = _sfe_displacements(lattice_parameter)
    config = SFE_CONFIG[sfe_type]
    try:
        atoms = config["create_fn"](lattice_parameter)
    except Exception as exc:
        warn(f"SFE {sfe_type} construction failed: {exc}", stacklevel=2)
        return {
            "displacements": displacements.tolist(),
            "displacement_fractions": (displacements / displacements[-1]).tolist(),
            "sfe_J_per_m2": [float("nan")] * len(displacements),
            "max_sfe": float("nan"),
            "status": {"converged": False, "error": str(exc)},
        }

    _set_iron_info(atoms)
    atoms.calc = calc
    initial_status = _relax(atoms, f"SFE {sfe_type} initial relaxation")
    initial_energy = _energy(atoms, f"SFE {sfe_type} initial energy")
    area = float(np.linalg.norm(np.cross(atoms.cell[1], atoms.cell[2])))
    x_mid = np.linalg.norm(atoms.cell[0]) / 2 + 0.1
    upper_indices = np.flatnonzero(atoms.positions[:, 0] <= x_mid)
    constraints = [FixedLine(index, direction=[1, 0, 0]) for index in range(len(atoms))]

    values = [0.0 if _finite(initial_energy) else float("nan")]
    point_statuses: list[dict[str, Any]] = [initial_status]
    last_valid = atoms.copy()
    last_displacement = 0.0
    for displacement in displacements[1:]:
        trial = last_valid.copy()
        trial.calc = calc
        positions = trial.get_positions()
        positions[upper_indices, config["axis"]] += displacement - last_displacement
        trial.set_positions(positions)
        trial.set_constraint(constraints)
        status = _relax(trial, f"SFE {sfe_type} at {displacement:.5f} Angstrom")
        energy = _energy(trial, f"SFE {sfe_type} energy")
        trial.set_constraint()
        valid = status["error"] is None and _finite(energy) and _finite(initial_energy)
        if valid:
            values.append(
                (energy - initial_energy) / (2 * area) * EV_PER_A2_TO_J_PER_M2
            )
            last_valid = trial
            last_displacement = float(displacement)
        else:
            values.append(float("nan"))
        point_statuses.append(status)

    finite_values = np.asarray(values)[np.isfinite(values)]
    return {
        "displacements": displacements.tolist(),
        "displacement_fractions": (displacements / displacements[-1]).tolist(),
        "sfe_J_per_m2": values,
        "max_sfe": float(np.max(finite_values)) if finite_values.size else float("nan"),
        "status": point_statuses,
    }


def run_ts_calculation(
    calc: Calculator, lattice_parameter: float, direction: str
) -> dict[str, Any]:
    """
    Calculate an unrelaxed traction-separation curve from energy differences.

    Parameters
    ----------
    calc
        ASE calculator.
    lattice_parameter
        BCC lattice parameter in Angstrom.
    direction
        Cleavage-plane orientation, ``100`` or ``110``.

    Returns
    -------
    dict[str, Any]
        Energy and traction curves with point validity diagnostics.
    """
    energy_separations = np.arange(
        0.0, TS_MAX_SEPARATION + TS_STEP_SIZE / 2, TS_STEP_SIZE
    )
    energies: list[float] = []
    statuses: list[dict[str, Any]] = []
    area = float("nan")

    for separation in energy_separations:
        atoms = TS_CONFIG[direction](lattice_parameter)
        _set_iron_info(atoms)
        atoms.calc = calc
        cell = atoms.cell.array.copy()
        area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
        z_mid = cell[2, 2] / 2 - 0.1
        upper_indices = np.flatnonzero(atoms.positions[:, 2] >= z_mid)
        cell[2, 2] += separation
        atoms.set_cell(cell, scale_atoms=False)
        atoms.positions[upper_indices, 2] += separation
        energy = _energy(atoms, f"TS {direction} energy at {separation:.2f} Angstrom")
        energies.append(energy)
        statuses.append({"valid": _finite(energy), "separation": float(separation)})

    energy_array = np.asarray(energies)
    traction = np.diff(energy_array) / (area * TS_STEP_SIZE) * EV_PER_A3_TO_GPA
    traction_separations = np.concatenate(
        ([0.0], energy_separations[:-1] + TS_STEP_SIZE / 2)
    )
    traction = np.concatenate(([0.0], traction))
    positive = traction[np.isfinite(traction) & (traction > 0)]
    return {
        "energy_separations": energy_separations.tolist(),
        "energies": energies,
        "separations": traction_separations.tolist(),
        "traction": traction.tolist(),
        "max_traction": float(np.max(positive)) if positive.size else float("nan"),
        "status": statuses,
    }


def _write_structures(write_dir: Path, lattice_parameter: float) -> None:
    """
    Write representative benchmark structures for visualization.

    Parameters
    ----------
    write_dir
        Model output directory.
    lattice_parameter
        BCC lattice parameter in Angstrom.
    """
    structures_dir = write_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    equilibrium = bulk("Fe", "bcc", a=lattice_parameter, cubic=True)
    _set_iron_info(equilibrium)
    equilibrium.info["description"] = f"BCC Fe equilibrium (a0={lattice_parameter:.4f})"
    write(structures_dir / "equilibrium_bcc.extxyz", equilibrium)

    vacancy = create_bcc_supercell(lattice_parameter, VACANCY_SUPERCELL_SIZE)
    del vacancy[0]
    _set_iron_info(vacancy)
    vacancy.info["description"] = "BCC Fe vacancy supercell (3x3x3)"
    write(structures_dir / "vacancy.extxyz", vacancy)

    for name, config in SURFACE_CONFIG.items():
        try:
            create_fn = config["create_fn"]
            kwargs = (
                {"size": config["size"], "vacuum": config["vacuum"]}
                if "size" in config
                else {"layers": config["layers"], "vacuum": config["vacuum"]}
            )
            slab = create_fn(lattice_parameter, **kwargs)
            _set_iron_info(slab)
            slab.info["description"] = f"BCC Fe ({name}) surface slab"
            write(structures_dir / f"surface_{name}.extxyz", slab)
        except Exception as exc:
            warn(f"Surface {name} visualization failed: {exc}", stacklevel=2)


def run_iron_properties(model_name: str, model: Any) -> None:
    """
    Run the complete Iron Properties v2 benchmark for one model.

    Parameters
    ----------
    model_name
        Registered model name.
    model
        Model wrapper that supplies an ASE calculator.
    """
    calc = model.get_calculator(precision="high")
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"benchmark_version": BENCHMARK_VERSION}

    print(f"[{model_name}] Running EOS calculation...")
    eos = _run_section("EOS", lambda: run_eos_calculation(calc))
    results["eos"] = eos
    fitted_a0 = eos.get("a0")
    lattice_parameter = (
        float(fitted_a0) if _finite(fitted_a0) else REFERENCE_LATTICE_PARAMETER
    )
    results["lattice_parameter_source"] = (
        "eos" if _finite(fitted_a0) else "reference_fallback"
    )
    eos_energies = np.asarray(eos.get("energies", []), dtype=float)
    relative_eos = (
        (eos_energies - np.nanmin(eos_energies)) * 1000
        if np.isfinite(eos_energies).any()
        else np.full_like(eos_energies, np.nan)
    )
    _save_curve(
        write_dir,
        "eos_curve",
        {"volume": eos.get("volumes", []), "energy_meV": relative_eos.tolist()},
    )

    print(f"[{model_name}] Running elastic constants calculation...")
    elastic = _run_section(
        "elastic constants",
        lambda: run_elastic_calculation(calc, lattice_parameter),
    )
    results["elastic"] = elastic

    print(f"[{model_name}] Running Bain path calculation...")
    bain = _run_section(
        "Bain path", lambda: run_bain_path_calculation(calc, lattice_parameter)
    )
    results["bain_path"] = bain
    _save_curve(
        write_dir,
        "bain_path",
        {
            "ca_ratio": bain.get("ca_ratios", []),
            "energy_meV": bain.get("energies_meV", []),
        },
    )

    print(f"[{model_name}] Running vacancy calculation...")
    vacancy = _run_section(
        "vacancy", lambda: run_vacancy_calculation(calc, lattice_parameter)
    )
    results["vacancy"] = vacancy

    print(f"[{model_name}] Running surface calculations...")
    surfaces = _run_section(
        "surfaces", lambda: run_surface_calculations(calc, lattice_parameter)
    )
    results["surfaces"] = surfaces

    for sfe_type in SFE_CONFIG:
        print(f"[{model_name}] Running SFE {sfe_type} calculation...")
        sfe = _run_section(
            f"SFE {sfe_type}",
            lambda sfe_type=sfe_type: run_sfe_calculation(
                calc, lattice_parameter, sfe_type
            ),
        )
        results[f"sfe_{sfe_type}"] = sfe
        _save_curve(
            write_dir,
            f"sfe_{sfe_type}_curve",
            {
                "displacement": sfe.get("displacements", []),
                "displacement_fraction": sfe.get("displacement_fractions", []),
                "sfe_J_per_m2": sfe.get("sfe_J_per_m2", []),
            },
        )

    for direction in TS_CONFIG:
        print(f"[{model_name}] Running T-S ({direction}) calculation...")
        traction = _run_section(
            f"T-S {direction}",
            lambda direction=direction: run_ts_calculation(
                calc, lattice_parameter, direction
            ),
        )
        results[f"ts_{direction}"] = traction
        _save_curve(
            write_dir,
            f"ts_{direction}_curve",
            {
                "separation": traction.get("separations", []),
                "traction": traction.get("traction", []),
            },
        )
        _save_curve(
            write_dir,
            f"ts_{direction}_energy",
            {
                "separation": traction.get("energy_separations", []),
                "energy": traction.get("energies", []),
            },
        )

    try:
        _write_structures(write_dir, lattice_parameter)
    except Exception as exc:
        warn(f"Structure export failed: {exc}", stacklevel=2)

    safe_results = _json_safe(results)
    (write_dir / "results.json").write_text(
        json.dumps(safe_results, indent=2, allow_nan=False)
    )
    summary = {
        "benchmark_version": BENCHMARK_VERSION,
        "a0": eos.get("a0"),
        "B0": eos.get("B0"),
        "C11": elastic.get("C11"),
        "C12": elastic.get("C12"),
        "C44": elastic.get("C44"),
        "E_bcc_fcc_meV": bain.get("delta_E_meV"),
        "E_vac": vacancy.get("E_vac"),
        **{f"gamma_{name}": surfaces.get(f"gamma_{name}") for name in SURFACE_CONFIG},
        **{
            f"max_sfe_{name}": results[f"sfe_{name}"].get("max_sfe")
            for name in SFE_CONFIG
        },
        **{
            f"max_traction_{name}": results[f"ts_{name}"].get("max_traction")
            for name in TS_CONFIG
        },
    }
    (write_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False)
    )
    print(f"[{model_name}] Done. Results saved to {write_dir}")


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_iron_properties(model_name: str) -> None:
    """
    Run the Iron Properties benchmark for one registered model.

    Parameters
    ----------
    model_name
        Registered model name.
    """
    run_iron_properties(model_name, MODELS[model_name])
