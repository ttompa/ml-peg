"""Analyse the Iron Properties benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.integrate import trapezoid

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import load_metrics_config, write_struct_info
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "iron_properties" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "iron_properties"
REFERENCE_PATH = Path(__file__).with_name("reference_data")
METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

with (REFERENCE_PATH / "reference_values.json").open() as reference_file:
    REFERENCE_METADATA = json.load(reference_file)
DFT_REFERENCE = {
    name: entry["value"] for name, entry in REFERENCE_METADATA["values"].items()
}

GROUPS = (
    "Bulk response",
    "Defect & phase energetics",
    "Slip",
    "Cleavage",
)
SCALAR_BAD_ERROR = {
    "a0": 0.02,
    "B0": 0.50,
    "C11": 0.75,
    "C12": 0.75,
    "C44": 0.75,
    "E_vac": 0.20,
}
SURFACE_BAD_ERROR = 0.20
CURVE_BAD_ERROR = 0.30
BAIN_ENDPOINT_BAD_ERROR = 0.50
GSFE_LOCATION_BAD = 0.10
TS_LOCATION_BAD = 0.30

CURVE_FILES = {
    "eos": "eos_curve.csv",
    "bain": "bain_path.csv",
    "sfe_110": "sfe_110_curve.csv",
    "sfe_112": "sfe_112_curve.csv",
    "ts_100": "ts_100_curve.csv",
    "ts_110": "ts_110_curve.csv",
}
CURVE_CONFIG = {
    "eos": {
        "x": "volume",
        "y": "energy_meV",
        "title": "Equation of State",
        "x_label": "Volume (Å³/atom)",
        "y_label": "Relative energy (meV/atom)",
    },
    "bain": {
        "x": "ca_ratio",
        "y": "energy_meV",
        "title": "Bain Path",
        "x_label": "c/a ratio",
        "y_label": "Energy relative to BCC (meV/atom)",
    },
    "sfe_110": {
        "x": "displacement_fraction",
        "y": "sfe_J_per_m2",
        "title": "GSFE {110}<111>",
        "x_label": "Displacement (u/b)",
        "y_label": "Stacking-fault energy (J/m²)",
    },
    "sfe_112": {
        "x": "displacement_fraction",
        "y": "sfe_J_per_m2",
        "title": "GSFE {112}<111>",
        "x_label": "Displacement (u/b)",
        "y_label": "Stacking-fault energy (J/m²)",
    },
    "ts_100": {
        "x": "separation",
        "y": "traction",
        "title": "Traction-Separation (100)",
        "x_label": "Separation (Å)",
        "y_label": "Traction (GPa)",
    },
    "ts_110": {
        "x": "separation",
        "y": "traction",
        "title": "Traction-Separation (110)",
        "x_label": "Separation (Å)",
        "y_label": "Traction (GPa)",
    },
}


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


def _penalty(error: float, bad_error: float) -> float:
    """
    Map a non-negative error to a clipped penalty.

    Parameters
    ----------
    error
        Non-negative error.
    bad_error
        Error mapped to a unit penalty.

    Returns
    -------
    float
        Penalty between zero and one.
    """
    if not _finite(error):
        return 1.0
    return float(np.clip(error / bad_error, 0.0, 1.0))


def _rms_score(components: list[tuple[float, float]]) -> float:
    """
    Combine high-is-better component scores through RMS penalties.

    Parameters
    ----------
    components
        Score and weight pairs.

    Returns
    -------
    float
        Combined score between zero and one.
    """
    weight_sum = sum(weight for _, weight in components)
    if weight_sum <= 0:
        return 0.0
    squared_penalty = sum(
        weight * (1 - float(np.clip(score, 0.0, 1.0))) ** 2
        for score, weight in components
    )
    return 1 - float(np.sqrt(squared_penalty / weight_sum))


def _status_summary(status: Any) -> tuple[str, bool]:
    """
    Summarize nested convergence records and detect optimizer errors.

    Parameters
    ----------
    status
        Nested calculation diagnostics.

    Returns
    -------
    tuple[str, bool]
        Display summary and whether the result is usable.
    """
    converged: list[bool] = []
    valid: list[bool] = []
    errors: list[str] = []

    def visit(value: Any) -> None:
        """
        Collect status fields recursively.

        Parameters
        ----------
        value
            Nested status value.
        """
        if isinstance(value, dict):
            if isinstance(value.get("converged"), bool):
                converged.append(value["converged"])
            if isinstance(value.get("valid"), bool):
                valid.append(value["valid"])
            if value.get("error"):
                errors.append(str(value["error"]))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(status)
    if errors:
        return f"error: {errors[0]}", False
    if valid and not all(valid):
        return f"{sum(valid)}/{len(valid)} finite points", False
    if converged:
        return f"{sum(converged)}/{len(converged)} relaxations converged", True
    return "complete", True


def load_model_results(model_name: str) -> dict[str, Any] | None:
    """
    Load one model's result file.

    Parameters
    ----------
    model_name
        Registered model name.

    Returns
    -------
    dict[str, Any] or None
        Results when available.
    """
    result_path = CALC_PATH / model_name / "results.json"
    if not result_path.exists():
        return None
    return json.loads(result_path.read_text())


def load_curve(model_name: str, curve_type: str) -> pd.DataFrame:
    """
    Load one generated curve, returning an empty frame if unavailable.

    Parameters
    ----------
    model_name
        Registered model name.
    curve_type
        Curve identifier.

    Returns
    -------
    pandas.DataFrame
        Generated curve or an empty frame.
    """
    filename = CURVE_FILES.get(curve_type)
    if filename is None:
        return pd.DataFrame()
    path = CALC_PATH / model_name / filename
    if not path.exists():
        return pd.DataFrame()

    curve = pd.read_csv(path)
    if curve_type == "eos" and "energy_meV" not in curve and "energy" in curve:
        energy = pd.to_numeric(curve["energy"], errors="coerce")
        if energy.notna().any():
            curve["energy_meV"] = (energy - energy.min()) * 1000
    if (
        curve_type.startswith("sfe_")
        and "displacement_fraction" not in curve
        and "displacement" in curve
    ):
        displacement = pd.to_numeric(curve["displacement"], errors="coerce")
        endpoint = displacement.max()
        if _finite(endpoint) and endpoint > 0:
            curve["displacement_fraction"] = displacement / endpoint
    return curve


def load_reference_curve(curve_type: str) -> pd.DataFrame:
    """
    Load and normalize one bundled DFT reference curve.

    Parameters
    ----------
    curve_type
        Curve identifier.

    Returns
    -------
    pandas.DataFrame
        Normalized reference curve.
    """
    if curve_type == "eos":
        curve = pd.read_csv(
            REFERENCE_PATH / "eos.csv",
            sep=";",
            decimal=",",
            header=None,
            names=["volume", "energy_meV"],
            skipinitialspace=True,
        )
        curve["energy_meV"] -= curve["energy_meV"].min()
        return curve.sort_values("volume")
    if curve_type == "bain":
        curve = pd.read_csv(REFERENCE_PATH / "bain.csv")
        bcc_index = (curve["ca_ratio"] - 1).abs().idxmin()
        curve["energy_meV"] -= curve.loc[bcc_index, "energy_meV"]
        return curve
    if curve_type.startswith("sfe_"):
        curve = pd.read_csv(REFERENCE_PATH / f"{curve_type}.csv")
        endpoints = pd.DataFrame(
            {"displacement_fraction": [0.0, 1.0], "sfe_J_per_m2": [0.0, 0.0]}
        )
        return pd.concat([endpoints.iloc[:1], curve, endpoints.iloc[1:]])
    if curve_type.startswith("ts_"):
        curve = pd.read_csv(REFERENCE_PATH / f"{curve_type}.csv")
        return curve.rename(columns={"traction_GPa": "traction"})
    raise KeyError(f"Unknown Iron reference curve: {curve_type}")


def _clean_curve(
    frame: pd.DataFrame, x_column: str, y_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract sorted, finite, unique curve coordinates.

    Parameters
    ----------
    frame
        Source curve.
    x_column
        Horizontal coordinate column.
    y_column
        Vertical coordinate column.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Sorted coordinate arrays.
    """
    if frame.empty or not {x_column, y_column}.issubset(frame.columns):
        return np.array([]), np.array([])
    values = frame[[x_column, y_column]].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    values = values.groupby(x_column, as_index=False)[y_column].mean()
    values = values.sort_values(x_column)
    return values[x_column].to_numpy(), values[y_column].to_numpy()


def _integrated_curve_penalty(
    model_x: np.ndarray,
    model_y: np.ndarray,
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    domain: tuple[float, float],
    valid_fraction: float = 1.0,
) -> tuple[float, float, float]:
    """
    Return coverage-adjusted IAE penalty, relative IAE, and coverage.

    Parameters
    ----------
    model_x
        Model horizontal coordinates.
    model_y
        Model vertical coordinates.
    reference_x
        Reference horizontal coordinates.
    reference_y
        Reference vertical coordinates.
    domain
        Scoring-domain bounds.
    valid_fraction
        Fraction of sampled model points with finite values.

    Returns
    -------
    tuple[float, float, float]
        IAE penalty, relative IAE, and fractional coverage.
    """
    if len(model_x) < 3:
        return 1.0, float("nan"), 0.0
    domain_start, domain_end = domain
    overlap_start = max(domain_start, float(model_x.min()))
    overlap_end = min(domain_end, float(model_x.max()))
    coverage = (
        max(0.0, overlap_end - overlap_start)
        / (domain_end - domain_start)
        * float(np.clip(valid_fraction, 0.0, 1.0))
    )
    if coverage <= 0:
        return 1.0, float("nan"), 0.0

    grid = np.linspace(overlap_start, overlap_end, 501)
    model_values = np.interp(grid, model_x, model_y)
    reference_values = np.interp(grid, reference_x, reference_y)
    denominator = float(trapezoid(np.abs(reference_values), grid))
    relative_iae = (
        float(trapezoid(np.abs(model_values - reference_values), grid)) / denominator
        if denominator > 0
        else float("nan")
    )
    penalty = coverage * _penalty(relative_iae, CURVE_BAD_ERROR) + (1 - coverage)
    return penalty, relative_iae, coverage


def _curve_score(
    model: pd.DataFrame,
    reference: pd.DataFrame,
    x_column: str,
    y_column: str,
    domain: tuple[float, float],
    location_bad: float,
) -> dict[str, float]:
    """
    Score a curve using integrated, peak, and peak-location errors.

    Parameters
    ----------
    model
        Model curve.
    reference
        DFT reference curve.
    x_column
        Horizontal coordinate column.
    y_column
        Vertical coordinate column.
    domain
        Scoring-domain bounds.
    location_bad
        Peak-location error mapped to a unit penalty.

    Returns
    -------
    dict[str, float]
        Composite score and component errors.
    """
    model_x, model_y = _clean_curve(model, x_column, y_column)
    reference_x, reference_y = _clean_curve(reference, x_column, y_column)
    if model.empty or not {x_column, y_column}.issubset(model.columns):
        valid_fraction = 0.0
    else:
        model_values = model[[x_column, y_column]].apply(pd.to_numeric, errors="coerce")
        in_domain = model_values[x_column].between(*domain)
        domain_values = model_values.loc[in_domain, y_column]
        valid_fraction = (
            float(domain_values.notna().mean()) if not domain_values.empty else 0.0
        )
    iae_penalty, relative_iae, coverage = _integrated_curve_penalty(
        model_x,
        model_y,
        reference_x,
        reference_y,
        domain,
        valid_fraction,
    )
    reference_mask = (reference_x >= domain[0]) & (reference_x <= domain[1])
    reference_domain_x = reference_x[reference_mask]
    reference_domain_y = reference_y[reference_mask]
    reference_peak_index = int(np.argmax(reference_domain_y))
    reference_peak = float(reference_domain_y[reference_peak_index])
    reference_location = float(reference_domain_x[reference_peak_index])

    model_mask = (model_x >= domain[0]) & (model_x <= domain[1])
    model_domain_x = model_x[model_mask]
    model_domain_y = model_y[model_mask]
    peak_bracketed = (
        len(model_domain_x) >= 3
        and model_domain_x.min() <= reference_location <= model_domain_x.max()
        and np.any(model_domain_y > 0)
    )
    if peak_bracketed:
        model_peak_index = int(np.argmax(model_domain_y))
        model_peak = float(model_domain_y[model_peak_index])
        model_location = float(model_domain_x[model_peak_index])
        peak_error = abs(model_peak - reference_peak) / abs(reference_peak)
        location_error = abs(model_location - reference_location)
        peak_penalty = _penalty(peak_error, CURVE_BAD_ERROR)
        location_penalty = _penalty(location_error, location_bad)
    else:
        model_peak = float("nan")
        model_location = float("nan")
        peak_error = float("nan")
        location_error = float("nan")
        peak_penalty = 1.0
        location_penalty = 1.0

    score = 1 - float(
        np.sqrt(
            0.50 * iae_penalty**2 + 0.30 * peak_penalty**2 + 0.20 * location_penalty**2
        )
    )
    return {
        "score": score,
        "relative_iae": relative_iae,
        "coverage": coverage,
        "model_peak": model_peak,
        "reference_peak": reference_peak,
        "peak_error": peak_error,
        "model_location": model_location,
        "reference_location": reference_location,
        "location_error": location_error,
    }


def _scalar_detail(
    name: str,
    predicted: Any,
    reference: float,
    bad_error: float,
    unit: str,
    status: Any,
) -> dict[str, Any]:
    """
    Build one scalar component score and its display metadata.

    Parameters
    ----------
    name
        Component label.
    predicted
        Predicted scalar value.
    reference
        Reference scalar value.
    bad_error
        Relative error mapped to a unit penalty.
    unit
        Display unit.
    status
        Calculation diagnostics.

    Returns
    -------
    dict[str, Any]
        Component score and display metadata.
    """
    status_text, status_valid = _status_summary(status)
    valid = _finite(predicted) and status_valid
    error = (
        abs(float(predicted) - reference) / abs(reference) if valid else float("nan")
    )
    score = 1 - _penalty(error, bad_error) if valid else 0.0
    return {
        "name": name,
        "score": score,
        "predicted": float(predicted) if _finite(predicted) else None,
        "reference": reference,
        "error": error,
        "threshold": bad_error,
        "unit": unit,
        "status": status_text,
    }


def _curve_detail(
    name: str, score: dict[str, float], unit: str, status: Any
) -> dict[str, Any]:
    """
    Build display metadata for a composite curve score.

    Parameters
    ----------
    name
        Component label.
    score
        Curve score and errors.
    unit
        Peak-value display unit.
    status
        Calculation diagnostics.

    Returns
    -------
    dict[str, Any]
        Component display metadata.
    """
    status_text, _ = _status_summary(status)
    return {
        "name": name,
        "score": score["score"],
        "predicted": score["model_peak"],
        "reference": score["reference_peak"],
        "error": score["relative_iae"],
        "threshold": CURVE_BAD_ERROR,
        "unit": unit,
        "status": status_text,
        "peak_error": score["peak_error"],
        "location_error": score["location_error"],
        "coverage": score["coverage"],
    }


def _surface_detail(results: dict[str, Any]) -> dict[str, Any]:
    """
    Score the RMS error across the four surface orientations.

    Parameters
    ----------
    results
        Surface calculation results.

    Returns
    -------
    dict[str, Any]
        Composite surface score and display metadata.
    """
    components = []
    predictions = []
    references = []
    for orientation in ("100", "110", "111", "112"):
        detail = _scalar_detail(
            orientation,
            results.get(f"gamma_{orientation}"),
            DFT_REFERENCE[f"gamma_{orientation}"],
            SURFACE_BAD_ERROR,
            "J/m²",
            results.get("status", {}).get(orientation),
        )
        components.append((detail["score"], 0.25))
        predictions.append(detail["predicted"])
        references.append(detail["reference"])
    status_text, _ = _status_summary(results.get("status"))
    return {
        "name": "Surface energies",
        "score": _rms_score(components),
        "predicted": predictions,
        "reference": references,
        "error": None,
        "threshold": SURFACE_BAD_ERROR,
        "unit": "J/m²",
        "status": status_text,
    }


def _bain_detail(model_name: str, results: dict[str, Any]) -> dict[str, Any]:
    """
    Score Bain curve shape and the exact FCC endpoint.

    Parameters
    ----------
    model_name
        Registered model name.
    results
        Bain calculation results.

    Returns
    -------
    dict[str, Any]
        Composite Bain score and display metadata.
    """
    model = load_curve(model_name, "bain")
    reference = load_reference_curve("bain")
    model_x, model_y = _clean_curve(model, "ca_ratio", "energy_meV")
    reference_x, reference_y = _clean_curve(reference, "ca_ratio", "energy_meV")
    if model.empty or not {"ca_ratio", "energy_meV"}.issubset(model.columns):
        valid_fraction = 0.0
    else:
        model_values = model[["ca_ratio", "energy_meV"]].apply(
            pd.to_numeric, errors="coerce"
        )
        in_domain = model_values["ca_ratio"].between(0.72, 2.0)
        domain_values = model_values.loc[in_domain, "energy_meV"]
        valid_fraction = (
            float(domain_values.notna().mean()) if not domain_values.empty else 0.0
        )
    curve_penalty, relative_iae, coverage = _integrated_curve_penalty(
        model_x,
        model_y,
        reference_x,
        reference_y,
        (0.72, 2.0),
        valid_fraction,
    )
    endpoint = results.get("delta_E_meV")
    endpoint_reference = DFT_REFERENCE["E_bcc_fcc"]
    endpoint_error = (
        abs(float(endpoint) - endpoint_reference) / endpoint_reference
        if _finite(endpoint)
        else float("nan")
    )
    endpoint_penalty = _penalty(endpoint_error, BAIN_ENDPOINT_BAD_ERROR)
    score = 1 - float(np.sqrt(0.70 * curve_penalty**2 + 0.30 * endpoint_penalty**2))
    status_text, _ = _status_summary(results.get("status"))
    return {
        "name": "Bain path",
        "score": score,
        "predicted": float(endpoint) if _finite(endpoint) else None,
        "reference": endpoint_reference,
        "error": relative_iae,
        "threshold": CURVE_BAD_ERROR,
        "unit": "meV/atom",
        "status": status_text,
        "endpoint_error": endpoint_error,
        "coverage": coverage,
    }


def compute_group_scores(
    model_name: str, results: dict[str, Any]
) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]]]:
    """
    Compute four group scores and their drill-down details.

    Parameters
    ----------
    model_name
        Registered model name.
    results
        Calculation results.

    Returns
    -------
    tuple[dict[str, float], dict[str, list[dict[str, Any]]]]
        Group scores and component metadata.
    """
    eos = results.get("eos", {})
    elastic = results.get("elastic", {})
    vacancy = results.get("vacancy", {})
    bain = results.get("bain_path", {})
    surfaces = results.get("surfaces", {})

    bulk_details = [
        _scalar_detail(
            "Lattice parameter",
            eos.get("a0"),
            DFT_REFERENCE["a0"],
            SCALAR_BAD_ERROR["a0"],
            "Å",
            eos.get("status"),
        ),
        _scalar_detail(
            "Bulk modulus",
            eos.get("B0"),
            DFT_REFERENCE["B0"],
            SCALAR_BAD_ERROR["B0"],
            "GPa",
            eos.get("status"),
        ),
    ]
    for constant in ("C11", "C12", "C44"):
        bulk_details.append(
            _scalar_detail(
                constant,
                elastic.get(constant),
                DFT_REFERENCE[constant],
                SCALAR_BAD_ERROR[constant],
                "GPa",
                elastic.get("status"),
            )
        )
    bulk_score = _rms_score(
        [
            (bulk_details[0]["score"], 0.25),
            (bulk_details[1]["score"], 0.25),
            *((detail["score"], 1 / 6) for detail in bulk_details[2:]),
        ]
    )

    vacancy_detail = _scalar_detail(
        "Vacancy formation",
        vacancy.get("E_vac"),
        DFT_REFERENCE["E_vac"],
        SCALAR_BAD_ERROR["E_vac"],
        "eV",
        vacancy.get("status"),
    )
    bain_detail = _bain_detail(model_name, bain)
    defect_details = [vacancy_detail, bain_detail]
    defect_score = _rms_score(
        [(vacancy_detail["score"], 0.5), (bain_detail["score"], 0.5)]
    )

    slip_details = []
    for orientation in ("110", "112"):
        curve_score = _curve_score(
            load_curve(model_name, f"sfe_{orientation}"),
            load_reference_curve(f"sfe_{orientation}"),
            "displacement_fraction",
            "sfe_J_per_m2",
            (0.0, 1.0),
            GSFE_LOCATION_BAD,
        )
        slip_details.append(
            _curve_detail(
                f"GSFE {{{orientation}}}<111>",
                curve_score,
                "J/m²",
                results.get(f"sfe_{orientation}", {}).get("status"),
            )
        )
    slip_score = _rms_score(
        [(slip_details[0]["score"], 0.5), (slip_details[1]["score"], 0.5)]
    )

    surface_detail = _surface_detail(surfaces)
    cleavage_details = [surface_detail]
    for orientation in ("100", "110"):
        curve_score = _curve_score(
            load_curve(model_name, f"ts_{orientation}"),
            load_reference_curve(f"ts_{orientation}"),
            "separation",
            "traction",
            (0.0, 4.0),
            TS_LOCATION_BAD,
        )
        cleavage_details.append(
            _curve_detail(
                f"Traction-separation ({orientation})",
                curve_score,
                "GPa",
                results.get(f"ts_{orientation}", {}).get("status"),
            )
        )
    cleavage_score = _rms_score(
        [
            (surface_detail["score"], 0.5),
            (cleavage_details[1]["score"], 0.25),
            (cleavage_details[2]["score"], 0.25),
        ]
    )

    scores = {
        "Bulk response": bulk_score,
        "Defect & phase energetics": defect_score,
        "Slip": slip_score,
        "Cleavage": cleavage_score,
    }
    details = {
        "Bulk response": bulk_details,
        "Defect & phase energetics": defect_details,
        "Slip": slip_details,
        "Cleavage": cleavage_details,
        "Score": [
            {
                "name": group,
                "score": score,
                "predicted": None,
                "reference": 1.0,
                "error": 1 - score,
                "threshold": 1.0,
                "unit": "score",
                "status": "complete",
            }
            for group, score in scores.items()
        ],
    }
    return scores, details


def _format_value(value: Any) -> str:
    """
    Format a scalar or sequence for Plotly hover text.

    Parameters
    ----------
    value
        Value to format.

    Returns
    -------
    str
        Compact display string.
    """
    if value is None:
        return "n/a"
    if isinstance(value, list):
        return ", ".join(_format_value(item) for item in value)
    if _finite(value):
        return f"{float(value):.5g}"
    return "n/a"


def create_breakdown_figure(
    model_name: str, group: str, details: list[dict[str, Any]]
) -> go.Figure:
    """
    Create a horizontal component-score drill-down plot.

    Parameters
    ----------
    model_name
        Registered model name.
    group
        Group display name.
    details
        Component metadata.

    Returns
    -------
    plotly.graph_objects.Figure
        Score-breakdown figure.
    """
    names = [detail["name"] for detail in details]
    custom_data = [
        [
            _format_value(detail.get("predicted")),
            _format_value(detail.get("reference")),
            _format_value(detail.get("error")),
            _format_value(detail.get("threshold")),
            detail.get("unit", ""),
            detail.get("status", ""),
        ]
        for detail in details
    ]
    scores = [detail["score"] for detail in details]
    hovertemplate = (
        "Score: %{x:.3f}<br>Predicted: %{customdata[0]} %{customdata[4]}"
        "<br>Reference: %{customdata[1]} %{customdata[4]}"
        "<br>Error: %{customdata[2]}<br>Bad threshold: %{customdata[3]}"
        "<br>Status: %{customdata[5]}<extra></extra>"
    )
    figure = go.Figure(
        go.Bar(
            x=scores,
            y=names,
            orientation="h",
            customdata=custom_data,
            hovertemplate=hovertemplate,
        )
    )
    zero_indices = [index for index, score in enumerate(scores) if score == 0]
    if zero_indices:
        figure.add_trace(
            go.Scatter(
                x=[0] * len(zero_indices),
                y=[names[index] for index in zero_indices],
                mode="markers",
                marker={"size": 10},
                customdata=[custom_data[index] for index in zero_indices],
                hovertemplate=hovertemplate,
                showlegend=False,
            )
        )
    figure.update_layout(
        title=f"{group} breakdown — {model_name}",
        xaxis={"title": "Component score", "range": [0, 1]},
        yaxis={"autorange": "reversed"},
        template="plotly_white",
        height=max(340, 80 + 55 * len(details)),
    )
    return figure


def create_curve_figure(
    frame: pd.DataFrame, curve_type: str, model_name: str
) -> go.Figure:
    """
    Plot a model curve together with its bundled DFT reference.

    Parameters
    ----------
    frame
        Model curve.
    curve_type
        Curve identifier.
    model_name
        Registered model name.

    Returns
    -------
    plotly.graph_objects.Figure
        Comparison figure.
    """
    config = CURVE_CONFIG[curve_type]
    reference = load_reference_curve(curve_type)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=reference[config["x"]],
            y=reference[config["y"]],
            mode="lines+markers",
            name="DFT reference",
            line={"width": 2, "dash": "dash", "color": "gray"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame[config["x"]],
            y=frame[config["y"]],
            mode="lines+markers",
            name=model_name,
            line={"width": 2},
            marker={"size": 5},
        )
    )
    if curve_type == "bain":
        figure.add_vline(x=1.0, line_dash="dot", annotation_text="BCC")
        figure.add_vline(x=np.sqrt(2), line_dash="dot", annotation_text="FCC")
    if curve_type.startswith(("sfe_", "ts_")):
        for trace_name, data, color in (
            ("DFT peak", reference, "gray"),
            ("Model peak", frame, "red"),
        ):
            x, y = _clean_curve(data, config["x"], config["y"])
            if len(y) and np.any(y > 0):
                peak = int(np.argmax(y))
                figure.add_trace(
                    go.Scatter(
                        x=[x[peak]],
                        y=[y[peak]],
                        mode="markers",
                        name=trace_name,
                        marker={"size": 10, "color": color, "symbol": "x"},
                    )
                )
    figure.update_layout(
        title=f"{config['title']} — {model_name}",
        xaxis_title=config["x_label"],
        yaxis_title=config["y_label"],
        template="plotly_white",
        height=500,
    )
    return figure


def save_figures_for_model(
    model_name: str, details: dict[str, list[dict[str, Any]]]
) -> None:
    """
    Save curve and score-breakdown figures for one model.

    Parameters
    ----------
    model_name
        Registered model name.
    details
        Group and overall-score component metadata.
    """
    figure_dir = OUT_PATH / "figures" / model_name
    figure_dir.mkdir(parents=True, exist_ok=True)
    for curve_type in CURVE_FILES:
        frame = load_curve(model_name, curve_type)
        if not frame.empty:
            create_curve_figure(frame, curve_type, model_name).write_json(
                figure_dir / f"{curve_type}.json"
            )
    for group, group_details in details.items():
        filename = group.lower().replace(" ", "_").replace("&", "and")
        create_breakdown_figure(model_name, group, group_details).write_json(
            figure_dir / f"breakdown_{filename}.json"
        )


def collect_metrics() -> pd.DataFrame:
    """
    Collect the four Iron group scores across available models.

    Returns
    -------
    pandas.DataFrame
        One row of group scores per model.
    """
    rows: list[dict[str, str | float]] = []
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    for model_name in MODELS:
        results = load_model_results(model_name)
        if results is None:
            continue
        scores, details = compute_group_scores(model_name, results)
        rows.append({"Model": model_name, **scores})
        save_figures_for_model(model_name, details)
    return pd.DataFrame(rows).reindex(columns=["Model", *GROUPS])


@pytest.fixture
def iron_properties_collection() -> pd.DataFrame:
    """
    Provide collected Iron group scores.

    Returns
    -------
    pandas.DataFrame
        One row of group scores per model.
    """
    return collect_metrics()


@pytest.fixture
@build_table(
    filename=OUT_PATH / "iron_properties_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(iron_properties_collection: pd.DataFrame) -> dict[str, dict]:
    """
    Build the Iron dashboard table.

    Parameters
    ----------
    iron_properties_collection
        Collected group scores.

    Returns
    -------
    dict[str, dict]
        Group scores keyed by model.
    """
    return {
        column: dict(
            zip(
                iron_properties_collection["Model"],
                iron_properties_collection[column],
                strict=True,
            )
        )
        for column in GROUPS
    }


def test_iron_properties(metrics: dict[str, dict]) -> None:
    """
    Generate Iron analysis artifacts.

    Parameters
    ----------
    metrics
        Generated group metrics.
    """
    structure_path = CALC_PATH / "mock" / "structures" / "equilibrium_bcc.extxyz"
    if structure_path.exists():
        write_struct_info(data_path=structure_path, out_path=OUT_PATH, index=0)
