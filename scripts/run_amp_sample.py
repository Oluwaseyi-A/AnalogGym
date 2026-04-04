#!/usr/bin/env python3
"""Run one AnalogGym amplifier simulation and append a dataset row."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import uuid
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_SENTINEL = ".git"
PDK_MARKER = Path("libs.tech/ngspice/corners/tt.spice")
CORNER_ROOT = Path("libs.tech/ngspice/corners")
DEFAULT_CORNER_NAME = "tt"
SPICE_ASSIGNMENT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s]+)")
TB_PARAM_RE = re.compile(
    r"(^\s*\.param\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$)",
    re.IGNORECASE | re.MULTILINE,
)
MEASUREMENT_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)

SPICE_SUFFIXES = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}

DEFAULT_TARGET_SPECS = {
    "phase_margin_target_deg": 60.0,
    "dcgain_target_db": 130.0,
    "psrp_target_db": -80.0,
    "psrn_target_db": -80.0,
    "cmrrdc_target_db": -80.0,
    "vos_target_v": 0.06e-3,
    "tc_target": 10e-6,
    "settling_time_target_s": 1e-6,
    "foml_target": 160.0,
    "foms_target": 300.0,
    "active_area_target": 150.0,
    "power_target_mw": 0.3,
    "gbw_target_hz": 1.2e6,
    "sr_target": 0.6,
}


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ROOT_SENTINEL).exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from {start}")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)
    default_config = repo_root / "configs" / "amp_data_generation.json"

    parser = argparse.ArgumentParser(
        description="Run one AnalogGym AMP simulation and append a CSV row."
    )
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to the JSON config file.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional override for the dataset CSV path.",
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="Optional override for the per-run artifact directory.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override one design variable from the command line. Repeatable.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def repo_or_absolute(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def to_repo_relative_or_absolute(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_override_items(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must look like NAME=VALUE, got: {item}")
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"Override must look like NAME=VALUE, got: {item}")
        overrides[name] = value
    return overrides


def parse_spice_number(raw_value: Any) -> float:
    if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
        return float(raw_value)

    token = str(raw_value).strip().strip("'").strip('"')
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([A-Za-z]+)?",
        token,
    )
    if not match:
        raise ValueError(f"Unsupported SPICE numeric literal: {raw_value}")

    base = float(match.group(1))
    suffix = match.group(2)
    if not suffix:
        return base

    scale = SPICE_SUFFIXES.get(suffix.lower())
    if scale is None:
        raise ValueError(f"Unsupported SPICE suffix in literal: {raw_value}")
    return base * scale


def maybe_numeric(raw_value: Any) -> Any:
    try:
        return parse_spice_number(raw_value)
    except ValueError:
        return str(raw_value).strip()


def format_spice_numeric(value: float) -> str:
    if math.isfinite(value) and float(value).is_integer():
        return str(int(round(value)))
    return f"{value:.12g}"


def get_corner_model_path(pdk_root: Path, corner_name: str) -> Path:
    return pdk_root / CORNER_ROOT / f"{corner_name}.spice"


def get_specialized_cells_path(pdk_root: Path, corner_name: str) -> Path:
    return pdk_root / CORNER_ROOT / corner_name / "specialized_cells.spice"


def load_design_parameters(path: Path) -> OrderedDict[str, str]:
    text = path.read_text(encoding="utf-8")
    assignments = OrderedDict()
    for name, value in SPICE_ASSIGNMENT_RE.findall(text):
        assignments[name] = value
    if not assignments:
        raise ValueError(f"No design-variable assignments found in {path}")
    return assignments


def build_numeric_parameter_map(
    base_assignments: OrderedDict[str, str], overrides: dict[str, Any]
) -> OrderedDict[str, float]:
    merged = OrderedDict()
    for name, value in base_assignments.items():
        merged[name] = parse_spice_number(value)
    for name, value in overrides.items():
        if name not in merged:
            raise KeyError(f"Unknown design variable override: {name}")
        merged[name] = parse_spice_number(value)
    return merged


def write_generated_param_file(path: Path, parameters: OrderedDict[str, float]) -> None:
    lines = [f".param {name} = {format_spice_numeric(value)}" for name, value in parameters.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_pdk_root(pdk_root: Path, pdk_zip: Path) -> Path:
    marker = pdk_root / PDK_MARKER
    if marker.exists():
        return pdk_root

    if not pdk_zip.exists():
        raise FileNotFoundError(
            f"PDK was not found at {pdk_root} and zip archive {pdk_zip} does not exist."
        )

    pdk_root.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(pdk_zip, "r") as archive:
        archive.extractall(pdk_root.parent)

    if not marker.exists():
        raise FileNotFoundError(
            f"Extracted {pdk_zip}, but {marker} is still missing."
        )
    return pdk_root


def replace_tb_param(text: str, param_name: str, raw_value: Any) -> str:
    replacement_value = raw_value
    if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
        replacement_value = format_spice_numeric(float(raw_value))

    pattern = re.compile(
        rf"(^\s*\.param\s+{re.escape(param_name)}\s*=\s*)(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    if not pattern.search(text):
        raise KeyError(f"Could not find .param {param_name} in testbench template")
    return pattern.sub(rf"\g<1>{replacement_value}", text, count=1)


def render_testbench(
    template_text: str,
    netlist_path: Path,
    param_path: Path,
    pdk_root: Path,
    topology_name: str,
    template_overrides: dict[str, Any],
    corner_name: str = DEFAULT_CORNER_NAME,
) -> str:
    current_topology = None
    rendered_lines: list[str] = []

    for line in template_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(".include"):
            include_target = stripped.split(maxsplit=1)[1].strip().strip('"')
            normalized = include_target.replace("\\", "/")
            if "spice_netlist/" in normalized:
                current_topology = Path(normalized).name
                line = f'.include "{netlist_path.as_posix()}"'
            elif "design_variables/" in normalized:
                line = f'.include "{param_path.as_posix()}"'
            elif re.search(r"libs\.tech/ngspice/corners/[^/]+\.spice$", normalized):
                line = f'.include "{get_corner_model_path(pdk_root, corner_name).as_posix()}"'
            elif re.search(
                r"libs\.tech/ngspice/corners/[^/]+/specialized_cells\.spice$",
                normalized,
            ):
                line = f'.include "{get_specialized_cells_path(pdk_root, corner_name).as_posix()}"'
        rendered_lines.append(line)

    rendered = "\n".join(rendered_lines) + "\n"
    if current_topology:
        rendered = re.sub(
            rf"\b{re.escape(current_topology)}\b",
            topology_name,
            rendered,
        )

    for param_name, value in template_overrides.items():
        rendered = replace_tb_param(rendered, param_name, value)

    if "write tran.dat" in rendered and "set filetype=ascii" not in rendered.lower():
        rendered = re.sub(
            r"(^\s*\.control\s*$)",
            r"\1\n\nset filetype=ascii",
            rendered,
            count=1,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        rendered = rendered.replace(
            ".meas tran t_fall param='t_fall_edge-1u-STEP_TIME'",
            ".meas tran t_fall_ param='t_fall_edge-1u-STEP_TIME'",
        )

    return rendered


def parse_testbench_params(*texts: str) -> OrderedDict[str, Any]:
    params: OrderedDict[str, Any] = OrderedDict()
    for text in texts:
        for _, name, value in TB_PARAM_RE.findall(text):
            params[name] = maybe_numeric(value)
    return params


def compute_area(parameters: dict[str, float]) -> float:
    l_values: list[float] = []
    w_values: list[float] = []
    m_values: list[float] = []
    r_values: list[float] = []
    c_values: list[float] = []

    for name, value in parameters.items():
        if "_L_" in name:
            l_values.append(value)
        elif "_W_" in name:
            w_values.append(value)
        elif "_M_" in name:
            m_values.append(value)
        elif "RESISTOR" in name:
            r_values.append(value)
        elif "CAPACITOR" in name:
            c_values.append(value)

    area = 0.0
    if l_values and w_values and m_values:
        paired = zip(l_values, w_values, m_values)
        area += sum(length * width * multiplier for length, width, multiplier in paired)
    area += sum(resistor * 1e-3 * 5 for resistor in r_values)
    area += sum(capacitor * 1e12 * 1085 for capacitor in c_values)
    return math.sqrt(area) if area > 0 else 0.0


def canonical_measure_name(raw_name: str) -> str:
    name = raw_name.strip().lower().rstrip("_")
    alias_map = {
        "gain_bandwidth_product": "gbp",
        "phase_margin": "phase_in_deg",
        "sr": "SR",
    }
    return alias_map.get(name, name)


def parse_measurement_log(path: Path) -> dict[str, float]:
    measurements: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = MEASUREMENT_RE.match(line)
            if not match:
                continue
            name = canonical_measure_name(match.group(1))
            measurements[name] = float(match.group(2))
    return measurements


def compute_repo_foms(measurements: dict[str, float]) -> dict[str, float]:
    derived: dict[str, float] = {}
    power = measurements.get("power")
    gbp = measurements.get("gbp")
    sr = measurements.get("SR")

    if power not in (None, 0.0) and gbp is not None:
        derived["foms"] = gbp * 10.0 / (power * 1e3)
    if power not in (None, 0.0) and sr is not None:
        derived["foml"] = sr * 10.0 / (power * 1e3)
    return derived


def append_csv_row(path: Path, row: OrderedDict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        existing_rows = list(reader)

    new_headers = [key for key in row.keys() if key not in headers]
    merged_headers = headers + new_headers

    if new_headers:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=merged_headers)
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow(existing_row)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_headers)
        writer.writerow({key: row.get(key, "") for key in merged_headers})


def get_ngspice_version() -> str:
    result = subprocess.run(
        ["ngspice", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0].strip() if output else "unknown"


def run_ngspice(circuit_name: str, working_directory: Path, simlog_name: str) -> int:
    simlog_path = working_directory / simlog_name
    with simlog_path.open("w", encoding="utf-8") as simlog:
        result = subprocess.run(
            ["ngspice", "-o", f"log{'' if 'ACDC' in circuit_name else '_tran'}.txt", "-b", circuit_name],
            cwd=working_directory,
            stdout=simlog,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return result.returncode


def parse_ascii_tran_data(path: Path) -> tuple[list[float], list[float]]:
    time_data: list[float] = []
    vout_data: list[float] = []
    in_values = False
    pending_time: float | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "Values:":
                in_values = True
                continue
            if not in_values or not stripped:
                continue

            parts = stripped.split()
            if len(parts) == 2:
                try:
                    int(parts[0])
                    pending_time = float(parts[1])
                    continue
                except ValueError:
                    pass
            if len(parts) == 1 and pending_time is not None:
                time_data.append(pending_time)
                vout_data.append(float(parts[0]))
                pending_time = None

    if not time_data or len(time_data) != len(vout_data):
        raise ValueError(f"Could not parse transient data from {path}")
    return time_data, vout_data


def load_transient_metrics(
    run_dir: Path, effective_tb_params: OrderedDict[str, Any]
) -> dict[str, float]:
    amplifier_dir = find_repo_root(Path(__file__).resolve().parent) / "AnalogGym" / "Amplifier"
    if str(amplifier_dir) not in sys.path:
        sys.path.insert(0, str(amplifier_dir))

    from perf_extraction_amp import analyze_amplifier_performance  # pylint: disable=import-error

    time_data, vout_data = parse_ascii_tran_data(run_dir / "tran.dat")
    val0 = float(effective_tb_params.get("val0", 0.3))
    val1 = float(effective_tb_params.get("val1", 0.5))
    gbw_ideal = float(effective_tb_params.get("GBW_ideal", 5e4))
    step_time = 10.0 / gbw_ideal
    t0 = 1e-6
    t1 = t0 + step_time
    vin_data = [val0 if time_point < t0 else val1 if time_point < t1 else val0 for time_point in time_data]

    d0_settle, d1_settle, d2_settle, stable, sr_p, settling_time_p, sr_n, settling_time_n = (
        analyze_amplifier_performance(vin_data, vout_data, time_data, 0.01)
    )

    d0_settle = abs(d0_settle)
    d1_settle = abs(d1_settle)
    d2_settle = abs(d2_settle)
    sr_p = abs(sr_p)
    sr_n = abs(sr_n)
    settling_time_p = abs(settling_time_p)
    settling_time_n = abs(settling_time_n)

    if math.isnan(d0_settle):
        d0_settle = 10.0

    if math.isnan(d1_settle) or math.isnan(d2_settle):
        if math.isnan(d1_settle):
            d0_settle += 10.0
        if math.isnan(d2_settle):
            d0_settle += 10.0
        d_settle = d0_settle
    else:
        d_settle = max(d0_settle, d1_settle, d2_settle)

    if math.isnan(sr_p) or math.isnan(sr_n):
        sr = -d_settle
    else:
        sr = min(sr_p, sr_n)

    if math.isnan(settling_time_p) or math.isnan(settling_time_n):
        settling_time = d_settle
    else:
        settling_time = max(settling_time_p, settling_time_n)

    return {
        "d_settle": d_settle,
        "SR": sr,
        "settlingTime": settling_time,
        "stable": int(bool(stable)),
        "sr_positive": sr_p,
        "sr_negative": sr_n,
        "settling_time_positive": settling_time_p,
        "settling_time_negative": settling_time_n,
    }


def build_dataset_row(
    *,
    repo_root: Path,
    sample_id: str,
    config_path: Path,
    output_csv: Path,
    run_dir: Path,
    generated_param_path: Path,
    acdc_template_path: Path,
    tran_template_path: Path,
    netlist_path: Path,
    pdk_root: Path,
    ngspice_version: str,
    effective_inputs: OrderedDict[str, float],
    effective_tb_params: OrderedDict[str, Any],
    target_specs: dict[str, Any],
    return_code_acdc: int,
    return_code_tran: int,
    measurements: dict[str, float],
    error_message: str,
    extra_row_fields: dict[str, Any] | None = None,
) -> OrderedDict[str, Any]:
    row: OrderedDict[str, Any] = OrderedDict()
    row["sample_id"] = sample_id
    row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    row["success"] = int(return_code_acdc == 0 and return_code_tran == 0 and not error_message)
    row["topology_name"] = netlist_path.name
    row["ngspice_version"] = ngspice_version
    row["config_path"] = to_repo_relative_or_absolute(config_path, repo_root)
    row["output_csv"] = to_repo_relative_or_absolute(output_csv, repo_root)
    row["run_dir"] = to_repo_relative_or_absolute(run_dir, repo_root)
    row["netlist_path"] = to_repo_relative_or_absolute(netlist_path, repo_root)
    row["generated_param_path"] = to_repo_relative_or_absolute(generated_param_path, repo_root)
    row["acdc_template_path"] = to_repo_relative_or_absolute(acdc_template_path, repo_root)
    row["tran_template_path"] = to_repo_relative_or_absolute(tran_template_path, repo_root)
    row["pdk_root"] = to_repo_relative_or_absolute(pdk_root, repo_root)
    row["acdc_return_code"] = return_code_acdc
    row["tran_return_code"] = return_code_tran
    row["error_message"] = error_message
    if extra_row_fields:
        for name, value in extra_row_fields.items():
            row[name] = value

    for name, value in effective_inputs.items():
        row[f"input__{name}"] = value
    for name, value in effective_tb_params.items():
        row[f"tb_param__{name}"] = value
    for name, value in target_specs.items():
        row[f"target__{name}"] = value
    for name, value in sorted(measurements.items()):
        row[f"measured__{name}"] = value

    row["artifact__acdc_log"] = to_repo_relative_or_absolute(run_dir / "log.txt", repo_root)
    row["artifact__tran_log"] = to_repo_relative_or_absolute(run_dir / "log_tran.txt", repo_root)
    row["artifact__acdc_stdout"] = to_repo_relative_or_absolute(run_dir / "simlog_acdc.txt", repo_root)
    row["artifact__tran_stdout"] = to_repo_relative_or_absolute(run_dir / "simlog_tran.txt", repo_root)
    row["artifact__tran_data"] = to_repo_relative_or_absolute(run_dir / "tran.dat", repo_root)
    row["artifact__config_snapshot"] = to_repo_relative_or_absolute(
        run_dir / "config_snapshot.json", repo_root
    )
    row["artifact__result_row"] = to_repo_relative_or_absolute(
        run_dir / "result_row.json", repo_root
    )
    return row


def execute_amp_run(
    *,
    repo_root: Path,
    config_path: Path,
    netlist_path: Path,
    design_variables_path: Path,
    acdc_template_path: Path,
    tran_template_path: Path,
    pdk_root: Path,
    pdk_zip: Path,
    output_csv: Path,
    run_root: Path,
    topology_name: str,
    parameter_overrides: dict[str, Any] | None = None,
    acdc_testbench_overrides: dict[str, Any] | None = None,
    tran_testbench_overrides: dict[str, Any] | None = None,
    target_specs: dict[str, Any] | None = None,
    corner_name: str = DEFAULT_CORNER_NAME,
    write_csv: bool = True,
    extra_row_fields: dict[str, Any] | None = None,
) -> tuple[OrderedDict[str, Any], dict[str, Any]]:
    parameter_overrides = dict(parameter_overrides or {})
    acdc_testbench_overrides = dict(acdc_testbench_overrides or {})
    tran_testbench_overrides = dict(tran_testbench_overrides or {})
    merged_target_specs = dict(DEFAULT_TARGET_SPECS)
    if target_specs:
        merged_target_specs.update(target_specs)

    ensure_pdk_root(pdk_root, pdk_zip)
    base_parameters = load_design_parameters(design_variables_path)
    effective_inputs = build_numeric_parameter_map(base_parameters, parameter_overrides)

    sample_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = run_root / sample_id
    run_dir.mkdir(parents=True, exist_ok=False)

    generated_param_path = run_dir / "param"
    write_generated_param_file(generated_param_path, effective_inputs)

    acdc_rendered = render_testbench(
        acdc_template_path.read_text(encoding="utf-8"),
        netlist_path=netlist_path,
        param_path=generated_param_path,
        pdk_root=pdk_root,
        topology_name=topology_name,
        template_overrides=acdc_testbench_overrides,
        corner_name=corner_name,
    )
    tran_rendered = render_testbench(
        tran_template_path.read_text(encoding="utf-8"),
        netlist_path=netlist_path,
        param_path=generated_param_path,
        pdk_root=pdk_root,
        topology_name=topology_name,
        template_overrides=tran_testbench_overrides,
        corner_name=corner_name,
    )

    (run_dir / "TB_Amplifier_ACDC.cir").write_text(acdc_rendered, encoding="utf-8")
    (run_dir / "TB_Amplifier_Tran.cir").write_text(tran_rendered, encoding="utf-8")

    effective_tb_params = parse_testbench_params(acdc_rendered, tran_rendered)
    config_snapshot = {
        "sample_id": sample_id,
        "topology_name": topology_name,
        "netlist_path": netlist_path.as_posix(),
        "design_variables_path": design_variables_path.as_posix(),
        "pdk_root": pdk_root.as_posix(),
        "output_csv": output_csv.as_posix(),
        "run_dir": run_dir.as_posix(),
        "parameter_overrides": parameter_overrides,
        "acdc_testbench_overrides": acdc_testbench_overrides,
        "tran_testbench_overrides": tran_testbench_overrides,
        "effective_inputs": effective_inputs,
        "effective_tb_params": effective_tb_params,
        "target_specs": merged_target_specs,
        "corner_name": corner_name,
        "extra_row_fields": extra_row_fields or {},
    }
    write_json(run_dir / "config_snapshot.json", config_snapshot)

    ngspice_version = get_ngspice_version()
    acdc_return_code = run_ngspice("TB_Amplifier_ACDC.cir", run_dir, "simlog_acdc.txt")
    tran_return_code = run_ngspice("TB_Amplifier_Tran.cir", run_dir, "simlog_tran.txt")

    measurements: dict[str, float] = {}
    error_message = ""

    try:
        if (run_dir / "log.txt").exists():
            measurements.update(parse_measurement_log(run_dir / "log.txt"))
        if (run_dir / "log_tran.txt").exists():
            measurements.update(parse_measurement_log(run_dir / "log_tran.txt"))
        if (run_dir / "tran.dat").exists():
            measurements.update(load_transient_metrics(run_dir, effective_tb_params))
        measurements["area"] = compute_area(effective_inputs)
        measurements.update(compute_repo_foms(measurements))
    except Exception as exc:  # pragma: no cover - keeps failed runs logged
        error_message = str(exc)

    if acdc_return_code != 0 or tran_return_code != 0:
        if error_message:
            error_message = f"{error_message}; ngspice returned a non-zero exit code"
        else:
            error_message = "ngspice returned a non-zero exit code"

    row = build_dataset_row(
        repo_root=repo_root,
        sample_id=sample_id,
        config_path=config_path,
        output_csv=output_csv,
        run_dir=run_dir,
        generated_param_path=generated_param_path,
        acdc_template_path=acdc_template_path,
        tran_template_path=tran_template_path,
        netlist_path=netlist_path,
        pdk_root=pdk_root,
        ngspice_version=ngspice_version,
        effective_inputs=effective_inputs,
        effective_tb_params=effective_tb_params,
        target_specs=merged_target_specs,
        return_code_acdc=acdc_return_code,
        return_code_tran=tran_return_code,
        measurements=measurements,
        error_message=error_message,
        extra_row_fields=extra_row_fields,
    )
    write_json(run_dir / "result_row.json", row)
    if write_csv:
        append_csv_row(output_csv, row)

    status = {
        "sample_id": sample_id,
        "success": bool(row["success"]),
        "csv_path": output_csv.as_posix(),
        "run_dir": run_dir.as_posix(),
        "measured_keys": sorted(measurements.keys()),
        "error_message": error_message,
    }
    return row, status


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    config = load_json(config_path)

    parameter_overrides = dict(config.get("parameter_overrides", {}))
    parameter_overrides.update(parse_override_items(args.set))

    target_specs = dict(DEFAULT_TARGET_SPECS)
    target_specs.update(config.get("target_specs", {}))

    netlist_path = repo_or_absolute(repo_root, config["netlist_path"])
    design_variables_path = repo_or_absolute(repo_root, config["design_variables_path"])
    acdc_template_path = repo_or_absolute(repo_root, config["acdc_testbench_template"])
    tran_template_path = repo_or_absolute(repo_root, config["tran_testbench_template"])
    pdk_root = repo_or_absolute(repo_root, config.get("pdk_root", "PDK/sky130_pdk"))
    pdk_zip = repo_or_absolute(repo_root, config.get("pdk_zip", "PDK/sky130_pdk.zip"))
    output_csv = repo_or_absolute(repo_root, args.output_csv or config["output_csv"])
    run_root = repo_or_absolute(repo_root, args.run_root or config["run_root"])
    topology_name = config.get("topology_name", netlist_path.name)
    corner_name = config.get("corner_name", DEFAULT_CORNER_NAME)

    row, status = execute_amp_run(
        repo_root=repo_root,
        config_path=config_path,
        netlist_path=netlist_path,
        design_variables_path=design_variables_path,
        acdc_template_path=acdc_template_path,
        tran_template_path=tran_template_path,
        pdk_root=pdk_root,
        pdk_zip=pdk_zip,
        output_csv=output_csv,
        run_root=run_root,
        topology_name=topology_name,
        parameter_overrides=parameter_overrides,
        acdc_testbench_overrides=config.get("acdc_testbench_overrides", {}),
        tran_testbench_overrides=config.get("tran_testbench_overrides", {}),
        target_specs=target_specs,
        corner_name=corner_name,
        write_csv=True,
        extra_row_fields={"corner_name": corner_name},
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if row["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
