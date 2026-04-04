#!/usr/bin/env python3
"""Run config-driven AMP batch simulations with LHS sampling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_amp_sample import (
    DEFAULT_CORNER_NAME,
    DEFAULT_TARGET_SPECS,
    execute_amp_run,
    find_repo_root,
    load_design_parameters,
    load_json,
    parse_spice_number,
    repo_or_absolute,
    write_json,
)


DEFAULT_BATCH_CONFIG = "configs/amp_lhs_batch.json"
DEFAULT_TOPOLOGY_REGISTRY = "configs/amp_topology_registry.json"


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(
        description="Run a config-driven AMP batch using LHS design sampling."
    )
    parser.add_argument(
        "--config",
        default=str(repo_root / DEFAULT_BATCH_CONFIG),
        help="Path to the batch JSON config.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["serial", "parallel"],
        default=None,
        help="Optional override for execution.mode in the batch config.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional override for execution.max_workers in the batch config.",
    )
    return parser.parse_args()


def merge_dicts(*parts: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        if part:
            merged.update(part)
    return merged


def write_rows_to_csv(path: Path, rows: list[OrderedDict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return

    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def classify_design_variable(name: str) -> str | None:
    if "_L_" in name:
        return "MOSFET_L"
    if "_W_" in name:
        return "MOSFET_W"
    if "_M_" in name:
        return "MOSFET_M"
    if "CAPACITOR" in name:
        return "CAPACITOR"
    if "CURRENT" in name:
        return "CURRENT"
    if "RESISTOR" in name:
        return "RESISTOR"
    return None


def resolve_bound_spec(
    *,
    topology_name: str,
    variable_name: str,
    variable_kind: str,
    batch_config: dict[str, Any],
) -> dict[str, Any]:
    topology_overrides = (
        batch_config.get("topology_variable_overrides", {}).get(topology_name, {})
    )
    if variable_name in topology_overrides:
        return dict(topology_overrides[variable_name])

    presets = batch_config.get("design_bound_presets", {})
    if variable_kind not in presets:
        raise KeyError(
            f"No LHS bound preset configured for {variable_name} ({variable_kind})."
        )
    return dict(presets[variable_kind])


def get_sampled_design_variables(
    *,
    topology_name: str,
    netlist_path: Path,
    design_parameters: OrderedDict[str, str],
    batch_config: dict[str, Any],
) -> list[tuple[str, str, dict[str, Any]]]:
    config = batch_config.get("design_sampling", {})
    include_only_referenced = bool(config.get("include_only_referenced_in_netlist", True))
    exclude_names = set(config.get("exclude", []))
    include_names = set(config.get("include", []))

    netlist_text = netlist_path.read_text(encoding="utf-8")
    sampled: list[tuple[str, str, dict[str, Any]]] = []

    for variable_name in design_parameters.keys():
        if include_names and variable_name not in include_names:
            continue
        if variable_name in exclude_names:
            continue

        variable_kind = classify_design_variable(variable_name)
        if variable_kind is None:
            continue

        if include_only_referenced and variable_name not in netlist_text:
            continue

        sampled.append(
            (
                variable_name,
                variable_kind,
                resolve_bound_spec(
                    topology_name=topology_name,
                    variable_name=variable_name,
                    variable_kind=variable_kind,
                    batch_config=batch_config,
                ),
            )
        )
    if not sampled:
        raise ValueError(f"No LHS variables selected for topology {topology_name}")
    return sampled


def latin_hypercube_unit(
    sample_count: int,
    dimensions: int,
    seed: int,
) -> list[list[float]]:
    rng = random.Random(seed)
    matrix = [[0.0 for _ in range(dimensions)] for _ in range(sample_count)]
    for dim in range(dimensions):
        intervals = [(index + rng.random()) / sample_count for index in range(sample_count)]
        rng.shuffle(intervals)
        for row_idx, value in enumerate(intervals):
            matrix[row_idx][dim] = value
    return matrix


def scale_lhs_value(unit_value: float, spec: dict[str, Any]) -> float:
    raw_min = parse_spice_number(spec["min"])
    raw_max = parse_spice_number(spec["max"])
    if raw_max < raw_min:
        raise ValueError(f"Invalid bound range: {spec}")

    scale = spec.get("scale", "linear")
    if scale == "log":
        if raw_min <= 0 or raw_max <= 0:
            raise ValueError(f"Log-scale bounds must be positive: {spec}")
        value = math.exp(math.log(raw_min) + unit_value * (math.log(raw_max) - math.log(raw_min)))
    else:
        value = raw_min + unit_value * (raw_max - raw_min)

    dtype = spec.get("dtype", "float")
    if dtype == "int":
        value = int(round(value))
        value = max(int(math.ceil(raw_min)), min(int(math.floor(raw_max)), value))
        return float(value)
    return value


def build_design_samples(
    *,
    topology_name: str,
    topology_spec: dict[str, Any],
    batch_config: dict[str, Any],
    sample_count: int,
    seed: int,
    repo_root: Path,
) -> tuple[list[OrderedDict[str, float]], list[dict[str, Any]]]:
    netlist_path = repo_or_absolute(repo_root, topology_spec["netlist_path"])
    design_variables_path = repo_or_absolute(repo_root, topology_spec["design_variables_path"])
    design_parameters = load_design_parameters(design_variables_path)
    selected_variables = get_sampled_design_variables(
        topology_name=topology_name,
        netlist_path=netlist_path,
        design_parameters=design_parameters,
        batch_config=batch_config,
    )

    unit_samples = latin_hypercube_unit(sample_count, len(selected_variables), seed)
    overrides: list[OrderedDict[str, float]] = []
    bound_records: list[dict[str, Any]] = []

    for variable_name, variable_kind, spec in selected_variables:
        bound_records.append(
            {
                "variable_name": variable_name,
                "variable_kind": variable_kind,
                "bound_spec": spec,
            }
        )

    for row in unit_samples:
        sample_override: OrderedDict[str, float] = OrderedDict()
        for dim_idx, (variable_name, _, spec) in enumerate(selected_variables):
            sample_override[variable_name] = scale_lhs_value(row[dim_idx], spec)
        overrides.append(sample_override)
    return overrides, bound_records


def resolve_environment_scenarios(batch_config: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = list(batch_config.get("environment_scenarios", []))
    if scenarios:
        return scenarios
    return [
        {
            "name": "nominal_tt",
            "corner_name": batch_config.get("corner_name", DEFAULT_CORNER_NAME),
            "acdc_testbench_overrides": batch_config.get("acdc_testbench_overrides", {}),
            "tran_testbench_overrides": batch_config.get("tran_testbench_overrides", {}),
        }
    ]


def build_batch_plan(
    *,
    repo_root: Path,
    batch_config: dict[str, Any],
    registry: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    topology_names = batch_config.get("topologies", [])
    if not topology_names:
        raise ValueError("Batch config must list at least one topology.")

    samples_per_topology = int(batch_config.get("sampling", {}).get("samples_per_topology", 0))
    if samples_per_topology <= 0:
        raise ValueError("sampling.samples_per_topology must be positive.")

    base_seed = int(batch_config.get("sampling", {}).get("seed", 1234))
    scenarios = resolve_environment_scenarios(batch_config)
    batch_root = repo_or_absolute(repo_root, batch_config["batch_root"])
    results_root = batch_root / "results"

    plan: list[dict[str, Any]] = []
    plan_metadata: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_config["batch_id"],
        "samples_per_topology": samples_per_topology,
        "scenario_names": [scenario["name"] for scenario in scenarios],
        "topologies": {},
    }

    registry_topologies = registry.get("topologies", {})
    for topology_index, topology_name in enumerate(topology_names):
        if topology_name not in registry_topologies:
            raise KeyError(f"Topology {topology_name} is missing from the registry.")

        topology_spec = registry_topologies[topology_name]
        topology_seed = base_seed + topology_index
        design_samples, bound_records = build_design_samples(
            topology_name=topology_name,
            topology_spec=topology_spec,
            batch_config=batch_config,
            sample_count=samples_per_topology,
            seed=topology_seed,
            repo_root=repo_root,
        )

        plan_metadata["topologies"][topology_name] = {
            "seed": topology_seed,
            "sampled_variable_bounds": bound_records,
            "sample_count": samples_per_topology,
        }

        topology_overrides = batch_config.get("topology_run_overrides", {}).get(topology_name, {})
        topology_target_specs = merge_dicts(
            batch_config.get("target_specs", {}),
            topology_spec.get("target_specs", {}),
            topology_overrides.get("target_specs", {}),
        )

        default_acdc = merge_dicts(
            batch_config.get("acdc_testbench_overrides", {}),
            topology_spec.get("acdc_testbench_overrides", {}),
            topology_overrides.get("acdc_testbench_overrides", {}),
        )
        default_tran = merge_dicts(
            batch_config.get("tran_testbench_overrides", {}),
            topology_spec.get("tran_testbench_overrides", {}),
            topology_overrides.get("tran_testbench_overrides", {}),
        )

        for lhs_index, design_override in enumerate(design_samples):
            lhs_point_id = f"{topology_name}_lhs_{lhs_index:04d}"
            for scenario in scenarios:
                scenario_name = scenario["name"]
                corner_name = scenario.get("corner_name", DEFAULT_CORNER_NAME)
                run_root = batch_root / "runs" / topology_name / scenario_name
                output_csv = results_root / "all_samples.csv"

                plan.append(
                    {
                        "batch_id": batch_config["batch_id"],
                        "topology_name": topology_name,
                        "lhs_point_id": lhs_point_id,
                        "scenario_name": scenario_name,
                        "corner_name": corner_name,
                        "netlist_path": repo_or_absolute(repo_root, topology_spec["netlist_path"]).as_posix(),
                        "design_variables_path": repo_or_absolute(
                            repo_root, topology_spec["design_variables_path"]
                        ).as_posix(),
                        "acdc_template_path": repo_or_absolute(
                            repo_root,
                            topology_spec.get(
                                "acdc_testbench_template",
                                registry.get("defaults", {}).get("acdc_testbench_template"),
                            ),
                        ).as_posix(),
                        "tran_template_path": repo_or_absolute(
                            repo_root,
                            topology_spec.get(
                                "tran_testbench_template",
                                registry.get("defaults", {}).get("tran_testbench_template"),
                            ),
                        ).as_posix(),
                        "pdk_root": repo_or_absolute(
                            repo_root,
                            topology_spec.get("pdk_root", registry.get("defaults", {}).get("pdk_root")),
                        ).as_posix(),
                        "pdk_zip": repo_or_absolute(
                            repo_root,
                            topology_spec.get("pdk_zip", registry.get("defaults", {}).get("pdk_zip")),
                        ).as_posix(),
                        "output_csv": output_csv.as_posix(),
                        "run_root": run_root.as_posix(),
                        "parameter_overrides": dict(design_override),
                        "acdc_testbench_overrides": merge_dicts(
                            default_acdc, scenario.get("acdc_testbench_overrides", {})
                        ),
                        "tran_testbench_overrides": merge_dicts(
                            default_tran, scenario.get("tran_testbench_overrides", {})
                        ),
                        "target_specs": topology_target_specs,
                    }
                )

    return plan, plan_metadata


def run_plan_item(plan_item: dict[str, Any], config_path: Path, repo_root: Path) -> dict[str, Any]:
    row, status = execute_amp_run(
        repo_root=repo_root,
        config_path=config_path,
        netlist_path=Path(plan_item["netlist_path"]),
        design_variables_path=Path(plan_item["design_variables_path"]),
        acdc_template_path=Path(plan_item["acdc_template_path"]),
        tran_template_path=Path(plan_item["tran_template_path"]),
        pdk_root=Path(plan_item["pdk_root"]),
        pdk_zip=Path(plan_item["pdk_zip"]),
        output_csv=Path(plan_item["output_csv"]),
        run_root=Path(plan_item["run_root"]),
        topology_name=plan_item["topology_name"],
        parameter_overrides=plan_item["parameter_overrides"],
        acdc_testbench_overrides=plan_item["acdc_testbench_overrides"],
        tran_testbench_overrides=plan_item["tran_testbench_overrides"],
        target_specs=plan_item["target_specs"],
        corner_name=plan_item["corner_name"],
        write_csv=False,
        extra_row_fields={
            "batch_id": plan_item["batch_id"],
            "lhs_point_id": plan_item["lhs_point_id"],
            "scenario_name": plan_item["scenario_name"],
            "corner_name": plan_item["corner_name"],
        },
    )
    return {"row": row, "status": status}


def execute_batch_plan(
    *,
    plan: list[dict[str, Any]],
    execution_mode: str,
    max_workers: int,
    config_path: Path,
    repo_root: Path,
) -> tuple[list[OrderedDict[str, Any]], list[dict[str, Any]]]:
    rows: list[OrderedDict[str, Any]] = []
    statuses: list[dict[str, Any]] = []

    if execution_mode == "serial":
        for plan_item in plan:
            result = run_plan_item(plan_item, config_path, repo_root)
            rows.append(result["row"])
            statuses.append(result["status"])
        return rows, statuses

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(run_plan_item, plan_item, config_path, repo_root): plan_item
            for plan_item in plan
        }
        for future in as_completed(future_to_item):
            result = future.result()
            rows.append(result["row"])
            statuses.append(result["status"])
    rows.sort(key=lambda row: (row.get("topology_name", ""), row.get("lhs_point_id", ""), row.get("scenario_name", "")))
    statuses.sort(key=lambda status: status.get("sample_id", ""))
    return rows, statuses


def write_batch_outputs(
    *,
    batch_root: Path,
    plan: list[dict[str, Any]],
    plan_metadata: dict[str, Any],
    rows: list[OrderedDict[str, Any]],
    statuses: list[dict[str, Any]],
) -> None:
    results_root = batch_root / "results"
    write_json(batch_root / "plan.json", {"metadata": plan_metadata, "plan": plan})
    write_json(batch_root / "status.json", {"runs": statuses})

    all_samples_csv = results_root / "all_samples.csv"
    write_rows_to_csv(all_samples_csv, rows)

    by_topology_root = results_root / "by_topology"
    grouped: dict[str, list[OrderedDict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["topology_name"]), []).append(row)
    for topology_name, topology_rows in grouped.items():
        write_rows_to_csv(by_topology_root / f"{topology_name}.csv", topology_rows)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    batch_config = load_json(config_path)
    registry_path = repo_or_absolute(
        repo_root, batch_config.get("topology_registry_path", DEFAULT_TOPOLOGY_REGISTRY)
    )
    registry = load_json(registry_path)

    execution = dict(batch_config.get("execution", {}))
    execution_mode = args.execution_mode or execution.get("mode", "serial")
    max_workers = int(args.max_workers or execution.get("max_workers", 4))

    plan, plan_metadata = build_batch_plan(
        repo_root=repo_root,
        batch_config=batch_config,
        registry=registry,
    )
    batch_root = repo_or_absolute(repo_root, batch_config["batch_root"])
    batch_root.mkdir(parents=True, exist_ok=True)
    write_json(batch_root / "batch_config_snapshot.json", batch_config)

    rows, statuses = execute_batch_plan(
        plan=plan,
        execution_mode=execution_mode,
        max_workers=max_workers,
        config_path=config_path,
        repo_root=repo_root,
    )
    write_batch_outputs(
        batch_root=batch_root,
        plan=plan,
        plan_metadata=plan_metadata,
        rows=rows,
        statuses=statuses,
    )

    success_count = sum(1 for row in rows if row.get("success"))
    failure_count = len(rows) - success_count
    print(
        json.dumps(
            {
                "batch_id": batch_config["batch_id"],
                "execution_mode": execution_mode,
                "max_workers": max_workers,
                "batch_root": batch_root.as_posix(),
                "total_runs": len(rows),
                "successful_runs": success_count,
                "failed_runs": failure_count,
                "results_csv": (batch_root / "results" / "all_samples.csv").as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
