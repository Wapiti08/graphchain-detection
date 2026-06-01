from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from parsers.events import Event


@dataclass(frozen=True)
class GenerateGraphResult:
    stem: str
    out_dir: Path
    graph_pt: Path
    graph_full_pt: Path
    tgn_pt: Optional[Path]
    stats: Dict[str, Any]


def _load_events(
    *,
    repo_root: Path,
    dataset: str,
    scenario: str,
    qut_kind: str,
    package_name: str,
    only_ioc_logs: bool,
    limit_per_file: Optional[int],
) -> tuple[List[Event], str]:
    if dataset == "synthchain":
        from parsers.synthchain import load_synthchain_events

        events = load_synthchain_events(
            scenario,
            project_root=repo_root,
            only_ioc_logs=bool(only_ioc_logs),
            limit_per_file=limit_per_file,
        )
        return events, f"synthchain_{scenario}"

    import pandas as pd

    from config.qut_sources import QUT_SOURCES
    from parsers.qut.processed import (
        parse_filetop_row,
        parse_install_row,
        parse_opensnoop_row,
        parse_pattern_row,
        parse_syscall_row,
        parse_tcp_row,
    )

    row_parsers = {
        "install_traces": parse_install_row,
        "syscall_traces": parse_syscall_row,
        "opensnoop_traces": parse_opensnoop_row,
        "filetop_traces": parse_filetop_row,
        "tcp_traces": parse_tcp_row,
        "pattern_traces": parse_pattern_row,
    }

    if qut_kind == "all":
        if not package_name:
            raise ValueError("--package-name is required when qut_kind=all")
        from parsers.qut.join import load_qut_processed_dfs, parse_qut_joined_package

        dfs = load_qut_processed_dfs(repo_root, limit_per_file=limit_per_file)
        events = parse_qut_joined_package(package_name, dfs=dfs)
        return events, f"qut_joined_{package_name}"

    if qut_kind not in row_parsers:
        raise ValueError(f"Unknown qut_kind={qut_kind!r}; expected one of {sorted(row_parsers)} or 'all'")

    spec = QUT_SOURCES[qut_kind]
    path = repo_root / spec.rel_path
    df = pd.read_csv(path)
    if limit_per_file is not None:
        df = df.head(limit_per_file)

    row_parser = row_parsers[qut_kind]

    events: List[Event] = []
    for _, row in df.iterrows():
        events.extend(row_parser(row))
    return events, f"qut_{qut_kind}"


def _save_graph_artifacts(
    *,
    events: List[Event],
    stem: str,
    out_dir: Path,
    causal: str,
    causal_window: float,
    export_tgn: bool,
    verbose: bool,
) -> GenerateGraphResult:
    import torch

    from graphcore import build_hetero_graph, hetero_to_tgn_event_stream

    if causal != "off":
        from graphcore.augment import augment_events_with_causal

        events = augment_events_with_causal(events, mode=causal, window=float(causal_window))

    data, stats = build_hetero_graph(events)
    stats_dict = {
        "num_events": stats.num_events,
        "num_nodes_by_type": {k.value: v for k, v in stats.num_nodes_by_type.items()},
        "num_edges_by_type": {k.value: v for k, v in stats.num_edges_by_type.items()},
    }

    graph_pt = out_dir / f"{stem}.pt"
    graph_full_pt = out_dir / f"{stem}.full.pt"
    torch.save({"data_dict": data.to_dict(), "stats": stats_dict}, graph_pt)
    torch.save({"data": data, "stats": stats_dict}, graph_full_pt)

    tgn_pt: Optional[Path] = None
    if export_tgn:
        stream = hetero_to_tgn_event_stream(data, cat_hash_buckets=8, include_meta=False)
        tgn_pt = out_dir / f"{stem}.tgn.pt"
        payload: Dict[str, Any] = {
            "src": stream.src,
            "dst": stream.dst,
            "t": stream.t,
            "msg": stream.msg,
            "etype": stream.etype,
            "y_ioc": stream.y_ioc,
            "y_ioc_line": stream.y_ioc_line,
        }
        if stream.row_idx is not None:
            payload["row_idx"] = stream.row_idx
        if stream.source_file is not None:
            payload["source_file"] = list(stream.source_file)
        if stream.ioc_type is not None:
            payload["ioc_type"] = list(stream.ioc_type)
        torch.save(payload, tgn_pt)

    if verbose:
        print(f"Saved: {graph_pt}")
        print(f"Saved: {graph_full_pt}")
        print(f"Events: {stats.num_events}")
        if tgn_pt is not None:
            print(f"Saved: {tgn_pt}")

    return GenerateGraphResult(
        stem=stem,
        out_dir=out_dir,
        graph_pt=graph_pt,
        graph_full_pt=graph_full_pt,
        tgn_pt=tgn_pt,
        stats=stats_dict,
    )


def generate_graph(
    *,
    repo_root: Path,
    dataset: str,
    out: str | Path = "artifacts/graphs",
    scenario: str = "sc1",
    qut_kind: str = "syscall_traces",
    package_name: str = "",
    limit_per_file: Optional[int] = None,
    only_ioc_logs: bool = False,
    name: str = "",
    causal: str = "off",
    causal_window: float = 50.0,
    export_tgn: bool = False,
    verbose: bool = True,
) -> GenerateGraphResult:
    """
    Parse dataset events, build a heterogeneous graph, and write artifacts under ``out``.

    Returns paths to ``{stem}.pt``, ``{stem}.full.pt``, and optionally ``{stem}.tgn.pt``.
    """
    try:
        import torch
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing torch. Use the python env where torch+pyg are installed.") from e

    repo_root = repo_root.resolve()
    out_dir = (repo_root / out).resolve() if not Path(out).is_absolute() else Path(out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    events, default_stem = _load_events(
        repo_root=repo_root,
        dataset=dataset,
        scenario=scenario,
        qut_kind=qut_kind,
        package_name=package_name,
        only_ioc_logs=only_ioc_logs,
        limit_per_file=limit_per_file,
    )
    stem = str(name).strip() or default_stem
    return _save_graph_artifacts(
        events=events,
        stem=stem,
        out_dir=out_dir,
        causal=causal,
        causal_window=causal_window,
        export_tgn=export_tgn,
        verbose=verbose,
    )


def generate_qut_all_packages(
    *,
    repo_root: Path,
    out: str | Path = "artifacts/graphs",
    limit_per_file: Optional[int] = None,
    causal: str = "off",
    causal_window: float = 50.0,
    export_tgn: bool = True,
    skip_existing: bool = False,
    max_packages: Optional[int] = None,
    verbose: bool = True,
) -> List[GenerateGraphResult]:
    """Batch-export ``qut_joined_<pkg>`` artifacts for every package (six tables loaded once)."""
    from parsers.qut.join import list_qut_package_names, load_qut_processed_dfs, parse_qut_joined_package

    repo_root = repo_root.resolve()
    out_dir = (repo_root / out).resolve() if not Path(out).is_absolute() else Path(out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    packages = list_qut_package_names(repo_root, limit_per_file=limit_per_file)
    if max_packages is not None:
        packages = packages[: int(max_packages)]

    if verbose:
        print(f"QUT batch: {len(packages)} packages -> {out_dir}")

    dfs = load_qut_processed_dfs(repo_root, limit_per_file=limit_per_file)
    results: List[GenerateGraphResult] = []
    skipped = 0

    for i, pkg in enumerate(packages, start=1):
        stem = f"qut_joined_{pkg}"
        tgn_path = out_dir / f"{stem}.tgn.pt"
        if skip_existing and export_tgn and tgn_path.is_file():
            skipped += 1
            continue

        if verbose and (i == 1 or i % 100 == 0 or i == len(packages)):
            print(f"[{i}/{len(packages)}] {pkg}")

        events = parse_qut_joined_package(pkg, dfs=dfs)
        res = _save_graph_artifacts(
            events=events,
            stem=stem,
            out_dir=out_dir,
            causal=causal,
            causal_window=causal_window,
            export_tgn=export_tgn,
            verbose=False,
        )
        results.append(res)

    if verbose:
        print(f"QUT batch done: wrote {len(results)}, skipped {skipped} (existing .tgn.pt)")
    return results


def generate_synthchain_all_scenarios(
    *,
    repo_root: Path,
    scenarios: Optional[List[str]] = None,
    out: str | Path = "artifacts/graphs",
    only_ioc_logs: bool = False,
    limit_per_file: Optional[int] = None,
    causal: str = "off",
    causal_window: float = 50.0,
    export_tgn: bool = True,
    skip_existing: bool = False,
    verbose: bool = True,
) -> List[GenerateGraphResult]:
    """批量导出 SynthChain sc1..sc7（或自定义 scenario 列表）的 .tgn.pt。"""
    if scenarios is None:
        scenarios = [f"sc{i}" for i in range(1, 8)]

    repo_root = repo_root.resolve()
    out_dir = (repo_root / out).resolve() if not Path(out).is_absolute() else Path(out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"SynthChain batch: {len(scenarios)} scenarios -> {out_dir}")

    results: List[GenerateGraphResult] = []
    for sc in scenarios:
        stem = f"synthchain_{sc}"
        tgn_path = out_dir / f"{stem}.tgn.pt"
        if skip_existing and export_tgn and tgn_path.is_file():
            continue
        events, _ = _load_events(
            repo_root=repo_root,
            dataset="synthchain",
            scenario=sc,
            qut_kind="all",
            package_name="",
            only_ioc_logs=only_ioc_logs,
            limit_per_file=limit_per_file,
        )
        results.append(
            _save_graph_artifacts(
                events=events,
                stem=stem,
                out_dir=out_dir,
                causal=causal,
                causal_window=causal_window,
                export_tgn=export_tgn,
                verbose=verbose,
            )
        )
    if verbose:
        print(f"SynthChain batch done: wrote {len(results)}")
    return results
