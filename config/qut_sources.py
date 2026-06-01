"""
QUT-DV25 processed CSV sources and column mappings.

The dataset defines six trace families (see QUT-DV25 paper / Harvard Dataverse):
  filetop, install, opensnoop, pattern, syscall, tcp.

Processed tables are package-level aggregates (no per-event timestamp).
Parsers use ``order`` as pseudo-time. Paths follow the Dataverse layout under
``data/QUT-DV25_Datasets/QUT-DV25_Processed_Datasets/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple


@dataclass(frozen=True)
class QUTSourceSpec:
    name: str
    rel_path: str
    level_col: str = "Level"
    package_col: str = "Package_Name"
    label_col: str = "Label"
    columns: Dict[str, str] = field(default_factory=dict)


def _csv(rel_dir: str, stem: str) -> str:
    base = "data/QUT-DV25_Datasets/QUT-DV25_Processed_Datasets"
    return f"{base}/{rel_dir}/{stem}.csv"


# Order matches pseudo-time segmentation in parsers.qut.join (coarse bins).
QUT_SOURCE_KEYS: Tuple[str, ...] = (
    "install_traces",
    "syscall_traces",
    "opensnoop_traces",
    "filetop_traces",
    "tcp_traces",
    "pattern_traces",
)

QUT_SOURCES: Dict[str, QUTSourceSpec] = {
    "filetop_traces": QUTSourceSpec(
        name="filetop_traces",
        rel_path=_csv("QUT-DV25_Filetop_Traces", "QUT-DV25_Filetop_Traces"),
        columns={
            "total_reads": "Total_Reads",
            "total_writes": "Total_Writes",
            "read_bytes": "Total_Read_Data_Transfer",
            "write_bytes": "Total_Write_Data_Transfer",
            "read_processes": "Read_Processes",
            "write_processes": "Write_Processes",
            "file_access_processes": "File_Access_Processes",
        },
    ),
    "install_traces": QUTSourceSpec(
        name="install_traces",
        rel_path=_csv("QUT-DV25_Install_Traces", "QUT-DV25_Install_Traces"),
        columns={
            "total_dependencies": "Total_Dependency",
            "direct_dependencies": "Direct_Dependencies",
            "indirect_dependencies": "Indirect_Dependencies",
        },
    ),
    "opensnoop_traces": QUTSourceSpec(
        name="opensnoop_traces",
        rel_path=_csv("QUT-DV25_Opensnoop_Traces", "QUT-DV25_Opensnoop_Traces"),
        columns={
            "total_paths": "Total_Paths",
            "root_install": "Root_DIR_Installation",
            "tmp_install": "Temporary_DIR_Installation",
            "home_install": "Home_DIR_Installation",
            "etc_install": "Etc_DIR_Installation",
            "other_install": "Other_DIR_Installation",
            "user_access": "User_Access",
            "sys_access": "Sys_Access",
            "python_kw": "Python_Related_Keywords",
            "install_kw": "Install_Package_Keywords",
        },
    ),
    "pattern_traces": QUTSourceSpec(
        name="pattern_traces",
        rel_path=_csv("QUT-DV25_Pattern_Traces", "QUT-DV25_Pattern_Traces"),
        columns={f"pattern_{i}": f"Pattern_{i}" for i in range(1, 11)},
    ),
    "syscall_traces": QUTSourceSpec(
        name="syscall_traces",
        rel_path=_csv("QUT-DV25_SysCall_Traces", "QUT-DV25_SysCall_Traces"),
        columns={
            "unique_syscall_list": "Unique_System_Calls_List",
            "total_syscalls": "Total_System_Calls",
            "unique_syscalls": "Unique_System_Calls",
        },
    ),
    "tcp_traces": QUTSourceSpec(
        name="tcp_traces",
        rel_path=_csv("QUT-DV25_TCP_Traces", "QUT-DV25_TCP_Traces"),
        columns={
            "state_transition": "State_Transition",
            "local_ips": "Local_IPs_Access",
            "remote_ips": "Remote_IPs_Access",
            "local_ports": "Local_Port_Access",
            "remote_ports": "Remote_Port_Access",
        },
    ),
}

# Kinds accepted by CLI / generate_graph (single-table or full join).
QUT_KIND_CHOICES: FrozenSet[str] = frozenset(set(QUT_SOURCE_KEYS) | {"all"})
