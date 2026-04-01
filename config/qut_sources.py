"""
QUT-DV25 processed CSV sources and column mappings.

These processed datasets are *package-level aggregates* (no per-event timestamp).
Parsers should produce canonical Events with `order` as pseudo-time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class QUTSourceSpec:
    name: str
    rel_path: str
    level_col: Optional[str] = "Level"
    package_col: str = "Package_Name"
    # Column mapping is intentionally left flexible; parser modules interpret it.
    columns: Dict[str, str] = None  # type: ignore[assignment]


QUT_SOURCES: Dict[str, QUTSourceSpec] = {
    "syscall_traces": QUTSourceSpec(
        name="syscall_traces",
        rel_path="data/QUT-DV25_Datasets/QUT-DV25_Processed_Datasets/"
        "QUT-DV25_SysCall_Traces/QUT-DV25_SysCall_Traces.csv",
        columns={
            "unique_syscall_list": "Unique_System_Calls_List",
            "total_syscalls": "Total_System_Calls",
            "unique_syscalls": "Unique_System_Calls",
        },
    ),
    "opensnoop_traces": QUTSourceSpec(
        name="opensnoop_traces",
        rel_path="data/QUT-DV25_Datasets/QUT-DV25_Processed_Datasets/"
        "QUT-DV25_Opensnoop_Traces/QUT-DV25_Opensnoop_Traces.csv",
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
    "filetop_traces": QUTSourceSpec(
        name="filetop_traces",
        rel_path="data/QUT-DV25_Datasets/QUT-DV25_Processed_Datasets/"
        "QUT-DV25_Filetop_Traces/QUT-DV25_Filetop_Traces.csv",
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
}

