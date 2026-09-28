#!/usr/bin/env python
"""
Wayfinder Service Shutdown Utility

Terminates running Prefect server and Watcher processes on Windows/Linux/macOS.
Frees up port 4200 and halts all document filesystem watching.

Usage:
    python stop_services.py
"""

import os
import sys
import subprocess
import signal
from typing import List, Set


def get_pids_on_port(port: int = 4200) -> Set[int]:
    """Finds process IDs listening on the specified port."""
    pids: Set[int] = set()
    if sys.platform == "win32":
        try:
            output = subprocess.check_output(
                f"netstat -ano | findstr :{port}",
                shell=True,
                text=True,
                stderr=subprocess.DEVNULL
            )
            for line in output.strip().split("\n"):
                parts = line.strip().split()
                if len(parts) >= 5 and "LISTENING" in parts:
                    pid = int(parts[-1])
                    if pid > 0:
                        pids.add(pid)
        except subprocess.CalledProcessError:
            pass
    else:
        try:
            output = subprocess.check_output(
                f"lsof -ti:{port}",
                shell=True,
                text=True,
                stderr=subprocess.DEVNULL
            )
            for line in output.strip().split("\n"):
                if line.strip():
                    pids.add(int(line.strip()))
        except subprocess.CalledProcessError:
            pass
    return pids


def get_matching_process_pids(keywords: List[str]) -> Set[int]:
    """Finds process IDs whose command line contains any of the specified keywords."""
    pids: Set[int] = set()
    current_pid = os.getpid()

    if sys.platform == "win32":
        for kw in keywords:
            cmd = f'powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like \'*{kw}*\' }} | Select-Object -ExpandProperty ProcessId"'
            try:
                output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
                for line in output.strip().split("\n"):
                    line = line.strip()
                    if line.isdigit():
                        pid = int(line)
                        if pid != current_pid and pid > 0:
                            pids.add(pid)
            except Exception:
                pass
    else:
        for kw in keywords:
            try:
                output = subprocess.check_output(f"pgrep -f '{kw}'", shell=True, text=True, stderr=subprocess.DEVNULL)
                for line in output.strip().split("\n"):
                    line = line.strip()
                    if line.isdigit():
                        pid = int(line)
                        if pid != current_pid and pid > 0:
                            pids.add(pid)
            except subprocess.CalledProcessError:
                pass

    return pids


def kill_pid(pid: int, description: str = ""):
    """Terminates process by PID, killing entire process tree on Windows."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                f"taskkill /F /PID {pid} /T",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            os.kill(pid, signal.SIGKILL)
        print(f"  [✓] Stopped {description} (PID: {pid})")
    except Exception as e:
        print(f"  [!] Failed to stop PID {pid}: {e}")


def main():
    print("=" * 60)
    print(" Wayfinder Ingestion - Service Shutdown Utility")
    print("=" * 60)

    # 1. Stop Document Watcher processes
    print("\n[1/2] Checking for running Document Watcher processes...")
    watcher_keywords = ["watch_documents.py", "wayfinder_ingestion.watcher", "pipeline.py --watch"]
    watcher_pids = get_matching_process_pids(watcher_keywords)

    if watcher_pids:
        print(f"Found {len(watcher_pids)} active watcher process(es):")
        for pid in watcher_pids:
            kill_pid(pid, "Document Watcher")
    else:
        print("  [i] No active Document Watcher processes found.")

    # 2. Stop Prefect Server processes
    print("\n[2/2] Checking for running Prefect Server processes...")
    prefect_pids = get_pids_on_port(4200)
    cmdline_prefect_pids = get_matching_process_pids(["prefect server start", "prefect.server"])
    all_prefect_pids = prefect_pids.union(cmdline_prefect_pids)

    if all_prefect_pids:
        print(f"Found {len(all_prefect_pids)} Prefect server process(es):")
        for pid in all_prefect_pids:
            kill_pid(pid, "Prefect Server")
    else:
        print("  [i] No active Prefect Server processes found on port 4200.")

    print("\n" + "=" * 60)
    print(" Shutdown complete! Both Watcher and Prefect Server are stopped.")
    print("=" * 60)


if __name__ == "__main__":
    main()
