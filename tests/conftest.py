"""
Pytest configuration and shared fixtures for Wayfinder.
Ensures src/ is on sys.path for test discovery and execution.
"""

import sys
from pathlib import Path

# Add src directory to sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
