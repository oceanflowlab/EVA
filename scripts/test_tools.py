#!/usr/bin/env python3
"""Compatibility entry point for the self-contained five-tool smoke test."""
from pathlib import Path
import runpy


if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tool_service_tests/test_five_tools.py'),
                   run_name='__main__')
