"""Import trainer dependencies once in the single-threaded forkserver.

The trainer is imported as a module, never executed here. Spawned collectors
share its read-only import pages instead of importing PyTorch independently.
"""
import importlib.util
import os
from pathlib import Path
import sys

path = Path(os.environ['NSGYM_FROZEN_RELEASE'])/'scripts/train.py'
spec = importlib.util.spec_from_file_location('nsgym_preloaded_trainer', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
