"""Optional local-only worker startup; copied as runtime/sitecustomize.py."""
import multiprocessing as mp
import os

method = os.environ.get('NSGYM_LOCAL_START_METHOD')
if method in ('fork', 'forkserver'):
    mp.set_start_method(method, force=True)
    if method == 'forkserver':
        mp.set_forkserver_preload(['nsgym_forkserver_preload'])
