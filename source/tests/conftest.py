"""Pytest session-wide setup.

Sets environment variables that must be present BEFORE torch and xgboost are
first imported in this process, to avoid a known OpenMP runtime conflict
between the two libraries (each bundles its own libomp/libgomp): running
both in the same process without this can deadlock a torch forward pass
(reproduced in this project when test_train_container.py's real DLRM
forward-pass test and test_export_xgboost_genesis.py's real xgb.train()
call are collected in the same pytest session -- confirmed by isolating the
two files together with strace-free bisection, and confirmed fixed by these
exact two env vars).

KMP_DUPLICATE_LIB_OK=TRUE: allows two OpenMP runtimes to coexist in one
process (the actual upstream-documented workaround for this conflict).
OMP_NUM_THREADS=1: avoids each library's own internal thread pool
contending for CPU cores, which is the usual secondary cause once the
duplicate-runtime crash/hang itself is no longer fatal.

Must be set via os.environ (not pytest.ini/[tool.pytest.ini_options] env)
before the first `import torch` or `import xgboost` anywhere in the
collected test suite -- conftest.py is imported before test modules, so
setting it here (module level, at import time) is early enough.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
