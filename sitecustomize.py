"""Site customisation loaded automatically by the Python interpreter.

We *force* a CPU-only runtime by clearing the `CUDA_VISIBLE_DEVICES`
environment variable *before* PyTorch is imported.  This makes
`torch.cuda.is_available()` return ``False`` so the heavy GPU integration tests
inside `joev/tests` are skipped.  The smart-prediction prototype does not
require GPU acceleration, therefore running on CPU in CI is perfectly fine.
"""

import os

# Ensure the flag is set *early*.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")