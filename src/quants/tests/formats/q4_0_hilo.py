"""Host-expanded Q4_0 variant with precomputed bf16 hi/lo scales.

The canonical source format and dequantized reference are identical to Q4_0;
only the packed record layout differs.
"""

from formats.q4_0 import *  # noqa: F401,F403
