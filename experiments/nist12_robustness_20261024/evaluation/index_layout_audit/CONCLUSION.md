# Retain the current production tree

Eight alternate cKDTree layouts were tested: leaf sizes 32, 64, 128 and 256,
each with median-balanced and midpoint trees. Every query used one worker.
Candidates requested two neighbors and fell back to the unchanged original
one-neighbor tree for equal/near minima. Timing includes that guard and fallback.

All returned distances and original-data indices match bitwise on all 131,072
cached queries in three alternating timing passes. Four synthetic duplicate,
exact/near-tie, random and float32/strided cases also pass for every layout.
Native CAD point signatures remain unchanged. Peak process RSS is 95.1 MB.

| Layout | Median seconds | Speed versus original |
| --- | ---: | ---: |
| Original leaf16, balanced, k1 | 0.3984 | 1.00× |
| Leaf32, balanced, k2 + fallback | 0.3539 | 1.13× |
| Leaf64, balanced, k2 + fallback | 0.3301 | 1.21× |
| Leaf128, balanced, k2 + fallback | 0.3331 | 1.20× |
| Leaf256, balanced, k2 + fallback | 0.3765 | 1.06× |

The midpoint variants range from 0.97× to 1.06×. Construction costs are
0.9–1.9 ms per alternate tree and are recorded separately. Other verified
workers and both required dense pilots continued running; two ordinary workers
were paused for the audit and automatically resumed after 32.045 seconds.

The best measured gain is a modest 20.7% query-speed improvement in this bounded
one-thread test. Four-thread performance and whole-fit speed were not measured.
The conservative near-distance guard is not a derived universal floating-point
traversal certificate: empirical parity alone does not prove that every future
query preserves the original index choice. Therefore this bounded investigation
ends without installing any alternate tree or changing numeric cache keys.

Evidence: `results.json` and `probe.py`. No pose fit, annotation read, model load,
installation, production source edit or previous fixed-receipt change occurred.
