# Pyrandcracker

**Pyrandcracker** recovers the internal state of Python's `random.Random`
(MT19937) from observed outputs, so that future outputs can be predicted.

[English](README.md) | [中文](README-zh.md)

## How it works

The implementation follows the canonical-state model of MT19937:

* the prediction state has **19937 bits** (the low 31 bits of the next raw
  word never influence the output stream);
* a full 32-bit output is an invertible (tempered) state word, so 624
  consecutive words determine the state directly;
* a truncated output (`getrandbits(v)` with `v < 32` returns the top `v`
  bits of one word) is a linear form over the state, so recovery becomes a
  system of linear equations over GF(2).

Three recovery paths are selected automatically:

| input | method | measured cost |
| --- | --- | --- |
| multiples of 32 bits | exact untempering | 624 words = 19968 bits, instant |
| uniform 5-bit values | constructive carrier/extension inverse | 4984 calls, ~3.8 s |
| anything else | bit-packed GF(2) elimination | 4–22 s (see performance) |

## Installation

```bash
pip install pyrandcracker
```

## Usage

At least 19937 observed bits are required in principle (the state dimension);
the solver refuses to guess when the submitted bits do not determine a unique
state, instead of silently returning a wrong one.

### 32-bit outputs

```python
import random
import time
from pyrandcracker import RandCracker

rd = random.Random()
rd.seed(time.time())

rc = RandCracker()
for _ in range(624):
    rc.submit(rd.getrandbits(32))      # 624 words = 19968 bits

rc.check()                             # True; rc.rnd is now positioned after
                                       # the submitted outputs
print(rd.getrandbits(32))
print(rc.rnd.getrandbits(32))          # same value
```

### Truncated outputs (any width)

```python
import random
from pyrandcracker import RandCracker

rd = random.Random()
rd.seed(1)

rc = RandCracker()
for _ in range(624 * 2):
    rc.submit(rd.getrandbits(16), 16)  # 19968 bits of 16-bit values

rc.check()
print(rd.getrandbits(16))
print(rc.rnd.getrandbits(16))
```

Minimum recovering horizons for uniform widths (`getrandbits(v)`):

| v | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calls | 19937 | 9969 | 6849 | 4985 | 4984 | 3738 | 2493 | 1247 | 624 |

These are *minimums*: submitting more than the minimum is always accepted.
The full-word path simply uses the last 624 words, the five-bit path solves
from the first 4984 calls and then verifies every submitted value, and the
generic path puts every submitted bit into the linear system (an
overdetermined but consistent system).  For example 2000 width-16 values
(32000 bits) recover in ~7 s.
### Five-bit outputs (the constructive path)

```python
import random
from pyrandcracker import RandCracker

rd = random.Random()
rd.seed(2)

rc = RandCracker()
for _ in range(4984):
    rc.submit(rd.getrandbits(5), 5)

rc.check()
print(rd.getrandbits(5))
print(rc.rnd.getrandbits(5))
```

### Discontinuous observations

If the observed values are separated by skipped calls, describe the exact
generation pattern with `set_generator_func`. The function replays the
schedule on a generator and returns the observed bits (most significant bit
first within each value):

```python
import random
from pyrandcracker import RandCracker

rd = random.Random()
rd.seed(3)

rc = RandCracker()
for _ in range(624):
    rc.submit(rd.getrandbits(16), 16)
rd.getrandbits(32)                     # one skipped word (not submitted)
for _ in range(1872):
    rc.submit(rd.getrandbits(8), 8)

def getRows(rnd):
    rows = []
    for _ in range(624):
        rows += list(map(int, bin(rnd.getrandbits(16))[2:].zfill(16)))
    rnd.getrandbits(32)                # the skipped call
    for _ in range(1872):
        rows += list(map(int, bin(rnd.getrandbits(8))[2:].zfill(8)))
    return rows

rc.set_generator_func(getRows)
rc.check()
print(rd.getrandbits(16))
print(rc.rnd.getrandbits(16))
```

Not every gap is recoverable: the solver raises `ValueError` when the
observed positions cannot determine the state (see below).

### Moving the recovered generator

```python
number = rc.rnd.getrandbits(32)
rc.offset(-1)                          # rewind one 32-bit word
print(rc.rnd.getrandbits(32) == number)

rc.offset_bits(-150)                   # rewind 150 bits (word-aligned)
```

`check(offset=True)` positions the generator so its next outputs replay the
submitted sequence from the first observation instead of continuing after it.

## Performance

Measured on Python 3.13 (Windows), seed `350234`, with the paper's minimum
recovering horizon for each width (`n_min(v)` calls):

| v | calls | bits | time | v | calls | bits | time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 19937 | 19937 | 6.2 s | 17 | 1247 | 21199 | 4.2 s |
| 2 | 9969 | 19938 | 4.8 s | 18 | 1247 | 22446 | 4.5 s |
| 3 | 6849 | 20547 | 5.8 s | 19 | 1246 | 23674 | 5.7 s |
| 4 | 4985 | 19940 | 4.3 s | 20 | 1246 | 24920 | 6.4 s |
| 5 | 4984 | 24920 | 3.8 s | 21 | 1246 | 26166 | 7.3 s |
| 6 | 3738 | 22428 | 7.7 s | 22 | 1246 | 27412 | 8.4 s |
| 7 | 3115 | 21805 | 9.0 s | 23 | 1246 | 28658 | 10.2 s |
| 8 | 2493 | 19944 | 4.4 s | 24 | 1246 | 29904 | 11.5 s |
| 9 | 2492 | 22428 | 5.0 s | 25 | 1246 | 31150 | 11.3 s |
| 10 | 2492 | 24920 | 5.9 s | 26 | 1246 | 32396 | 14.8 s |
| 11 | 1869 | 20559 | 4.9 s | 27 | 1246 | 33642 | 17.6 s |
| 12 | 1869 | 22428 | 5.6 s | 28 | 1246 | 34888 | 19.9 s |
| 13 | 1869 | 24297 | 6.6 s | 29 | 1246 | 36134 | 21.7 s |
| 14 | 1869 | 26166 | 6.7 s | 30 | 1246 | 37380 | 8.1 s |
| 15 | 1869 | 28035 | 7.1 s | 31 | 1246 | 38626 | 8.1 s |
| 16 | 1247 | 19952 | 4.1 s | | | | |

All 31 widths recovered and predicted the next value correctly.  Time is
dominated by two terms: building the observation matrix (≈ constant, one
batched `getrandbits` per canonical basis state) and the GF(2) elimination
(the cost depends on the fill-in of the matrix, so it is not monotone in the
width — v = 29 is the slowest point at 21.7 s).

For comparison, the original dense `int64` implementation needed hours for
these inputs: it replayed the whole schedule once per raw basis state and ran
a Python-level row loop over a matrix holding one 64-bit integer per bit
(~60× more memory traffic per row XOR).
## Correctness notes

* **Information bound: 19937 bits.** Fewer bits always return `False` from
  `check()`; more bits of a bad pattern (e.g. the same widths in a different
  order) can still leave the state undetermined, in which case `check()`
  raises instead of returning a wrong state.
* **Order matters.** The same multiset of widths can recover or not depending
  on their order.
* Legacy arguments `force_martix`, `force_sage`, `force_numpy` are accepted:
  `force_martix` disables the untempering and five-bit fast paths, and
  `force_sage` uses SageMath's M4RI elimination when Sage is installed (see
  `pyrandcracker.sagemath_support`).

## License

MIT
