# -*- coding: utf-8 -*-
"""RandCracker: recover a Python ``random.Random`` state from its outputs.

The implementation follows the companion paper "State Recovery from
Truncated Mersenne Twister Outputs: Delayed States and Optimal Observation
Schedules".  Three recovery paths are used automatically:

* full 32-bit words (any multiple of 32 bits per submission):
  exact untempering, no linear algebra (624 words = 19968 bits suffice,
  and this is the minimum for full words since the canonical state has
  19937 bits and the last word is indivisible);
* uniform 5-bit observations, >= 4984 calls: the paper's constructive
  residual inverse (carrier families a_i/h_i plus the tabulated extension
  state) with a linear-cost correction after the prefix solve;
* anything else (mixed widths, top-v truncations, custom replays):
  a bit-packed GF(2) solve over the 19937 canonical coordinates.

Semantics of the recovered generator ``self.rnd``:

* ``check()`` (default) positions it right AFTER the last submitted
  output, so the next ``getrandbits`` predicts the following value;
* ``check(offset=True)`` positions it so its next outputs replay the
  submitted sequence from the first (retained) observation.

``offset(n)``/``offset_bits(bits)`` move the generator forward/backward;
the internal state is always kept as a genuine post-twist array, so
backward moves across block boundaries are exact.
"""
from __future__ import annotations

import random

from . import fivebit, gf2
from .mt19937 import (CANON_BITS, N, canonical_words_from_bits, set_state,
                      temper, twist, untemper, untwist)

MIN_BITS = CANON_BITS                    # 19937: information lower bound
MIN_WORDS_FAST = 624                     # full-word fast path


class RandCracker:
    def __init__(self, detail: bool = False):
        self.rnd = None
        self.bit_count = 0
        if detail:
            try:
                self.trange = __import__("tqdm").trange
            except ModuleNotFoundError:
                print("Warning: tqdm module not found, using range instead.")
                self.trange = range
        else:
            self.trange = range

        # bit_list: [(num, bits), ...] in submission order
        self.bit_list = []
        self.M = []                       # legacy attribute (unused now)
        self.MT19937_state_list = []      # untempered words of the fast path
        self.use_martix = False           # legacy spelling of use_matrix
        self._generator_func = None

    # ------------------------------------------------------------- input

    def submit(self, num: int, bits: int = 32):
        """Submit one observed value.

        :param num:  the observed value (as returned by ``getrandbits``).
        :param bits: how many bits that value had (default 32).
        """
        if self.rnd is not None:
            raise RuntimeError("state already recovered; create a new "
                               "RandCracker for a new stream")
        if not isinstance(num, int) or not isinstance(bits, int):
            raise TypeError("num and bits must be integers")
        if bits < 1:
            raise ValueError("bits must be >= 1")
        if num < 0:
            raise ValueError("num must be non-negative")
        if num.bit_length() > bits:
            raise ValueError("num needs %d bits, more than the specified %d"
                             % (num.bit_length(), bits))

        if bits % 32 == 0 and not self.use_martix:
            copy_num = num
            for _ in range(bits // 32):
                self.MT19937_state_list.append(untemper(copy_num & 0xFFFFFFFF))
                copy_num >>= 32
        else:
            # a non-word submission makes every earlier word submission part
            # of the general linear system as well
            self.use_martix = True

        self.bit_count += bits
        self.bit_list.append((num, bits))

    def set_generator_func(self, func):
        """Register a custom replay function for non-consecutive streams.

        ``func(rng)`` must consume exactly the same random calls that
        produced the submitted bits (same widths, same order, including any
        skipped values) and return the concatenated observed bits as a list
        of 0/1, most significant first inside each submitted value.

        e.g. for 624 x getrandbits(16), one dropped getrandbits(16), then
        1248 x getrandbits(8)::

            def getRows(rnd):
                rows = []
                for _ in range(624):
                    rows += list(map(int, bin(rnd.getrandbits(16))[2:].zfill(16)))
                drop = rnd.getrandbits(16)
                for _ in range(624 * 2):
                    rows += list(map(int, bin(rnd.getrandbits(8))[2:].zfill(8)))
                return rows
        """
        self._generator_func = func
        self.use_martix = True

    # ------------------------------------------------------------ solving

    def check(self, force_martix=False, offset=False, force_sage=False,
              force_numpy=False):
        """Try to recover the state; returns True on success.

        Returns False while fewer than 19937 bits (the information bound)
        have been submitted.  Raises ValueError when the submitted bits can
        never or do not uniquely determine a state.

        :param force_martix: disable the untemper and five-bit fast paths.
        :param offset: if True, position the recovered generator at the
            first (retained) submitted output instead of after the last.
        :param force_sage / force_numpy: solver backends (legacy).
        """
        if self.rnd is not None:
            return True
        if self.bit_count < MIN_BITS:
            return False

        widths = [bits for _, bits in self.bit_list]

        if not force_martix and self._generator_func is None \
                and not self.use_martix:
            # fast path: everything came as full 32-bit words
            if len(self.MT19937_state_list) >= MIN_WORDS_FAST:
                return self._solve_words(offset=offset)

        if not force_martix and self._generator_func is None \
                and all(w == fivebit.WIDTH for w in widths) \
                and len(widths) >= fivebit.HORIZON:
            # the paper's constructive five-bit recovery
            return self._solve_fivebit(offset=offset)

        return self._solve_generic(offset=offset, force_sage=force_sage,
                                   force_numpy=force_numpy)

    def get_random(self):
        """The recovered random.Random (None before a successful check)."""
        return self.rnd

    # --------------------------------------------------------- fast paths
    # --------------------------------------------------------- fast paths

    def _solve_words(self, offset=False):
        words = [w & 0xFFFFFFFF
                 for w in self.MT19937_state_list[-MIN_WORDS_FAST:]]
        # the retained 624 words are consecutive tempered outputs; verify
        # the untempering roundtrip and the submission->word ordering
        stream = []
        for num, bits in self.bit_list:
            n = num
            for _ in range(bits // 32):
                stream.append(n & 0xFFFFFFFF)
                n >>= 32
        retained = stream[-MIN_WORDS_FAST:]
        if len(retained) < MIN_WORDS_FAST or \
                any(temper(w) != r for w, r in zip(words, retained)):
            raise ValueError("word submissions are not 624 consecutive "
                             "32-bit outputs of one stream")
        rng = random.Random()
        if offset:
            # next output replays the first retained submission; a window of
            # 624 consecutive real outputs is a genuine twist image, so
            # deeper backward moves stay exact
            set_state(rng, words, index=0)
            self.rnd = rng
        else:
            set_state(rng, words, index=0)
            for _ in range(MIN_WORDS_FAST):
                rng.getrandbits(32)
            self.rnd = rng
        return True

    def _solve_fivebit(self, offset=False):
        words = fivebit.solve_fivebit(self.bit_list, trange=self.trange)
        return self._finish(words, offset)

    def _solve_generic(self, offset=False, force_sage=False,
                       force_numpy=False):
        rows, _ = gf2.build_system(self.bit_list,
                                   generator_func=self._generator_func,
                                   trange=self.trange)
        x = None
        if force_sage and not force_numpy:
            from . import sagemath_support
            if sagemath_support.have_sage() and \
                    rows.shape[0] * (CANON_BITS + 1) <= sagemath_support.SAGE_MAX_CELLS:
                x = sagemath_support.solve_gf2_sage(rows, CANON_BITS)
        if x is None:
            x = gf2.solve_rows(rows)
        return self._finish(canonical_words_from_bits(x), offset)

    def _finish(self, words, offset):
        """Install the solved state, verify it, then position as asked."""
        rng = random.Random()
        set_state(rng, twist(list(words)), index=0)
        self.rnd = rng
        self._verify_replay()          # from this position the submissions
        if not offset:                 # are reproduced exactly
            self._replay_all()
        return True

    # ---------------------------------------------------------- utilities

    def _replay_all(self):
        """Consume exactly the submitted calls, positioning after them."""
        if self._generator_func is not None:
            self._generator_func(self.rnd)
        else:
            for _, bits in self.bit_list:
                self.rnd.getrandbits(bits)

    def _verify_replay(self):
        """Check the recovered generator against a fresh replay."""
        rng = random.Random()
        rng.setstate(self.rnd.getstate())
        if self._generator_func is not None:
            bits = self._generator_func(rng)
            expect = []
            for num, b in self.bit_list:
                expect.extend((num >> (b - 1 - p)) & 1 for p in range(b))
            if list(bits) != expect:
                raise ValueError(
                    "recovered state does not reproduce the submissions")
        else:
            for t, (num, b) in enumerate(self.bit_list):
                if rng.getrandbits(b) != num:
                    raise ValueError(
                        "recovered state does not reproduce submission %d" % t)

    # ---------------------------------------------------------- moving

    def offset(self, n: int):
        """Move the recovered generator by n whole 32-bit words.

        Positive n consumes outputs; negative n rewinds (exact, including
        across 624-word block boundaries).
        """
        if self.rnd is None:
            raise ValueError("Random number generator not initialized.")
        if n >= 0:
            for _ in range(n):
                self.rnd.getrandbits(32)
            return
        state = list(self.rnd.getstate()[1])
        words, idx = state[:N], state[N]
        idx -= -n
        while idx < 0:
            words = untwist(words)
            idx += N
        self.rnd.setstate((3, tuple(words) + (idx,), None))

    def offset_bits(self, bits: int):
        """Move by `bits` random bits (word-aligned when rewinding).

        Forward moves replay getrandbits(bits); a rewind of k bits rewinds
        ceil(k/32) words, since every getrandbits call consumes whole
        32-bit words.
        """
        if self.rnd is None:
            raise ValueError("Random number generator not initialized.")
        if bits >= 0:
            self.rnd.getrandbits(bits)
        else:
            k = -bits
            self.offset(-((k + 31) // 32))
