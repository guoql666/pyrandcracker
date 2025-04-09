class RandCracker:
    def __init__(self):
        self.rnd = None
        self.bit_count = 0
        self.counter = 0
        # bit_list example : [(1, 1), (number, bits), ...]
        self.bit_list = []
        self.MT19937_bit32_list = []
        self.use_martix = False

    def submit(self, num: int, bits: int = 32):
        """
        Submit a random number to the cracker.
        :param num: The random number to submit.
        :param bits: The number of bits in the random number (default is 32).
        """

        if bits % 32 == 0 and not self.use_martix:
            bits_round = bits // 32
            copy_num = num
            submit_list = []
            for _ in range(bits_round):
                submit_list.append(copy_num & 0xFFFFFFFF)
                copy_num >>= 32
            
            for sub_num in submit_list:
                self._submit(sub_num)
        
        else:
            self.use_martix = True
        
        self.bit_count += bits
        self.bit_list.append((num, bits))

    
    def check(self, force_martix = False):

        if self.bit_count < 19968:
            raise ValueError("Not enough bits submitted. At least 19968 bits are required.")

        if self.use_martix or force_martix:
            self.solve_martix()
            return True
        elif self.bit_count >= 19968:
            assert (len(self.MT19937_bit32_list) >= 624)
            self.MT19937_bit32_list = self.MT19937_bit32_list[-624:]
            self._regen()
            return True
        else:
            return False
        

    def _submit(self, num: int):
        bits = self._to_bitarray(num)
        assert (all([x == 0 or x == 1 for x in bits]))
        self.counter += 1
        self.MT19937_bit32_list.append(self._harden_inverse(bits))


    def solve_martix(self):
        pass


    def _predict_32(self):
        if self.bit_count < 19968:
            raise ValueError("Didn't recieve enough bits to predict")

        if self.counter >= 624:
            self._regen()
        self.counter += 1

        return self._harden(self.MT19937_bit32_list[self.counter - 1])


    def _to_bitarray(self, num):
        k = [int(x) for x in bin(num)[2:]]
        return [0] * (32 - len(k)) + k


    def _to_int(self, bits):
        return int("".join(str(i) for i in bits), 2)


    def _or_nums(self, a, b):
        if len(a) < 32:
            a = [0] * (32 - len(a)) + a
        if len(b) < 32:
            b = [0] * (32 - len(b)) + b

        return [x[0] | x[1] for x in zip(a, b)]


    def _xor_nums(self, a, b):
        if len(a) < 32:
            a = [0] * (32 - len(a)) + a
        if len(b) < 32:
            b = [0] * (32 - len(b)) + b

        return [x[0] ^ x[1] for x in zip(a, b)]


    def _and_nums(self, a, b):
        if len(a) < 32:
            a = [0] * (32 - len(a)) + a
        if len(b) < 32:
            b = [0] * (32 - len(b)) + b

        return [x[0] & x[1] for x in zip(a, b)]


    def _decode_harden_midop(self, enc, and_arr, shift):

        NEW = 0
        XOR = 1
        OK = 2
        work = []
        for i in range(32):
            work.append((NEW, enc[i]))
        changed = True
        while changed:
            changed = False
            for i in range(32):
                status = work[i][0]
                data = work[i][1]
                if i >= 32 - shift and status == NEW:
                    work[i] = (OK, data)
                    changed = True
                elif i < 32 - shift and status == NEW:
                    if and_arr[i] == 0:
                        work[i] = (OK, data)
                        changed = True
                    else:
                        work[i] = (XOR, data)
                        changed = True
                elif status == XOR:
                    i_other = i + shift
                    if work[i_other][0] == OK:
                        work[i] = (OK, data ^ work[i_other][1])
                        changed = True

        return [x[1] for x in work]


    def _harden(self, bits):
        bits = self._xor_nums(bits, bits[:-11])
        bits = self._xor_nums(bits, self._and_nums(bits[7:] + [0] * 7, self._to_bitarray(0x9d2c5680)))
        bits = self._xor_nums(bits, self._and_nums(bits[15:] + [0] * 15, self._to_bitarray(0xefc60000)))
        bits = self._xor_nums(bits, bits[:-18])
        return bits


    def _harden_inverse(self, bits):
        # inverse for: bits = _xor_nums(bits, bits[:-11])
        bits = self._xor_nums(bits, bits[:-18])
        # inverse for: bits = _xor_nums(bits, _and_nums(bits[15:] + [0] * 15 , _to_bitarray(0xefc60000)))
        bits = self._decode_harden_midop(bits, self._to_bitarray(0xefc60000), 15)
        # inverse for: bits = _xor_nums(bits, _and_nums(bits[7:] + [0] * 7 , _to_bitarray(0x9d2c5680)))
        bits = self._decode_harden_midop(bits, self._to_bitarray(0x9d2c5680), 7)
        # inverse for: bits = _xor_nums(bits, bits[:-11])
        bits = self._xor_nums(bits, [0] * 11 + bits[:11] + [0] * 10)
        bits = self._xor_nums(bits, bits[11:21])

        return bits


    def _regen(self):
        # C code translated from python sources
        N = 624
        M = 397
        MATRIX_A = 0x9908b0df
        LOWER_MASK = 0x7fffffff
        UPPER_MASK = 0x80000000
        mag01 = [self._to_bitarray(0), self._to_bitarray(MATRIX_A)]

        l_bits = self._to_bitarray(LOWER_MASK)
        u_bits = self._to_bitarray(UPPER_MASK)

        for kk in range(0, N - M):
            y = self._or_nums(self._and_nums(self.MT19937_bit32_list[kk], u_bits), self._and_nums(self.MT19937_bit32_list[kk + 1], l_bits))
            self.MT19937_bit32_list[kk] = self._xor_nums(self._xor_nums(self.MT19937_bit32_list[kk + M], y[:-1]), mag01[y[-1] & 1])

        for kk in range(N - M, N - 1):
            y = self._or_nums(self._and_nums(self.MT19937_bit32_list[kk], u_bits), self._and_nums(self.MT19937_bit32_list[kk + 1], l_bits))
            self.MT19937_bit32_list[kk] = self._xor_nums(self._xor_nums(self.MT19937_bit32_list[kk + (M - N)], y[:-1]), mag01[y[-1] & 1])

        y = self._or_nums(self._and_nums(self.MT19937_bit32_list[N - 1], u_bits), self._and_nums(self.MT19937_bit32_list[0], l_bits))
        self.MT19937_bit32_list[N - 1] = self._xor_nums(self._xor_nums(self.MT19937_bit32_list[M - 1], y[:-1]), mag01[y[-1] & 1])

        self.counter = 0


    def untwist(self):
        w, n, m = 32, 624, 397
        a = 0x9908B0DF
        
        # I like bitshifting more than these custom functions...
        MT = [self._to_int(x) for x in self.MT19937_bit32_list]

        for i in range(n-1, -1, -1):
            result = 0
            tmp = MT[i]
            tmp ^= MT[(i + m) % n]
            if tmp & (1 << w-1):
                tmp ^= a
            result = (tmp << 1) & (1 << w-1)
            tmp = MT[(i - 1 + n) % n]
            tmp ^= MT[(i + m-1) % n]
            if tmp & (1 << w-1):
                tmp ^= a
                result |= 1
            result |= (tmp << 1) & ((1 << w-1) - 1)
            MT[i] = result

        self.MT19937_bit32_list = [self._to_bitarray(x) for x in MT]


    def offset(self, n):
        if n >= 0:
            [self._predict_32() for _ in range(n)]
        else:
            [self.untwist() for _ in range(-n // 624 + 1)]
            [self._predict_32() for _ in range(624 - (-n % 624))]
        