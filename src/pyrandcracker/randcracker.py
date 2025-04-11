class RandCracker:
    def __init__(self, detail = False):
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

        # bit_list example : [(1, 1), (number, bits), ...]
        self.bit_list = []
        self.M = []
        self.MT19937_state_list = []
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
            for _ in range(bits_round):
                sub_num = copy_num & 0xFFFFFFFF
                self._submit(sub_num)
                copy_num >>= 32
        else:
            self.use_martix = True
        
        self.bit_count += bits
        self.bit_list.append((num, bits))

    
    def check(self, force_martix = False, offset = False):
        if self.bit_count < 13397:
            raise ValueError("Not enough bits submitted. At least 19968 bits are required.")

        if self.use_martix or force_martix:
            return self._solve_martix(offset = offset)
            
        elif self.bit_count >= 19937:
            assert (len(self.MT19937_state_list) >= 624)
            self.MT19937_state_list = self.MT19937_state_list[-624:]
            self._regen(offset = offset)
            return True
        else:
            return False
        

    def get_random(self):
        return self.rnd


    def _submit(self, num: int):
        bits = self._to_bitarray(num)
        assert (all([x == 0 or x == 1 for x in bits]))
        self.MT19937_state_list.append(self._harden_inverse(bits))


    def _solve_martix(self, offset = False):
        n = len(self.bit_list)
        
        np = __import__("numpy")
        random = __import__("random")
        rng = random.Random()
        for i in self.trange(19968):
            state = [0]*624
            temp = "0"*i + "1"*1 + "0"*(19968-1-i)
            for j in range(624):
                state[j] = int(temp[32*j:32*j+32],2)
            rng.setstate((3,tuple(state+[624]),None)) 
            
            row = self._getRows(rng)
            # breakpoint()
            if len(row) != self.bit_count:
                raise ValueError("Row length mismatch")
            
            self.M.append(row)

        self.M = np.array(self.M, dtype=int) % 2

        y = []
        for num, bits in self.bit_list:
            y.extend(list(map(int, bin(num)[2:].zfill(bits))))

        y = np.array(y, dtype=int)
        
        from .matrix_utils import solve_left
        try:
            s = solve_left(self.M, y, trange = self.trange)
        except ValueError as e:
            print("solve_left error:", e)
            return False
        
        G=[]
        for i in range(624):
            C=0
            for j in range(32):
                C<<=1
                C|=int(s[32*i+j])

            G.append(C)
        
        self.rnd = random.Random()
        for i in range(624):
            G[i]=int(G[i])
        state_result = (int(3), tuple(G+[int(624)]), None)
        self.rnd.setstate(state_result)

        if not offset:
            self._getRows(self.rnd)
        
        return True


    def _getRows(self, rng):
        row=[]
        for _, bits in self.bit_list:
            row+=list(map(int, (bin(rng.getrandbits(bits))[2:].zfill(bits))))
        return row


    def set_generator_func(self, func):
        """
        This function generates a row of bits based on the random number generator (rng).
        must same as your want to predict random method.

        :param rng: A random number generator object with a method getrandbits(bits)
            that generates a random number with the specified number of bits.
        :return: A list of integers (0 or 1) representing the binary representation
                of the generated random numbers concatenated together.

        e.g.
        def func(rng):
            row=[]
            for _, bits in self.bit_list:
                #need to attention at zfill 
                row+=list(map(int, (bin(rng.getrandbits(bits))[2:].zfill(bits))))
            return row
        """
        self._getRows = func


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


    def _regen(self, offset = False):
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
            y = self._or_nums(self._and_nums(self.MT19937_state_list[kk], u_bits), self._and_nums(self.MT19937_state_list[kk + 1], l_bits))
            self.MT19937_state_list[kk] = self._xor_nums(self._xor_nums(self.MT19937_state_list[kk + M], y[:-1]), mag01[y[-1] & 1])

        for kk in range(N - M, N - 1):
            y = self._or_nums(self._and_nums(self.MT19937_state_list[kk], u_bits), self._and_nums(self.MT19937_state_list[kk + 1], l_bits))
            self.MT19937_state_list[kk] = self._xor_nums(self._xor_nums(self.MT19937_state_list[kk + (M - N)], y[:-1]), mag01[y[-1] & 1])

        y = self._or_nums(self._and_nums(self.MT19937_state_list[N - 1], u_bits), self._and_nums(self.MT19937_state_list[0], l_bits))
        self.MT19937_state_list[N - 1] = self._xor_nums(self._xor_nums(self.MT19937_state_list[M - 1], y[:-1]), mag01[y[-1] & 1])
        
        self.rnd = __import__("random").Random()
        if not offset:
            self._untwist()
            state = [self._to_int(x) for x in self.MT19937_state_list] + [624]
            self.rnd.setstate((3, tuple(state), None))
        else:
            state = [self._to_int(x) for x in self.MT19937_state_list] + [624]
            self.rnd.setstate((3, tuple(state), None))


    def _untwist(self):
        if self.rnd is None:
            raise ValueError("Random number generator not initialized.")
        
        w, n, m = 32, 624, 397
        a = 0x9908B0DF

        # I like bitshifting more than these custom functions...
        MT = [self._to_int(x) for x in self.MT19937_state_list]

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

        self.MT19937_state_list = [self._to_bitarray(x) for x in MT]


    def offset(self, n):
        if self.rnd is None:
            raise ValueError("Random number generator not initialized.")
        
        if n >= 0:
            [self.rnd.getrandbits(32) for _ in range(n)]
        else:
            state = list(self.rnd.getstate()[1])
            cycle = state[-1]
            # hint: n is postive, cycle is negative
            #       so pos actually equals cycle - abs(n)
            pos = cycle + n

            if pos >= 1:
                self.rnd.setstate((3, tuple(state[:-1] + [pos]), None)) 
            else:
                self.MT19937_state_list = [self._to_bitarray(x) for x in state[:-1] ]
                [self._untwist() for _ in range(-n // 624 + 1)]
                new_state = [self._to_int(x) for x in self.MT19937_state_list] + [624]
                self.rnd.setstate((3, tuple(new_state), None))
                [self.rnd.getrandbits(32) for _ in range(624 - (-n % 624) )]

    
    def offset_bits(self, bits):
        if bits >= 0:
            self.rnd.getrandbits(bits)
        else:
            rounds = (-bits) // 32
            plus = 0 if (-bits) % 32 == 0 else 1
            self.offset(-(rounds + plus))
