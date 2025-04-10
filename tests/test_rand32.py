
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pyrandcracker import RandCracker
import random


def test_rand32():
    rd = random.Random()
    rd.seed(2)
    rc = RandCracker()
    data = [rd.getrandbits(32) for _ in range(625)]
    for num in data:
        rc.submit(num)
    rc.check()
    print(rd.getrandbits(32))
    print(rc.rnd.getrandbits(32))


def test_offset32():
    rd = random.Random()
    rd.seed(2)
    rc = RandCracker()
    data = [rd.getrandbits(32) for _ in range(625)]
    for num in data:
        rc.submit(num)
    rc.check()
    rc.offset(-1)
    print(data[-1])
    print(rc.rnd.getrandbits(32))
    

def test_rand16():
    rd = random.Random()
    rd.seed(2)
    rc = RandCracker(detail=True)
    data = [rd.getrandbits(16) for _ in range(624*2)]
    for num in data:
        rc.submit(num, 16)
    rc.check()
    print(rd.getrandbits(16))
    print(rc.rnd.getrandbits(16))
    # breakpoint()

def test_rand_func():
    rd = random.Random()
    rd.seed(2)
    rc = RandCracker(detail=True)
    data16 = [rd.getrandbits(16) for _ in range(624)]
    drop = rd.getrandbits(16)
    data8 = [rd.getrandbits(8) for _ in range(624*2)]
    for num in data16:
        rc.submit(num, 16)
    for num in data8:
        rc.submit(num, 8)

    # generate the rows, must same as submit data`s  generate method
    def getRows(rnd):
        rows = []
        for _ in range(624):
            rows += list(map(int, (bin(rnd.getrandbits(16))[2:].zfill(16)))) 
        drop = rd.getrandbits(16)
        for _ in range(624*2):
            rows += list(map(int, (bin(rnd.getrandbits(8))[2:].zfill(8)))) 
        return rows
    
    rc.set_generator_func(getRows)
    rc.check()
    print(rd.getrandbits(16))
    print(rc.rnd.getrandbits(16))

if __name__ == "__main__":
    test_rand_func()