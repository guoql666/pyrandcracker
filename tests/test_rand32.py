
import os
import sys
import random
import pytest

from src.pyrandcracker import RandCracker

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_rand32():
    rd = random.Random()
    rd.seed(2)
    rc = RandCracker()
    data = [rd.getrandbits(32) for _ in range(625)]
    for num in data:
        rc.submit(num)
    rc.check()
    target_num = rd.getrandbits(32)
    predicted_num = rc.rnd.getrandbits(32)
    assert target_num == predicted_num
    print(target_num)
    print(predicted_num)


def test_offset32():
    rd = random.Random()
    rd.seed(2)
    rc = RandCracker()
    data = [rd.getrandbits(32) for _ in range(625)]
    for num in data:
        rc.submit(num)
    rc.check()
    rc.offset(-1)
    target_num = data[-1]
    predicted_num = rc.rnd.getrandbits(32)
    assert target_num == predicted_num
    print(target_num)
    print(predicted_num)
    

@pytest.mark.skip
def test_rand16():
    rd = random.Random()
    rd.seed(2)
    rc = RandCracker(detail=True)
    data = [rd.getrandbits(16) for _ in range(624*2)]
    for num in data:
        rc.submit(num, 16)
    rc.check()
    target_num = rd.getrandbits(16)
    predicted_num = rc.rnd.getrandbits(16)
    assert target_num == predicted_num
    print(target_num)
    print(predicted_num)


@pytest.mark.skip
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
        drop = rnd.getrandbits(16)
        for _ in range(624*2):
            rows += list(map(int, (bin(rnd.getrandbits(8))[2:].zfill(8)))) 
        return rows
    pass

    rc.set_generator_func(getRows)
    rc.check()
    target_num = rd.getrandbits(16)
    predicted_num = rc.rnd.getrandbits(16)
    assert target_num == predicted_num
    print(target_num)
    print(predicted_num)


if __name__ == "__main__":
    test_rand16()