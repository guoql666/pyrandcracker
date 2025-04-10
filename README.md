# Pyrandcracker

**Pyrandcracker** is a tool for predicting the random numbers generated by Python’s `random` library.

[English](https://github.com/guoql666/pyrandcracker/blob/master/README.md) | [中文](https://github.com/guoql666/pyrandcracker/blob/master/README-zh.md)

English Version is generate by GPT-4o

## Project Introduction

Pyrandcracker leverages the properties of random number generators (such as the MT19937 algorithm) by collecting enough random number samples to reverse-engineer the generator’s internal state and thus predict future random numbers.

## Features
- Supports operations on random numbers with any number of bits.
- Migrates some functions from the SageMath matrix module.
- Predicts the generator’s internal state when provided with at least 19937 bits of random numbers.

## Installation

```bash
$ pip install pyrandcracker
```

## Usage

### 32-bit Submission
The project supports input of numbers with any bit-length as long as the total submitted bits exceed 19937 bits. Due to the properties of MT19937, the project is optimized when the number of submitted bits is a multiple of 32. This optimization is only effective when each submission is exactly 32 bits or a multiple thereof.

If you strongly prefer solving using matrix methods in certain cases, you can enforce this by passing `force_matrix = True` when calling the solve method. (Not recommended.)

```python
from randcracker import RandCracker
import time
# Initialize the random number generator
rd = random.Random()
rd.seed(time.time())
# Initialize the predictor
rc = RandCracker()

data = [rd.getrandbits(64) for _ in range(312)]
for num in data:
    # Submitting a total of 312 * 64 = 19968 bits
    rc.submit(num)
# Check if the solution is obtainable and automatically solve
rc.check()

print(f"next random number is {rd.getrandbits(32)}")
# You can either use rc.rnd to access the cracked Random class,
# or use the rc.get_random function to explicitly obtain and save the variable.
print(f"predict next random number is {rc.rnd.getrandbits(32)}")
```

### Arbitrary Bit Submission

Sometimes you might not be able to obtain bits in multiples of 32, and the number of bits you get may vary. In this case, you need to solve a system of linear equations. You can specify the number of bits of each submitted number through the second parameter of the submit function to inform the predictor.

```python
from randcracker import RandCracker
import time
# Initialize the random number generator
rd = random.Random()
rd.seed(time.time())
# Initialize the predictor. The 'detail' parameter uses tqdm’s trange 
# to display a progress bar, but may slightly impact performance and produce unnecessary output.
# The default value for the detail parameter is False.
rc = RandCracker(detail = True)
data = [rd.getrandbits(16) for _ in range(624*2)]
for num in data:
    # Submitting a total of 624*2*16 = 19968 bits. You can submit more and it will still compute.
    rc.submit(num, 16)
# Check if the solution is obtainable and automatically solve
rc.check()
print(f"next random number is {rd.getrandbits(16)}")
print(f"predict next random number is {rc.rnd.getrandbits(16)}")
```
Note that due to limitations with numpy and Python, the solving process may be relatively slow (in the worst-case scenario, such as submitting 19937 numbers of 1 bit each, the prediction might take over an hour, so please be patient). Future optimizations with C-Python will be considered.

### Custom Function Prediction for Submissions

The built-in solver only supports consecutive random number submissions, but attackers often face situations where known information is discontinuous. If the specific generation process is known—such as knowing which parts of the data are discontinuous and how many random bits are skipped in-between—the internal state can still be recovered. The predictor provides the set_generator_func function to allow users to define a custom function for processing non-continuous submissions.

```python
from randcracker import RandCracker
import time
# Initialize the random number generator
rd = random.Random()
rd.seed(time.time())
# Initialize the predictor
rc = RandCracker(detail = False)
# First, generate 624 16-bit numbers
data16 = [rd.getrandbits(16) for _ in range(624)]
# Then drop one 16-bit random number
drop = rd.getrandbits(16)
# Then generate 624*2 8-bit numbers
data8 = [rd.getrandbits(8) for _ in range(624*2)]
for num in data16:
    # Submit 624 16-bit numbers
    rc.submit(num, 16)
for num in data8:
    # Submit 624*2 8-bit numbers
    rc.submit(num, 8)

# Define a custom function that takes a Random class as parameter. 
# The internal logic of the function must match the actual generation process, though the values can differ.
# In this example, first submit 624 16-bit numbers, then drop one 16-bit number,
# and finally submit 1248 8-bit numbers.
def getRows(rnd):
    rows = []
    for _ in range(624):
        # Note that list(map(int, (bin() )) is necessary,
        # and zfill must correspond with the specific bit-length,
        # i.e., generating 16 bits should use zfill with 16.
        rows += list(map(int, (bin(rnd.getrandbits(16))[2:].zfill(16)))) 
    drop = rnd.getrandbits(16)
    for _ in range(624*2):
        rows += list(map(int, (bin(rnd.getrandbits(8))[2:].zfill(8)))) 
    # Finally, return a list whose length is the total number of submitted bits, with each element being 0 or 1.
    return rows
# Pass the custom function to set_generator_func
rc.set_generator_func(getRows)
# Check if the solution is obtainable and automatically solve
rc.check()
print(f"next random number is {rd.getrandbits(16)}")
print(f"predict next random number is {rc.rnd.getrandbits(16)}")
```

### Moving Your Generator

The predictor also provides an offset function so you can freely move your random number generator. 
However, note that the offset is calculated based on submissions of random numbers that are 32 bits or less.
If you generate a 64-bit random number, you need to apply the offset function twice to get the same result.

```python
# Assuming rc has successfully predicted the state
number = rc.getrandbits(32)
# Use offset(-1) to roll back to the previous prediction
rc.offset(-1)
print(f"random number is {number}")
print(f"after offset, random number is {rc.rnd.getrandbits(32)}")
```

### Retaining the Original Generator

Sometimes, we may want the generator’s next random number to match exactly the first number we submitted, effectively preserving the original state.
You can certainly achieve this goal using the offset method. However, if you have used the set_generator_func method with a complex function, the program might take significantly longer to restore the random number generator to its current state. 
This not only adds extra waiting time but also increases complexity.

To address this, the solve method provides an `offset` parameter (default is `False`). By setting `offset = True`, you can retrieve the original generator directly.
