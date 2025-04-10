# Pyrandcracker

**Pyrandcracker** 是一个可以预测 Python `random` 库生成随机数的工具。

[English](https://github.com/guoql666/pyrandcracker/blob/master/README.md) | [中文](https://github.com/guoql666/pyrandcracker/blob/master/README-zh.md)

英文版由GPT-4o生成

## 项目简介

Pyrandcracker 利用随机数生成器（如 MT19937 算法）的特性，通过收集足够的随机数样本，逆向推导出生成器的内部状态，从而预测后续生成的随机数。

## 功能特点
- 支持任意比特位随机数进行运算
- 迁移部分sagemath矩阵功能
- 输入19937位随机数，预测其生成器内部状态

## 安装

```bash
$ pip install pyrandcracker
```

## 使用方法

### 32位提交
项目支持任意比特位输入，仅需总提交位数高于19937位
由于MT19937特性，项目对提交位数为32的任意倍进行优化。
该优化仅适用于每次提交的均为32或其倍数位时有效。
如果有时您强烈希望使用矩阵方法求解，您可以通过在调用solve方法时添加force_martix = True来强制使用矩阵方法求解。（不建议）

```python
from randcracker import RandCracker
import time
# 初始化随机数生成器
rd = random.Random()
rd.seed(time.time())
# 初始化预测器
rc = RandCracker()

data = [rd.getrandbits(64) for _ in range(312)]
for num in data:
    # 提交共计312 * 64 = 19968位
    rc.submit(num)
# 检查是否可解并自动求解
rc.check()

print(f"next random number is {rd.getrandbits(32)}")
# 除了可以使用rc.rnd来获取破解后的Random类外，也可以使用rc.get_random函数来显式获取并保存变量
print(f"predict next random number is {rc.rnd.getrandbits(32)}")
```

### 任意位数提交

但有时候我们并不一定都能获取到32的倍数位，有可能会或多或少
此时需要对线性方程组进行求解。
此时可以通过对submit函数提交每次提交的比特位来告诉预测器，提交的数值对应的是多少位。

```python
from randcracker import RandCracker
import time
# 初始化随机数生成器
rd = random.Random()
rd.seed(time.time())
# 初始化预测器, detail参数会调用tqdm库的trange，来显示进度条，但会稍微影响性能并造成不必要的输出
# 默认detail参数为False
rc = RandCracker(detail = True)
data = [rd.getrandbits(16) for _ in range(624*2)]
for num in data:
    # 提交共计624*2*16 = 19968位，当然提交多了也可以进行计算
    rc.submit(num, 16)
# 检查是否可解并自动求解
rc.check()
print(f"next random number is {rd.getrandbits(16)}")
print(f"predict next random number is {rc.rnd.getrandbits(16)}")
```
需要注意的是，由于numpy和python语言限制，求解速度会相对较慢( 当最极端情况，如提交了19937个1bit时，预测时间可能会超过1小时，请耐心等待 )。后续会考虑使用cpython进行优化。

### 自定义函数预测提交

内置的求解器仅可以求解连续的随机数提交，但攻击者经常会遇到已知的信息不连续的情况，如果能获取其具体的生成情况，如知道已知信息中哪些是不连续的，且知道中间间隔了多少个多少比特的随机数，则仍然可以进行恢复。
预测器提供了set_generator_func函数接口，允许用户对非连续的状态提交自定义函数进行处理

```python
from randcracker import RandCracker
import time
# 初始化随机数生成器
rd = random.Random()
rd.seed(time.time())
# 初始化预测器
rc = RandCracker(detail = False)
# 先生成了624位16位数
data16 = [rd.getrandbits(16) for _ in range(624)]
# 中间舍弃了1个16位随机数
drop = rd.getrandbits(16)
# 然后生成了624*2个8位的随机数
data8 = [rd.getrandbits(8) for _ in range(624*2)]
for num in data16:
    # 提交624个16位随机数
    rc.submit(num, 16)
for num in data8:
    # 提交624*2个8位随机数
    rc.submit(num, 8)

# 自定义函数，接受一个Random类，该Random类来自内置库random，要求函数内部必须与实际生成情况一致，但可以值不相同
# 如在本例，先提交了624个16位随机数，然后舍弃一个16位随机数后，又提交了1248个8位随机数
# 那么在其中，你也必须先使用传入的rnd参数对其先生成624个16位随机数，然后舍弃一个16位随机数后，再生成1248个8位随机数
def getRows(rnd):
    rows = []
    for _ in range(624):
        # 这里需要注意，list(map(int, (bin() ))是必要的，且zfill也需要与对应位数保持一致
        # 即生成16位，zfill中也需要填16, 下同。
        rows += list(map(int, (bin(rnd.getrandbits(16))[2:].zfill(16)))) 
    drop = rnd.getrandbits(16)
    for _ in range(624*2):
        rows += list(map(int, (bin(rnd.getrandbits(8))[2:].zfill(8)))) 
    # 最后返回一个列表，列表长度是提交的总位数，列表的每一位均是0或1
    return rows
# 通过set_generator_func传入自定义的函数
rc.set_generator_func(getRows)
# 检查是否可解并自动求解
rc.check()
print(f"next random number is {rd.getrandbits(16)}")
print(f"predict next random number is {rc.rnd.getrandbits(16)}")
```

### 移动你的生成器

同时，预测器提供了offset函数，你可以自由的移动你的random随机数生成器。但注意，这里的偏移是按照每次生成小于等于32位随机数来计算的，如果你生成了64位随机数，则需要offset两次才能得到相同的答案。
```python
# 这里假设你的rc已经成功得到预测了
number = rc.getrandbits(32)
# 使用offset(-1)倒回到上一次预测
rc.offset(-1)
print(f"random number is {number}")
print(f"after offset, random number is {rc.rnd.getrandbits(32)}")
```

### 保留原生成器

有时候，我们可能希望生成器下次生成的数正好是我们第一次提交的数，即保留原状态
您当然可以通过offset方法实现这一目标。但如果使用了set_generator_func方法，且设置的函数较为复杂
则程序可能会花费更长的时间去得到当前状态的随机数生成器，您不仅需要额外等待，且更为繁琐。
因此，我们在solve中提供了offset参数，默认为False，您可以通过设置offset = True来获取原生成器。
