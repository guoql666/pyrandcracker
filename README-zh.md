# Pyrandcracker

**Pyrandcracker** 通过收集 Python `random.Random`（MT19937）的输出，反推其内部状态，
从而预测后续随机数。

[English](README.md) | [中文](README-zh.md)

## 原理

实现遵循 MT19937 的规范态（canonical state）模型：

* 预测状态共 **19937 比特**（下一个原始字的低 31 位不影响输出流）；
* 一个完整的 32 位输出是一个可逆（已 temper）的状态字，因此 624 个连续字即可直接确定状态；
* 截断输出（`getrandbits(v)`，`v < 32` 时返回一个字的高 `v` 位）是状态上的线性函数，
  因此恢复问题转化为 GF(2) 上的线性方程组。

程序自动选择三条恢复路径：

| 输入 | 方法 | 实测开销 |
| --- | --- | --- |
| 32 位的整数倍 | 精确 untemper | 624 字 = 19968 比特，瞬间完成 |
| 均匀 5 比特 | 论文的载波/扩展态构造反演 | 4984 次调用，约 3.8 秒 |
| 其他任意宽度 | 位打包 GF(2) 消元 | 4~22 秒（见性能） |

## 安装

```bash
pip install pyrandcracker
```

## 用法

理论上至少需要 19937 个观测比特（即状态维数）。当已提交的比特无法唯一确定状态时，
求解器会直接报错，而不是静默返回一个错误的状态。

### 32 位输出

```python
import random
import time
from pyrandcracker import RandCracker

rd = random.Random()
rd.seed(time.time())

rc = RandCracker()
for _ in range(624):
    rc.submit(rd.getrandbits(32))      # 624 字 = 19968 比特

rc.check()                             # 返回 True；rc.rnd 此时位于已提交输出之后
print(rd.getrandbits(32))
print(rc.rnd.getrandbits(32))          # 结果相同
```

### 截断输出（任意宽度）

```python
import random
from pyrandcracker import RandCracker

rd = random.Random()
rd.seed(1)

rc = RandCracker()
for _ in range(624 * 2):
    rc.submit(rd.getrandbits(16), 16)  # 19968 比特的 16 位值

rc.check()
print(rd.getrandbits(16))
print(rc.rnd.getrandbits(16))
```

均匀宽度 `getrandbits(v)` 的最小恢复时域：

| v | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 调用次数 | 19937 | 9969 | 6849 | 4985 | 4984 | 3738 | 2493 | 1247 | 624 |

以上都是**最小**量：提交多于最小值同样会被接受。32 位快路径只取最后 624 个字；
五比特路径用前 4984 次求解、再对全部提交值做回放校验；通用路径把每一个提交比特
都纳入线性方程组（超定但相容）。例如提交 2000 个 16 位值（32000 比特）约 7 秒恢复。

### 5 比特输出（构造性路径）

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

### 不连续的观测

如果观测值之间存在被跳过的调用，用 `set_generator_func` 描述精确的生成过程。
该函数在一个生成器上重放该过程，并返回观测到的比特（每个值内部高位在前）：

```python
import random
from pyrandcracker import RandCracker

rd = random.Random()
rd.seed(3)

rc = RandCracker()
for _ in range(624):
    rc.submit(rd.getrandbits(16), 16)
rd.getrandbits(32)                     # 一个被跳过的字（未提交）
for _ in range(1872):
    rc.submit(rd.getrandbits(8), 8)

def getRows(rnd):
    rows = []
    for _ in range(624):
        rows += list(map(int, bin(rnd.getrandbits(16))[2:].zfill(16)))
    rnd.getrandbits(32)                # 被跳过的调用
    for _ in range(1872):
        rows += list(map(int, bin(rnd.getrandbits(8))[2:].zfill(8)))
    return rows

rc.set_generator_func(getRows)
rc.check()
print(rd.getrandbits(16))
print(rc.rnd.getrandbits(16))
```

并非任意缺口都可恢复：当观测位置无法确定状态时，求解器会抛出 `ValueError`（见下）。

### 移动生成器

```python
number = rc.rnd.getrandbits(32)
rc.offset(-1)                          # 回退一个 32 位字
print(rc.rnd.getrandbits(32) == number)

rc.offset_bits(-150)                   # 回退 150 比特（按字对齐）
```

`check(offset=True)` 会把生成器定位到"重放已提交序列的第一个观测"，而不是继续其后。

## 性能

环境 Python 3.13（Windows），随机种子 `350234`，每种宽度使用论文的最小恢复时域
（`n_min(v)` 次调用）：

| v | 调用 | 比特 | 耗时 | v | 调用 | 比特 | 耗时 |
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

31 种宽度全部恢复成功并正确预测下一个值。耗时主要来自两部分：构建观测矩阵
（近似常数，每个规范基态一次批量 `getrandbits`）与 GF(2) 消元（成本取决于矩阵
填充量，因此对宽度并不单调——v = 29 是最慢点，21.7 秒）。

作为对比，原实现（稠密 `int64`）在这些输入上需要小时级：它为每个原始基态重放
整段 schedule，并对"每个比特占一个 64 位整数"的矩阵跑 Python 级行循环（每次行
异或的内存流量约为现在的 60 倍）。

## 正确性说明

* **信息下界为 19937 比特。** 比特不足时 `check()` 一律返回 `False`；而比特够多但
  观测模式不佳（例如同样的宽度组合换了顺序）时，状态仍可能无法确定——此时
  `check()` 会抛错，而不是返回错误状态。
* **顺序有影响。** 相同的宽度多重集，改变顺序可能导致可恢复或不可恢复。
* 兼容旧参数 `force_martix`、`force_sage`、`force_numpy`：`force_martix` 会禁用
  快速 untemper 与五比特路径；`force_sage` 在安装了 SageMath 时使用其 M4RI 消元
  （见 `pyrandcracker.sagemath_support`）。

## 许可

MIT
