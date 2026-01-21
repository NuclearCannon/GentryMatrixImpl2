# Gentry Matrix Impl (Python)

本项目是Gentry矩阵乘法（见 https://eprint.iacr.org/2025/1935）的Python实现。

## 项目特点

- 多项式模理想采用 $Y^n+i$（而非 $Y^n-i$），简化了 $\circledast$ 运算。
- 支持直接在NTT形式的多项式上进行 $\circledast$ 运算。
- Key Switch 过程使用更大质数模数 $Q$，而非 $q\cdot q_o$。
- 大整数多项式采用 `ndarray[dtype=object]` 实现，仅用于算法验证/展示，性能较低。

## 主要模块

- `poly.py`：多项式及其运算（加法、乘法、NTT变换等）。
- `matrix.py`：矩阵相关操作，包括Gentry矩阵乘法。
- `key_switch.py`：密钥切换相关算法。
- `params.py`：参数设置与管理。
- `test/`：测试脚本，覆盖主要功能。

## 依赖

- numpy
- sympy


## 运行方法

在有 numpy 和 sympy 的环境下，运行：

```bash
python test/test_all.py
```

即可进行全部测试。


## 参考

- Gentry矩阵乘法论文：https://eprint.iacr.org/2025/1935

## 备注

本项目仅用于算法验证和学习交流，未做性能优化或安全加固。