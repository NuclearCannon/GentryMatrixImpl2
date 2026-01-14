import numpy as np
import random

def generate_hwt(shape, h):
    """
    生成HWT(h)分布的object类型numpy数组
    在全部的n个位置中有h个为1或者-1（具体哪一个则随机），其余为0
    """
    n = np.prod(shape)
    arr = np.zeros(n, dtype=object)
    
    # 随机选择h个位置
    indices = random.sample(range(n), h)
    
    # 为这h个位置随机分配1或-1
    for idx in indices:
        arr[idx] = random.choice([1, -1])
    
    return arr.reshape(shape)


def generate_uniform(shape, q):
    n = np.prod(shape)
    arr = np.zeros(n, dtype=object)
    # 为这h个位置随机分配1或-1
    for idx in range(n):
        arr[idx] = random.randint(0, q-1)
    return arr.reshape(shape)

def generate_sparse_ternary(shape, m):
    """
    生成SparseTernary(m)分布的object类型numpy数组
    在全部的n个位置中有m个1、m个-1和n-2m个0
    """
    n = np.prod(shape)
    if 2 * m > n:
        raise ValueError(f"m={m} is too large for shape {shape} with total elements {n}")
    
    arr = np.zeros(n, dtype=object)
    
    # 随机选择m个位置放置1
    indices_for_ones = random.sample(range(n), m)
    for idx in indices_for_ones:
        arr[idx] = 1
    
    # 从剩余位置中随机选择m个位置放置-1
    remaining_indices = [i for i in range(n) if i not in indices_for_ones]
    indices_for_neg_ones = random.sample(remaining_indices, m)
    for idx in indices_for_neg_ones:
        arr[idx] = -1
    
    return arr.reshape(shape)

def generate_discrete_gaussian(shape, sigma):
    """
    生成离散高斯分布的object类型numpy数组
    以0为均值，以sigma为标准差的离散高斯分布
    """
    n = np.prod(shape)
    
    # 生成标准正态分布的连续值，然后四舍五入到整数
    # TODO: 这样做是正当的吗？
    continuous_values = np.random.normal(0, sigma, n)
    discrete_values = np.round(continuous_values).astype(int)
    
    # 转换为object类型数组
    arr = np.empty(n, dtype=object)
    for i in range(n):
        arr[i] = int(discrete_values[i])
    
    return arr.reshape(shape)

def generate_zo(shape):
    """
    生成ZO分布的object类型numpy数组
    每个位置有1/4的概率取1，1/4的概率取-1，1/2的概率取0
    """
    n = np.prod(shape)
    
    arr = np.empty(n, dtype=object)
    for i in range(n):
        rand_val = random.random()
        if rand_val < 0.25:  # 1/4概率取1
            arr[i] = 1
        elif rand_val < 0.5:  # 1/4概率取-1
            arr[i] = -1
        else:  # 1/2概率取0
            arr[i] = 0
    
    return arr.reshape(shape)

# 示例使用
if __name__ == "__main__":
    # 测试HWT分布
    print("HWT分布 (shape=(3,4), h=5):")
    hwt_arr = generate_hwt((3, 4), 5)
    print(hwt_arr)
    print(f"非零元素个数: {np.count_nonzero(hwt_arr)}")
    print()
    
    # 测试SparseTernary分布
    print("SparseTernary分布 (shape=(3,4), m=3):")
    sparse_ternary_arr = generate_sparse_ternary((3, 4), 3)
    print(sparse_ternary_arr)
    print(f"1的个数: {np.sum(sparse_ternary_arr == 1)}")
    print(f"-1的个数: {np.sum(sparse_ternary_arr == -1)}")
    print(f"0的个数: {np.sum(sparse_ternary_arr == 0)}")
    print()
    
    # 测试离散高斯分布
    print("离散高斯分布 (shape=(3,4), sigma=2):")
    gaussian_arr = generate_discrete_gaussian((3, 4), 2)
    print(gaussian_arr)
    print()
    
    # 测试ZO分布
    print("ZO分布 (shape=(3,4)):")
    zo_arr = generate_zo((3, 4))
    print(zo_arr)
    print(f"1的个数: {np.sum(zo_arr == 1)}")
    print(f"-1的个数: {np.sum(zo_arr == -1)}")
    print(f"0的个数: {np.sum(zo_arr == 0)}")



