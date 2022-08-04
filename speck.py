"""Script to generate the implicit (unencoded) and explicit affine layers of a Speck instance for a fixed key."""
from collections import namedtuple
from functools import partial

import os

import sage.all

from boolcrypt.utilities import (
    substitute_variables, BooleanPolynomialRing, vector2int,
    int2vector, compose_affine, matrix2anf, compose_anf_fast, get_smart_print
)

from boolcrypt.modularaddition import get_implicit_modadd_anf

from argparse import ArgumentParser

# ws：一半的分组长度

SpeckInstance = namedtuple('SpeckInstance', 'name, default_rounds, ws, m, alpha, beta')

speck_instances = {
    8: SpeckInstance("Speck_8_16", 4, 4, 4, 2, 1),  # non-standard
    32: SpeckInstance("Speck_32_64", 22, 16, 4, 7, 2),
    64: SpeckInstance("Speck_64_128", 27, 32, 4, 8, 3),
    128: SpeckInstance("Speck_128_256", 34, 64, 4, 8, 3),
}


def get_round_keys(speck_instance, rounds, master_key):
    default_rounds = speck_instance.default_rounds
    n = speck_instance.ws
    m = speck_instance.m
    alpha = speck_instance.alpha
    beta = speck_instance.beta

    if rounds is None:
        rounds = default_rounds

    assert 1 <= rounds <= default_rounds

    def Constant(value, bitsize):
        assert 0 <= value <= 2**bitsize - 1
        return value

    def RotateLeft(val, r):
        width = n
        mask = 2 ** width - 1
        r = r % width
        return ((val << r) & mask) | ((val & mask) >> (width - r))

    def RotateRight(val, r):
        width = n
        mask = 2 ** width - 1
        r = r % width
        return ((val & mask) >> r) | (val << (width - r) & mask)

    def BvAdd(x, y):
        return (x + y) % (2 ** n)

    def rf(x, y, k):
        x = BvAdd(RotateRight(x, alpha), y) ^ k
        y = RotateLeft(y, beta) ^ x
        return x, y

    round_keys = [None for _ in range(rounds)]
    round_keys[0] = master_key[-1]
    l_values = list(reversed(master_key[:-1]))
    l_values.append(None)
    for i in range(rounds - 1):
        result = rf(l_values[i % len(l_values)], round_keys[i], Constant(i, n))
        l_values[(i + m - 1) % len(l_values)], round_keys[i + 1] = result

    return round_keys


def get_implicit_unencoded_affine_layers(
        speck_instance, rounds, master_key, only_x_names=False,
        return_also_explicit_affine_layers=False,
        return_implicit_round_functions=False  # only needed for debugging
):
    n = speck_instance.ws
    alpha = speck_instance.alpha
    beta = speck_instance.beta

    assert rounds is not None
    ws = n
    bpr = sage.all.GF(2)

    identity_matrix = partial(sage.all.identity_matrix, bpr)  # identity_matrix(n): 表示n*n的恒等矩阵
    zero_matrix = partial(sage.all.zero_matrix, bpr)  # zero_matrix(n, m) 表示n*m的0矩阵

    # ra: r表示right、a表示alpha，ra表示循环右移alpha位
    ra = sage.all.block_matrix(bpr, 2, 2, [
        [zero_matrix(ws - alpha, alpha), identity_matrix(ws - alpha)],
        [identity_matrix(alpha), zero_matrix(alpha, ws - alpha)]])
    # lb: 表示循环左移beta位
    lb = sage.all.block_matrix(bpr, 2, 2, [
        [zero_matrix(beta, ws - beta), identity_matrix(beta)],
        [identity_matrix(ws - beta), zero_matrix(ws - beta, beta)]])
    assert ra.is_square() and lb.is_square()

    ii = identity_matrix(ws)  # 16*16
    zz = zero_matrix(ws, ws)  # 16*16

    # 以下的x、y均表示长度位ws的比特向量，矩阵均为2ws的方阵
    # identity_rotateleft_matrix * [x,y] = [x, y<<<beta]
    identity_rotateleft_matrix = sage.all.block_matrix(bpr, 2, 2, [
        [ii, zz],
        [zz, lb]])
    # rotateright_identity_matrix * [x,y] = [x>>alpha, y]
    rotateright_identity_matrix = sage.all.block_matrix(bpr, 2, 2, [
        [ra, zz],
        [zz, ii]])
    # identity_xor_matrix * [x,y] = [x, x*y]
    identity_xor_matrix = sage.all.block_matrix(bpr, 2, 2, [
        [ii, zz],
        [ii, ii]])
    
    def bitvectors_to_gf2vector(x, y):
        return sage.all.vector(bpr, list(int2vector(x, ws)) + list(int2vector(y, ws)))

    # 生成轮密钥
    round_keys = get_round_keys(speck_instance, rounds, master_key)

    for i in range(len(round_keys)):
        round_keys[i] = bitvectors_to_gf2vector(round_keys[i], 0)

    #  获取隐式表达式的组件 x + y (mod 2^ws) = z
    implicit_pmodadd = get_implicit_modadd_anf(ws, permuted=True, only_x_names=only_x_names)  # if permuted=Ture, 返回(x,y,z)和 y^t 的 bool_poly, (x,y,z)的bool_poly等于0当且仅当x+y=z时
    bpr_pmodadd = implicit_pmodadd[0].parent()
    bpr_pmodadd = BooleanPolynomialRing(names=bpr_pmodadd.variable_names(), order="deglex")
    implicit_pmodadd = [bpr_pmodadd(str(f)) for f in implicit_pmodadd]  # 转换为列表的形式

    # compose_affine(L_m, L_t, R_m, R_t)，返回(M,t)满足L_m(R_m*x+R_t)+L_t = M*x + t
    # [x, y] = [x>>>alpha, x^(y<<<beta)]
    aux_linear_layer = compose_affine(
        rotateright_identity_matrix, 0,  
        *(compose_affine(identity_xor_matrix, 0, identity_rotateleft_matrix, 0))  # [x,y] = [x, x^(y<<<beta)]
    )[0] 

    # 
    implicit_round_functions = []
    explicit_affine_layers = []
    for i in range(rounds):
        # 生成 R-3 rounds 的仿射层
        if i not in [rounds - 2, rounds - 1]:
            # round function is S \circ affine  E  = S \circ AL
            # 1. affine = compose_affine(identity_rotateleft_matrix, 0, identity_matrix(2*ws), round_keys[i])
            # 2. affine = compose_affine(identity_xor_matrix, 0, affine[0], affine[1])
            # 3. affine = compose_affine(rotateright_identity_matrix, 0, affine[0], affine[1])
            affine = compose_affine(aux_linear_layer, 0, identity_matrix(2*ws), round_keys[i])  # 相当于上面3步
            
            # 生成隐式的affine: (matrix*[x x]) ^ cta = [affine[x], x]，[x, x] 和 [affine[x], x]均为列向量，x表示一个分组
            matrix = sage.all.block_matrix(bpr, 2, 2, [  # 2n*2n matrix，n是分组长度
                [affine[0], zero_matrix(2*ws, 2*ws)],
                [zero_matrix(2*ws, 2*ws), identity_matrix(2*ws)]]) # 对角矩阵
            cta = list(affine[1]) + [0 for _ in range(2*ws)]  # 将affine[1] 转换为列表形式
            
            # anf就是boo_poly的集合，其中每个boo_poly表示结果的每个bit
            anf = matrix2anf(matrix, bool_poly_ring=bpr_pmodadd, bin_vector=cta)
            
            if not return_implicit_round_functions:
                implicit_round_functions.append(anf)  # return_implicit_round_functions 为 false, 只返回隐式的仿射层 (y,x)
            else:
                implicit_round_functions.append(compose_anf_fast(implicit_pmodadd, anf))
            if return_also_explicit_affine_layers:
                explicit_affine_layers.append(affine)
        
        elif i == rounds - 2:
            # round function is explicit_affine_layers[-1][1] \circ S \circ explicit_affine_layers[-1][0]
            # affine = compose_affine(identity_rotateleft_matrix, 0, identity_matrix(2*ws), round_keys[i])
            # affine = compose_affine(identity_xor_matrix, 0, affine[0], affine[1])
            # affine = compose_affine(rotateright_identity_matrix, 0, affine[0], affine[1])
            affine = compose_affine(aux_linear_layer, 0, identity_matrix(2*ws), round_keys[i])  # 倒数第二轮affine
            matrix = sage.all.block_matrix(bpr, 2, 2, [
                [affine[0], zero_matrix(2*ws, 2*ws)],
                [zero_matrix(2*ws, 2*ws), identity_matrix(2*ws)]])
            cta = list(affine[1]) + [0 for _ in range(2*ws)]
            anf1 = matrix2anf(matrix, bool_poly_ring=bpr_pmodadd, bin_vector=cta)

            if return_also_explicit_affine_layers:
                explicit_affine_layers.append([affine]) 

            # A(x)          = L(x) + c
            # A^(-1)(x)     = L^(-1)(x) + L^(-1)(c)
            # A^(-1)(A(x))  = L^(-1)(L(x) + c) + L^(-1)(c) = x
            affine = compose_affine(identity_rotateleft_matrix, 0, identity_matrix(2*ws), round_keys[i+1])  # 最后一轮affine, 少了round_right_shift
            affine = compose_affine(identity_xor_matrix, 0, affine[0], affine[1])
            aux = affine[0] ** (-1)  # affine[0](x)=L(x), aux(x)=L^(-1)(x)
            matrix = sage.all.block_matrix(bpr, 2, 2, [
                [identity_matrix(2*ws), zero_matrix(2*ws, 2*ws)],
                [zero_matrix(2*ws, 2*ws), aux]])
            cta = [0 for _ in range(2*ws)] + list(aux * affine[1])
            anf2 = matrix2anf(matrix, bool_poly_ring=bpr_pmodadd, bin_vector=cta)  # anf2(x,x) = (x, A^(-1)(x))

            anf = compose_anf_fast(anf1, anf2)  # anf(x,x) = (A_{R-2}(x), A_{R-1}^(-1)(x))

            if not return_implicit_round_functions:
                implicit_round_functions.append(anf)
            else:
                implicit_round_functions.append(compose_anf_fast(implicit_pmodadd, anf))
            if return_also_explicit_affine_layers:
                explicit_affine_layers[-1].append(affine)
        else:
            continue

    if return_also_explicit_affine_layers:
        return implicit_round_functions, explicit_affine_layers
    else:
        return implicit_round_functions


def bitvectors_to_gf2vector(x, y, ws):
    return sage.all.vector(sage.all.GF(2), list(int2vector(x, ws)) + list(int2vector(y, ws)))


def gf2vector_to_bitvectors(v, ws):
    return vector2int(v[:ws]), vector2int(v[ws:])


def get_first_and_last_explicit_rounds(speck_instance, print_intermediate_values, filename=None):
    n = ws = speck_instance.ws
    alpha = speck_instance.alpha

    smart_print = get_smart_print(filename)

    def RotateRight_Identity(val, right_operand):
        r, width = alpha, n
        mask = 2 ** width - 1
        r = r % width
        return ((val & mask) >> r) | (val << (width - r) & mask), right_operand

    def PermutedBvAdd(x, y):
        return (x + y) % (2 ** n), y

    def first_explicit_round(v):
        x, y = gf2vector_to_bitvectors(v, ws)
        if print_intermediate_values:
            smart_print(f"\nplaintext:\n - ({hex(x)}, {hex(y)}) = {bitvectors_to_gf2vector(x, y, ws)}")
        x, y = RotateRight_Identity(x, y)
        x, y = PermutedBvAdd(x, y)
        v = bitvectors_to_gf2vector(x, y, ws)
        if print_intermediate_values:
            smart_print(f"\nRotateRight_Identity and PermutedBvAdd:\n - output | ({hex(x)}, {hex(y)}) = {v}")
            smart_print("")
        return v

    return first_explicit_round, None


if __name__ == '__main__':
    parser = ArgumentParser(prog="sage -python speck.py", description="Generate the implicit (unencoded) and explicit affine layers of a Speck instance for a fixed key")
    parser.add_argument("--key", nargs="+", help="the master key given as a hexadecimal representation of the words")
    parser.add_argument("--block-size", nargs="?", type=int, choices=[8, 32, 64, 128], help="the block size in bits of the Speck instance")
    parser.add_argument("--output-file", nargs="?", help="the file to store the implicit (unencoded) and explicit affine layers")

    args = parser.parse_args()

    # 判断输出文件是否存在
    # assert not os.path.isfile(args.output_file), f"{args.output_file} already exists"
    
    assert len(args.key) == 4, "key should be 4 words"  # 1 words = 2 bytes
    master_key = tuple(map(lambda k: int(k, 16), args.key))  # 把十六进制words转为十进制, master_key = (6424, 4368, 2312, 256)
    
    speck_instance = speck_instances[args.block_size]
    rounds = speck_instance.default_rounds

    # 返回的是轮函数的隐函数、还有轮函数中的仿射函数
    # 这里的implicit_affine_layers 可能造成歧义
    implicit_affine_layers, explicit_affine_layers = get_implicit_unencoded_affine_layers(speck_instance, rounds, master_key, return_also_explicit_affine_layers=True)
    for i, affine_layer in enumerate(implicit_affine_layers):
        # Wrap in tuple because BooleanPolynomialVector can't be pickled.
        implicit_affine_layers[i] = tuple(affine_layer)
    
    exit(1)
    sage.all.save((implicit_affine_layers, explicit_affine_layers), args.output_file, compress=True)  # 保存对象
