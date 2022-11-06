import sys
sys.path.append('./BoolCrypt')
from sage.all import *
from sage.crypto.sbox import SBox
from boolcrypt.utilities import (
    BooleanPolynomialRing, vector2int, int2vector, compose_affine, matrix2anf, compose_anf_fast,
    get_smart_print, get_anf_coeffmatrix_str, lut2anf,get_symbolic_anf, anf2matrix)
from boolcrypt.functionalequations import reduce, find_fixed_vars, solve_functional_equation
from boolcrypt.modularaddition import get_implicit_modadd_anf, get_modadd_anf
from collections import OrderedDict
def test_find_fixed_vars():
    sbox3b = SBox((3, 2, 0, 1))  # one linear component
    anf_l1 = list(lut2anf(list(sbox3b)))
    print(anf_l1)

    bpr = BooleanPolynomialRing(4, "y0, y1, x0, x1", order="lex")  # 定义一个有6个变量的布尔多项式环
    input_vars, output_vars = list(reversed(bpr.gens()[2:])), list(reversed(bpr.gens()[:2]))  # 定义输入变量和输出变量
    eqs = [bpr(f) for f in sbox3b.polynomials(X=input_vars, Y=output_vars)]  # 输出的多项式都等于0，如果y == sbox(x)
    print(f"before reduce: size is {len(eqs)}\n", eqs)
    eqs = reduce(eqs, mode="groebner", bpr=bpr)
    print(f"before reduce: size is {len(eqs)}\n", list(eqs))

    fixed_vars, new_eqs = find_fixed_vars(eqs, only_linear=False, verbose=True, debug=True) 
    print(f"{fixed_vars}\n{list(new_eqs)}")

def test_solve_functional_equation():
    # example 1: finding the inverse of F0 by solving F1(F0) = Identity
    f0 = lut2anf((1, 3, 2, 0))
    f1 = get_symbolic_anf(2, 2, 2)
    # print(f"f0:\n {list(f0)}\nf1:\n {list(f1)}")
    g0 = lut2anf(list(range(2**2)))  # identity
    input_vars = ["x0", "x1"]
    list_anfs = solve_functional_equation([f0, f1], [g0], [input_vars, input_vars], [input_vars],
        num_sat_solutions=1, return_mode="list_anfs", verbose=False)
    # print(list(list_anfs[0][0][1]))
    tmp = get_anf_coeffmatrix_str(list_anfs[0][0][1], input_vars=input_vars)  # f1
    # print(tmp)

    # x0*x1*a0_0_1 + x0*a0_0 + x1*a0_1 + a0, 
    # x0*x1*a1_0_1 + x0*a1_0 + x1*a1_1 + a1
    # [('a1', 1), ('a1_1', 0), ('a1_0', 1), ('a1_0_1', 0), ('a0', 1), ('a0_1', 1), ('a0_0', 1), ('a0_0_1', 0)]
    # x0*x1*a0_0_1 + x0*a0_0 + x1*a0_1 + a0 = x0 + x1 + 1 
    # x0*x1*a1_0_1 + x0*a1_0 + x1*a1_1 + a1 = x0 + 1

    # sample 2
    f0 = get_symbolic_anf(1, 2, 2, ct_terms=False)
    f1 = lut2anf([0, 2, 1, 3])  # (x, y) to (y, x)
    print(f"f0:\n{list(f0)}\nf1:\n{list(f1)}")

    input_vars = ["x0", "x1"]
    init_eq = ["a0_0", anf2matrix(f0, input_vars).determinant() + 1]
    # 因为这里的G是确定的值，因此不用指定多项式的变量名称
    # num_sat_solutions=None 表示找到所有的解
    # return_mode="symbolic_anf" 返回四元组 (symF, symG, eqs, num_sols)，其中symF表示的是F的符号化的anf，symG类似，eqs表示的是化简后的方程，num_sols表示的是解的个数
    # initial_equations=init_eq 可以作为额外的方程组添加到系统中（相当于除了满足F+G=0以外的限制条件）
    # 输入F、G是以笛卡尔积的方式进行匹配来构造方程组。例如，输入F1,F2,G，最后系统的方程组有F1+G=0, F2+G=0
    # reduction_mode 表示使用SAT对方程进行求解前，对方程组进行规约的方法，可选值为(None, “gauss” or “groebner”) 
    result = solve_functional_equation([f0, f1], [1, 1], [input_vars, input_vars], None, \
        num_sat_solutions=None, return_mode="symbolic_anf", initial_equations=init_eq, reduction_mode=None, \
        verbose=True, debug=True)
    tmp = get_anf_coeffmatrix_str(result[0][0][0], input_vars=input_vars)  # f0
    print(list(result))


def test_implicit_pmodadd_unencoded():
    ws = 2

    implicit_pmodadd = get_implicit_modadd_anf(ws, permuted=False, only_x_names=False)

    eqs = list(implicit_pmodadd)
    for index, eq in enumerate(eqs):
        print(f'eq{index}: {eq}')

    # 构建system：指明输入/输出变量、表达式
    names = implicit_pmodadd[0].parent().variable_names()[::-1]  # 将z变量放到前面
    bpr_pmodadd = BooleanPolynomialRing(names=names, order="deglex")
    system = [bpr_pmodadd(str(f)) for f in eqs]  # 将BooleanPolynomialVector对象中的布尔多项式放到列表中

    # 默认使用后面的变量来表示前面的变量
    fixed_vars, new_equations = find_fixed_vars(system)

    # 打印固定的变量
    # 将原来的字典逆序排序
    fixed_vars = OrderedDict(sorted(fixed_vars.items(), key=lambda t: t[0],reverse = False))
    for var, value in fixed_vars.items():
        print(f'{var} = {value}')

    # 打印化简后的表达式
    for index, eq in enumerate(new_equations):
        print(f'new_eqP{index}: {eq}')

    modadd_anf = get_modadd_anf(ws, only_x_names=False)
    for index, anf in enumerate(modadd_anf):
        print(f'anf{index}: {anf}')


if __name__ == "__main__":
    test_implicit_pmodadd_unencoded()
    # test_solve_functional_equation()
    # [[[<sage.rings.polynomial.pbori.pbori.BooleanPolynomialVector object at 0x7f1171d4ff80>], [<sage.rings.polynomial.pbori.pbori.BooleanPolynomialVector object at 0x7f1171d4ffc0>]]]
    # [[[<sage.rings.polynomial.pbori.pbori.BooleanPolynomialVector object at 0x7f18fb429300>, <sage.rings.polynomial.pbori.pbori.BooleanPolynomialVector object at 0x7f18fb4296c0>], [<sage.rings.polynomial.pbori.pbori.BooleanPolynomialVector object at 0x7f18fb429740>]]]
    