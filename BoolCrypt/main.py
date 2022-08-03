from boolcrypt.utilities import *
from boolcrypt.equivalence import *
from sage.all import *
import sboxU
import sage.all

# 获取自等价
def sample_se():
    # 获取线性的自等价编码
    all_linear_se = get_all_self_le([0, 2, 1, 3], return_matrices=True)  # 获取 [0,1,2,3]的所有线性自等价
    # print(all_linear_se[-1])
    # for linear_se in all_linear_se:
    #     print(linear_se[0], linear_se[1])

    # 获取仿射的自等价编码
    all_affine_se = get_all_self_ae([0, 2, 1, 3], return_lut=False)  # 获取 [0,1,2,3]的所有仿射自等价
    print(all_affine_se[-4])
    # for affine_se in all_affine_se:
    #     print(affine_se[0], affine_se[1])

if __name__ == "__main__":
    # sample_se()
    # x = PolynomialRing(GF(2**4), 'x').gen()
    s = "0231"
    lut = hex_string2lut(s, 1)
    mat = lut2matrix(lut, return_ct=True)
    # lut_inv = get_lut_inversion(5)
    poly = lut2poly(lut)  # lut -> mat
    print(lut)
    print(poly)
    print(mat)

    s1 = "0213"
    lut1 = hex_string2lut(s1, 1)
    mat1 = lut2matrix(lut1, return_ct=True)
    poly1 = lut2poly(lut1)  # lut -> mat
    print(poly1)
    print(poly*poly1)
    # x = PolynomialRing(GF(2**4), 'x').gen()
    # mat2 = lut2matrix(poly2lut(x), return_ct=True)
    # print(mat2)

