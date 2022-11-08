import sage.all
from ectools import *
import collections
Affine = collections.namedtuple('Affine', ['matrix', 'vector', 'size', 'inverse'])

def random_affines(size, field, number):
    # field = sage.all.GF(modulu)
    vs = sage.all.VectorSpace(field, size)

    _TRIVIAL = False
    def _get_affine_encodings():
        if not _TRIVIAL:
            while True:
                matrix = sage.all.matrix(field, size, entries=[vs.random_element() for _ in range(size)])
                if matrix.is_invertible():
                    break
            vector = sage.all.vector(field, list(vs.random_element()))
        else:
            matrix = sage.all.identity_matrix(field, size)
            vector = sage.all.vector(field, [0 for _ in range(size)])
        inverse_matrix = matrix.inverse()
        inverse_affine_encoding = Affine(
            inverse_matrix,
            inverse_matrix * vector,
            size,
            inverse=None
        )
        affine_encoding = Affine(
            matrix,
            vector,
            size,
            inverse=inverse_affine_encoding
        )
        return affine_encoding

    affine_encodings = []
    for _ in range(number):
        affine_encodings.append(_get_affine_encodings())
    return affine_encodings

def affine_encode(affine:Affine, vars):
    # field = sage.all.GF(13)
    # size = 3
    # polyr = sage.all.PolynomialRing(field, size, "z")
    # z = polyr.gens()
    # # x,y,z = sage.all.var('x'), sage.all.var('y'), sage.all.var('y'),
    # # vs = sage.all.VectorSpace(field, size)
    #
    # affs = random_affines(size, field, 2)
    ret = []
    # print(affine[0].matrix)
    # print(affine[0].vector)
    for i, row in enumerate(affine.matrix):
        pr = affine.vector[i]
        for j in range(affine.size):
            pr += row[j] * vars[j]
        ret.append(pr)
    # print(len(ret))
    return ret

def ecdsa_verify(r, s, e, Q):
    if not (1 <= r <= n - 1):
        return False
    if not (1 <= s <= n - 1):
        return False
    #
    # dg = eval('0x' + hashlib.sha256(m).digest().hex())
    # e = dg % n

    w = inv(s, n)
    u1 = e * w % n
    u2 = r * w % n

    X = pointadd(kpoint(u1, G), kpoint(u2, Q))

    if X == O:
        return False

    v = X.x % n

    return v == r


def ecdsa_sign(e, d):
    k = random.randint(0, n)
    R = kpoint(k, G)
    r = R.x % n
    s = inv(k, n) * (e + r * d) % n
    return r, s


def sample_ecdsa_sign_verify():
    d = 123
    Q = kpoint(d, G)

    e = 1234
    r, s = ecdsa_sign(e, d)
    print(ecdsa_verify(r, s, e, Q))

def sign_e1(e):
    u1, u2, u3, u4, u5 = 0, 0, 0, 0, e & 1
    for i in range(1, 256):
        tmp = point(u1, u2) + kpoint((1 - u5), Gi0_list[i])
        u1, u2 = tmp.x, tmp.y
        u3 = u3 + (1 - u5) * ki0_list[i] + u5 * ki1_list[i]
        u4 = u4 + u5 * 2 ** (i-1)  # u4 + u5*2^i
        u5 = (e >> i) & 1  # e_{i+1}
    return u1, u2, u3, u4

def sign_e2():
    pass

def solve_equation(prs, fixed_vars, all_vars):
    eqs = []
    vars_num = len(all_vars)

    for i in range(vars_num):
        eqs.append(prs[i]==0)

    return sage.all.solve(eqs+fixed_vars, all_vars)


def test_affine():
    field = sage.all.GF(13)
    size = 3
    polyr = sage.all.PolynomialRing(field, size, "z")
    z = polyr.gens()

    # x,y,z = sage.all.var('x'), sage.all.var('y'), sage.all.var('y'),
    # vs = sage.all.VectorSpace(field, size)

    affs = random_affines(size, field, 2)
    z = affine_encode(affs[0], z)

    pr = z[0]**2 + z[1] + 3*z[2]

    fixed_vars = []
    fixed_vars.append(z[0]==3)
    fixed_vars.append(z[1]==5)
    print(solve_equation(pr, fixed_vars, z))



    # ret = []
    # print(affs[0].matrix)
    # print(affs[0].vector)
    # for i, row in enumerate(affs[0].matrix):
    #     pr = affs[0].vector[i]
    #     for j in range(size):
    #         pr += row[j]*z[j]
    # ret.append(pr)

    # return ret
        # print(row, row[0],row[1],row[2])
        # print(pr)

    # x = affs[0].vector

    # print(affine_encode(affs[0], vec1))
if __name__ == '__main__':

    # ki0_list = []
    # ki1_list = []
    # Gi0_list = []
    # Gi1_list = []
    #
    # # 预计算参数
    # for i in range(256):
    #     k0 = random.randint(0, n / 256)
    #     k1 = random.randint(0, n / 256)
    #     ki0_list.append(k0)
    #     ki1_list.append(k1)
    #     Gi0_list.append(kpoint(k0, G))
    #     Gi1_list.append(kpoint(k1, G))
    #
    # u1,u2,u3,u4 = sign_e1(123)
    # print(u4)

    test_affine()

    # affs = random_affines(2, 7, 2)
    # print(affs[0].matrix)
    # print(affs[0].vector)
    # x = affs[0].vector
    # print(affine_encode(affs[0], x))