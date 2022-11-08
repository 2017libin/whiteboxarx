import sage.all
from sage.all import *
from ectools import *
import collections
Affine = collections.namedtuple('Affine', ['matrix', 'vector', 'size', 'inverse'])
p = 13

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

# 替换方程组中变量的值
def eqs_subs(eqs, fixed_vars):
    try:
        new_eqs = []
        for eq in eqs:
            new_eqs.append(eq.subs(fixed_vars))
        return new_eqs
    except:
        return eqs

# 对方程组进行规约
def eqs_reduce(eqs):
    try:
        return list(Sequence(eqs).reduced())
    except:
        return eqs

def _eq_solve(eq, pring):
    try:
        # print(eq, pring)
        roots = pring(str(eq)).roots()
        # print(roots)
        return roots[0][0]
    except:
        return None

# 对方程组进行求解
def eqs_solve(eqs, fixed_vars, pring):
    ans = {}
    while len(eqs) > 0:
        eqs = eqs_subs(eqs, fixed_vars)  # 替换变量
        # print(type(eqs))
        # print(f'after subs: eqs = {eqs}')
        eqs = eqs_reduce(eqs)  # 化简方程组
        # print(type(eqs))
        # print(eqs)
        # print(f'after reduce: eqs = {_eq_solve(eqs[0], pring)}')
        find = False
        for index, eq in enumerate(eqs):
            for name in eq.parent().variable_names():
                try:
                    # print(f'name = {name}')
                    pring = PolynomialRing(GF(p), names=name)
                    val =  _eq_solve(eq, pring)
                    # print(f'val = {val}')
                    if val:
                        fixed_vars[name] = val
                        ans[name] = val
                        # print(eqs)
                        eqs.pop(index)
                        # print(eqs)
                        find = True
                except:
                    continue
            if find:
                break

    print(ans)

def test_affine():
    field = sage.all.GF(p)
    size = 4

    pring = sage.all.PolynomialRing(field, size, "z")
    z = pring.gens()

    # x,y,z = sage.all.var('x'), sage.all.var('y'), sage.all.var('y'),
    # vs = sage.all.VectorSpace(field, size)
    affs = random_affines(size, field, 2)
    new_z = affine_encode(affs[0], z)

    pr1 = z[0] + 5*z[1] + 3*z[2]*z[3]
    pr2 = z[0]**2 + z[1] + 4*z[2]

    fixed_vars = {}
    fixed_vars[z[0]] = 3
    fixed_vars[z[1]] = 5
    eqs = [pr1, pr2]
    # print(eqs)
    # print(fixed_vars)
    # print(eqs)
    # print(eqs_reduce(eqs))

    eqs_solve(eqs, fixed_vars, PolynomialRing(field, name='z'))
    # solve_equation(eqs, fixed_vars)
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

def test_sovle_multipolynomials():
    x = PolynomialRing(GF(7), 'x').gen()
    f = x - 1
    root = f.roots()  # 返回的是根和根的次数
    print(root)

    R = PolynomialRing(QQ, 2, 'ab')
    a, b = R.gens()
    I = (a**2 - b**2 - 3, a - 2 * b) * R
    B = I.groebner_basis()
    print(B)

if __name__ == '__main__':
    # test_sovle_multipolynomials()
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