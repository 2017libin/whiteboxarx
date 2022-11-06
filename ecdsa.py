from ectools import *


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


ki0_list = []
ki1_list = []
Gi0_list = []
Gi1_list = []


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


if __name__ == '__main__':

    # 预计算参数
    for i in range(256):
        k0 = random.randint(0, n / 256)
        k1 = random.randint(0, n / 256)
        ki0_list.append(k0)
        ki1_list.append(k1)
        Gi0_list.append(kpoint(k0, G))
        Gi1_list.append(kpoint(k1, G))

    u1,u2,u3,u4 = sign_e1(123)
    print(u4)