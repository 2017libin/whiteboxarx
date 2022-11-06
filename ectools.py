from sympy.ntheory.modular import isprime
import math
import random
import gmpy2

a = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC
b = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
p = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
n = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
Gx = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
Gy = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5

class point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __eq__(self, o):
        return self.x == o.x and self.y == o.y

    def isO(self):
        return self.x == 0 and self.y == 0

    def __str__(self):
        res = str(hex(self.x)) + ' ' + str(hex(self.y))
        return res

    def __add__(self, other):
        return pointadd(self, other)
O = point(0, 0)
G = point(Gx, Gy)

def inv(x, p):
    x %= p
    # assert isprime(p)
    return gmpy2.invert(x, p)
    # return pow(x, -1, p)

def ispointin(G: point):
    left = G.y * G.y
    right = G.x ** 3 + a * G.x + b
    return left % p == right % p

# [5] 点加运算， 参考《现代密码学》
def lmbd(P: point, Q: point):
    x1, x2 = P.x, Q.x
    y1, y2 = P.y, Q.y
    if P == Q:
        return (3 * x1 * x1 + a) * inv(2 * y1, p) % p
    return (y2 - y1) * inv(x2 - x1, p) % p


def pointadd(P: point, Q: point):
    if P.isO(): return Q
    if Q.isO(): return P
    if P.x == Q.x and (P.y + Q.y) % p == 0: return O
    x1, x2 = P.x, Q.x
    y1, y2 = P.y, Q.y
    ld = lmbd(P, Q)
    x = (ld * ld - x1 - x2) % p
    y = (ld * (x1 - x) - y1) % p
    return point(x, y)

# [6] 倍点运算
def kpoint(k: int, G: point):
    c = point(0, 0)
    while k:
        if k & 1: c = pointadd(c, G)
        G = pointadd(G, G)
        k >>= 1
    return c

def rand_prime(A, B):
    p = 4
    while not isprime(p):
        p = random.randint(A, B)
    return p

# [8] 生成用于coecdsa的大素数对p1和p2
def p1p2(n):
    n3 = n * n * n
    n4 = n3 * n
    p1, p2 = 0, 0
    while True:
        p1 = rand_prime(n3, n4)
        p2 = rand_prime(n3, n4)
        if math.gcd(p1 * p2, (p1 - 1) * (p2 - 1)) == 1:
            break
    # p1 = 0xb90ced3488a0dfa9b21532ec0b4e7dfa37d4f511b3c4e3202eef71e715f2610acf4c54039594de28e212890f9237adb69e6bffe043ffe5c8fcf05fc53b2daf68fb8f912e9ce44641105f2819ba85df2531e6b50225f11e65965c9f97dc380bcd68887ea267b9644da32b3e859f69046ec7288b88b7e46765d6d9d0e1e9882c7f
    # p2 = 0x29007560b08bcb5586f4d478b9dc1b8040ef89331f64e05e06bd45dd824b8c4026d575168dce5d431e7f7544613ec7ba8101f9bb47929df8ed212477b53df3dfb1ff481c3656f3e1fbb44a79fa6f5405cb64811f9c89581f5af0f573d5aefe54e7fc0ed39b87127278c53108aaf6a13c9fa6d57f87e3845506dfe85b025e610b
    return [p1, p2]

# [9] 密钥分片
def lcm(a, b):
    return a // math.gcd(a, b) * b

def L_fun(a, b, c, d):
    return (pow(a, b, c) - 1) // d