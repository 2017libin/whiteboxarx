R.<x,y,z, t> = PolynomialRing(GF(7))
f1 = t*x^2+x*y+z^3+z
f2 = x^2+x*y+3*z^2*t+z
print(f1.parent())
f1 = f1.subs(x=3)
f1 = f1.subs(y=2)
f2 = f2.subs(x=1)
f2 = f2.subs(y=2)
F = Sequence([f1,f2])
print(type(F))
print(F.reduced())