"""
    Implementation of elliptic curve operations to be used for Schnorr identification 
    Resources:
    - https://karpathy.github.io/2021/06/21/blockchain/
    - https://andrea.corbellini.name/2015/05/23/elliptic-curve-cryptography-finite-fields-and-discrete-logarithms/
"""

from dataclasses import dataclass

def _extended_euclid(a,b):
    if b == 0:
        return 1,0,a
    else:
        xp,yp,g = _extended_euclid(b,a%b)
        return yp, xp-(a//b)*yp, g

def _inverse(n,p):
    x, y,gcd = _extended_euclid(n,p)
    assert (n*x +p*y) % p == gcd
    if gcd == 1:
        # p is prime and n != 0
        return x % p
    else:
        raise ValueError(f"{n} has no multiplicative inverse mod {p}")

class Point:
    def __init__(self, curve, x,y) -> None:
        self.curve = curve
        self.x = x
        self.y = y

    def is_infinity(self):
        return self.x is None and self.y is None

    def __add__(self, other):
        if self.is_infinity(): 
            # INF + other = other
            return other
        if other.is_infinity():
            # self + INF = self
            return self

        if self.x == other.x and self.y != other.y:
            # perpendicular line meets the third point at inifnity 
            return self.curve.INF

        if self.x == other.x: # self.y == other.y
            # self = other, tangent (derivative of weierstrass representation)
            m = (3 * self.x**2 + self.curve.a)*_inverse(2*self.y,
                                                    self.curve.p)
        else:
            # classic tangent
            m = (self.y - other.y) * _inverse(self.x - other.x,
                                              self.curve.p)
        x = (m**2-self.x - other.x) % self.curve.p
        y = ((m*(x-self.x)) + self.y) % self.curve.p
        return Point(self.curve, x,-1*y)

    def __rmul__(self, k:int):
        # Double and add (in O(logn) instead of O(n))
        assert isinstance(k, int) and k >= 0
        result = self.curve.INF
        append = self
        while k:
            if k & 1: 
                result = result + append
            append = append + append
            k >>= 1
        return result

    def __repr__(self) -> str:
            if self.is_infinity():
                return "Point(INFINITY)"
            return f"Point({hex(self.x)}, {hex(self.y)})"

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False  # Ensure we only compare with another Point
    
        if self.is_infinity() and other.is_infinity():
            return True  # Both are the point at infinity
    
        return self.x == other.x and self.y == other.y and self.curve == other.curve

@dataclass
class Generator:
    G:Point
    h:int
    n:int

class EllipticCurve:
    def __init__(self,p,a,b):
        self.p = p
        self.a = a
        self.b = b
        self.G = "Not defined"
        self.INF = Point(self, None, None)

    def __call__(self, point:Point):
        return (point.y**2 - point.x ** 3 - self.a * point.x - self.b ) % self.p

    def set_generator(self, generator:Generator):
        self.generator = generator


# if __name__=="__main__":
#     curve = EllipticCurve(
#                 p= 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F,
#                 a= 0x0000000000000000000000000000000000000000000000000000000000000000, # a = 0
#                 b= 0x0000000000000000000000000000000000000000000000000000000000000007 # b = 7
#             )
#     x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
#     y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
#     G = Point(curve, x, y)
#     n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
#     h = 0x01
#     generator = Generator(G=G,h=h, n=n)
#     curve.set_generator(generator)
#     print(curve(G))
#
#     print(curve(Point(curve, 1032, 208417)))
#
#     pk = G + G + G
#     print(pk)
#     print(curve(pk))
#
#     pk = 3*G
#     print(pk)
#     print(curve(pk))
#
#
#     x, y, gcd = _extended_euclid(104, 48)
#     assert 104*x + 48*y == gcd
#
#     a = 1042910374
#     b = 4812371948
#     x, y, gcd = _extended_euclid(a,b)
#     assert a*x + b*y == gcd
#
#     n = 5
#     p = 13
#     i = _inverse(n,p)
#     assert (n * i)%p == 1 
#
#     n = 123481238419
#     p = 2**256 - 2**32 - 2**9 - 2**8 - 2**7 - 2**6 - 2**4 - 1
#     i = _inverse(n,p)
#     assert (n * i)%p == 1 
