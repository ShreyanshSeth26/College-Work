from math import gcd
import random

# Applied Cryptography Assignment 4
# Question 1 Part A
# Helper code for computing n, phi(n), a valid e, and d

random.seed(2022484)

def ExtendedGcd(A, B):
    if B == 0:
        return A, 1, 0
    G, X1, Y1 = ExtendedGcd(B, A % B)
    X = Y1
    Y = X1 - (A // B) * Y1
    return G, X, Y

# Computes modular inverse of A modulo N
def ModInverse(A, N):
    G, X, _ = ExtendedGcd(A, N)
    if G != 1:
        raise ValueError("Inverse does not exist")
    return X % N

# Small hand-computation example from the assignment
P = 61
Q = 53
N = P * Q
Phi = (P - 1) * (Q - 1)
E = 17
D = ModInverse(E, Phi)

print("========== Q1 Part A Helper ==========")
print(f"P = {P}")
print(f"Q = {Q}")
print(f"N = P * Q = {N}")
print(f"Phi(N) = (P - 1) * (Q - 1) = {Phi}")
print(f"Chosen Public Exponent E = {E}")
print(f"Gcd(E, Phi(N)) = {gcd(E, Phi)}")
print(f"Private Exponent D = {D}")
print()
print("Verification:")
print(f"(E * D) % Phi(N) = {(E * D) % Phi}")
print("If this value is 1, then D is correct.")
print("======================================")