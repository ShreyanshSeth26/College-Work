import random
from math import gcd

# Applied Cryptography Assignment 4
# Question 1 Part C
# Common Modulus Attack on textbook RSA

random.seed(2022484)

def ExtendedGcd(A, B):
    if B == 0:
        return A, 1, 0
    G, X1, Y1 = ExtendedGcd(B, A % B)
    X = Y1
    Y = X1 - (A // B) * Y1
    return G, X, Y

# Computes modular inverse using Extended Euclid
def ModInverse(A, N):
    G, X, _ = ExtendedGcd(A, N)
    if G != 1:
        raise ValueError("Inverse does not exist")
    return X % N

# Miller-Rabin probabilistic primality test
def IsProbablePrime(N, Rounds=40):
    if N < 2:
        return False

    SmallPrimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for Prime in SmallPrimes:
        if N == Prime:
            return True
        if N % Prime == 0:
            return False

    D = N - 1
    S = 0
    while D % 2 == 0:
        D //= 2
        S += 1

    for _ in range(Rounds):
        A = random.randrange(2, N - 1)
        X = pow(A, D, N)

        if X == 1 or X == N - 1:
            continue

        Passed = False
        for _ in range(S - 1):
            X = pow(X, 2, N)
            if X == N - 1:
                Passed = True
                break

        if not Passed:
            return False

    return True

# Generates a probable prime of the requested bit length
def GeneratePrime(Bits):
    while True:
        Candidate = random.getrandbits(Bits)
        Candidate |= (1 << (Bits - 1))
        Candidate |= 1
        if IsProbablePrime(Candidate):
            return Candidate

# Generates one shared RSA modulus n = p*q
def GenerateSharedModulus(Bits):
    Half = Bits // 2
    while True:
        P = GeneratePrime(Half)
        Q = GeneratePrime(Bits - Half)
        if P == Q:
            continue
        N = P * Q
        Phi = (P - 1) * (Q - 1)
        return P, Q, N, Phi

# Chooses two valid public exponents that are coprime to each other
def ChooseExponents(Phi):
    Choices = [3, 5, 17, 257, 65537]
    Valid = [E for E in Choices if E < Phi and gcd(E, Phi) == 1]

    for I in range(len(Valid)):
        for J in range(I + 1, len(Valid)):
            if gcd(Valid[I], Valid[J]) == 1:
                return Valid[I], Valid[J]

    raise ValueError("Suitable exponents not found")

# Encodes ASCII text as an integer
def StringToInt(Text):
    return int.from_bytes(Text.encode("ascii"), byteorder="big")

# Decodes an integer back to ASCII text
def IntToString(Value):
    if Value == 0:
        return ""
    Length = (Value.bit_length() + 7) // 8
    return Value.to_bytes(Length, byteorder="big").decode("ascii")

# Recovers the plaintext using the common modulus attack
def CommonModulusAttack(CipherOne, CipherTwo, EOne, ETwo, N):
    G, S, T = ExtendedGcd(EOne, ETwo)
    if G != 1:
        raise ValueError("Exponents must be coprime")

    if S < 0:
        PartOne = pow(ModInverse(CipherOne, N), -S, N)
    else:
        PartOne = pow(CipherOne, S, N)

    if T < 0:
        PartTwo = pow(ModInverse(CipherTwo, N), -T, N)
    else:
        PartTwo = pow(CipherTwo, T, N)

    Message = (PartOne * PartTwo) % N
    return Message, S, T

Bits = 512
MessageText = "HELLO_RSA"

print("========== Q1 Part C : Common Modulus Attack ==========")
print(f"Requested Shared Modulus Size = {Bits} bits")
print("Generating Shared Modulus...")
print()

P, Q, N, Phi = GenerateSharedModulus(Bits)
EOne, ETwo = ChooseExponents(Phi)
Message = StringToInt(MessageText)

print("Shared Modulus Setup Completed")
print()
print(f"P = {P}")
print(f"Q = {Q}")
print(f"N = {N}")
print(f"Phi(N) = {Phi}")
print()
print("Chosen Public Exponents")
print(f"E1 = {EOne}")
print(f"E2 = {ETwo}")
print(f"Gcd(E1, E2) = {gcd(EOne, ETwo)}")
print()
print("Original Message")
print(f"Message Text = {MessageText}")
print(f"Message Integer = {Message}")
print(f"Is Message < N ? {'Yes' if Message < N else 'No'}")
print()

CipherOne = pow(Message, EOne, N)
CipherTwo = pow(Message, ETwo, N)

print("Ciphertexts Intercepted by Eavesdropper")
print(f"C1 = M^E1 mod N = {CipherOne}")
print(f"C2 = M^E2 mod N = {CipherTwo}")
print()

Recovered, S, T = CommonModulusAttack(CipherOne, CipherTwo, EOne, ETwo, N)
RecoveredText = IntToString(Recovered)

print("Bezout Coefficients Found")
print(f"S = {S}")
print(f"T = {T}")
print(f"S*E1 + T*E2 = {S * EOne + T * ETwo}")
print()
print("Recovered Message")
print(f"Recovered Integer = {Recovered}")
print(f"Recovered Text = {RecoveredText}")
print()
print("Final Verification")
print(f"Recovered Text == Original Message ? {RecoveredText == MessageText}")
print("=======================================================")