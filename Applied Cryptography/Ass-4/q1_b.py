import random
from math import gcd

# Applied Cryptography Assignment 4
# Question 1 Part B
# RSA implementation from scratch

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

# Generates RSA public and private keys
def GenerateKeypair(Bits):
    E = 65537
    Half = Bits // 2

    while True:
        P = GeneratePrime(Half)
        Q = GeneratePrime(Bits - Half)

        if P == Q:
            continue

        N = P * Q
        if N.bit_length() != Bits:
            continue

        Phi = (P - 1) * (Q - 1)

        if gcd(E, Phi) == 1:
            D = ModInverse(E, Phi)
            return (E, N), (D, N), (P, Q, Phi)

# Textbook RSA encryption
def Encrypt(Message, E, N):
    if not (0 <= Message < N):
        raise ValueError("Message must satisfy 0 <= Message < N")
    return pow(Message, E, N)

# Textbook RSA decryption
def Decrypt(Cipher, D, N):
    if not (0 <= Cipher < N):
        raise ValueError("Cipher must satisfy 0 <= Cipher < N")
    return pow(Cipher, D, N)

# Encodes ASCII text as an integer
def StringToInt(Text):
    return int.from_bytes(Text.encode("ascii"), byteorder="big")

# Decodes an integer back to ASCII text
def IntToString(Value):
    if Value == 0:
        return ""
    Length = (Value.bit_length() + 7) // 8
    return Value.to_bytes(Length, byteorder="big").decode("ascii")

Bits = 512
StudentId = "2022484"

print("========== Q1 Part B : RSA Implementation ==========")
print(f"Requested Key Size = {Bits} bits")
print("Generating RSA Keypair...")
print()

PublicKey, PrivateKey, Extra = GenerateKeypair(Bits)
E, N = PublicKey
D, _ = PrivateKey
P, Q, Phi = Extra

print("Key Generation Completed")
print()
print("Public Key:")
print(f"E = {E}")
print(f"N = {N}")
print()
print("Private Key:")
print(f"D = {D}")
print()
print("Internal Values:")
print(f"P = {P}")
print(f"Q = {Q}")
print(f"Phi(N) = {Phi}")
print()
print(f"Check Gcd(E, Phi(N)) = {gcd(E, Phi)}")
print(f"Check (E * D) % Phi(N) = {(E * D) % Phi}")
print()

Message = StringToInt(StudentId)

print("Message Preparation")
print(f"Student ID String = {StudentId}")
print(f"ASCII Integer Form = {Message}")
print(f"Is Message < N ? {'Yes' if Message < N else 'No'}")
print()

Cipher = Encrypt(Message, E, N)
Recovered = Decrypt(Cipher, D, N)
RecoveredText = IntToString(Recovered)

print("Encryption Result")
print(f"Ciphertext = {Cipher}")
print()
print("Decryption Result")
print(f"Recovered Integer = {Recovered}")
print(f"Recovered Text = {RecoveredText}")
print()
print("Final Verification")
print(f"Recovered Text == Original Student ID ? {RecoveredText == StudentId}")
print("====================================================")