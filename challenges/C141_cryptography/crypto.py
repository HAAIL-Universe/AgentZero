"""
C141: Cryptography -- AES, RSA, Elliptic Curves, SHA-256, HMAC
New domain: bit manipulation, modular arithmetic, finite fields.
Standalone implementation -- no external crypto libraries.
"""

import struct
import os
import math

# ============================================================
# Utility: Modular Arithmetic
# ============================================================

def gcd(a, b):
    """Greatest common divisor (Euclidean algorithm)."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def extended_gcd(a, b):
    """Extended GCD: returns (g, x, y) such that a*x + b*y = g."""
    if a == 0:
        return b, 0, 1
    g, x, y = extended_gcd(b % a, a)
    return g, y - (b // a) * x, x

def mod_inverse(a, m):
    """Modular multiplicative inverse of a mod m."""
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"No modular inverse for {a} mod {m}")
    return x % m

def mod_pow(base, exp, mod):
    """Fast modular exponentiation."""
    result = 1
    base = base % mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return result

def miller_rabin(n, k=20):
    """Miller-Rabin primality test with k rounds."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    # Deterministic witnesses for small numbers
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
        if a >= n:
            continue
        x = mod_pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = mod_pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def generate_prime(bits):
    """Generate a random prime of given bit length."""
    while True:
        # Generate random odd number of correct bit length
        n = int.from_bytes(os.urandom(bits // 8), 'big')
        n |= (1 << (bits - 1)) | 1  # Set high bit and low bit
        if miller_rabin(n):
            return n

# ============================================================
# SHA-256
# ============================================================

class SHA256:
    """SHA-256 hash function (FIPS 180-4)."""

    # Initial hash values (first 32 bits of fractional parts of sqrt of first 8 primes)
    H0 = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]

    # Round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
    K = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ]

    MASK = 0xFFFFFFFF

    def __init__(self):
        self._h = list(self.H0)
        self._buffer = b""
        self._length = 0

    @staticmethod
    def _rotr(x, n):
        return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

    @staticmethod
    def _shr(x, n):
        return x >> n

    def _compress(self, block):
        assert len(block) == 64
        w = list(struct.unpack('>16L', block))
        for i in range(16, 64):
            s0 = self._rotr(w[i-15], 7) ^ self._rotr(w[i-15], 18) ^ self._shr(w[i-15], 3)
            s1 = self._rotr(w[i-2], 17) ^ self._rotr(w[i-2], 19) ^ self._shr(w[i-2], 10)
            w.append((w[i-16] + s0 + w[i-7] + s1) & self.MASK)

        a, b, c, d, e, f, g, h = self._h

        for i in range(64):
            S1 = self._rotr(e, 6) ^ self._rotr(e, 11) ^ self._rotr(e, 25)
            ch = (e & f) ^ (~e & g) & self.MASK
            temp1 = (h + S1 + ch + self.K[i] + w[i]) & self.MASK
            S0 = self._rotr(a, 2) ^ self._rotr(a, 13) ^ self._rotr(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            temp2 = (S0 + maj) & self.MASK

            h = g
            g = f
            f = e
            e = (d + temp1) & self.MASK
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & self.MASK

        for i, val in enumerate([a, b, c, d, e, f, g, h]):
            self._h[i] = (self._h[i] + val) & self.MASK

    def update(self, data):
        if isinstance(data, str):
            data = data.encode('utf-8')
        self._buffer += data
        self._length += len(data)
        while len(self._buffer) >= 64:
            self._compress(self._buffer[:64])
            self._buffer = self._buffer[64:]
        return self

    def digest(self):
        # Padding
        msg = self._buffer
        msg_len = self._length
        msg += b'\x80'
        while (len(msg) + 8) % 64 != 0:
            msg += b'\x00'
        msg += struct.pack('>Q', msg_len * 8)

        # Process remaining blocks
        h = list(self._h)
        sha = SHA256()
        sha._h = h
        for i in range(0, len(msg), 64):
            sha._compress(msg[i:i+64])
        return struct.pack('>8L', *sha._h)

    def hexdigest(self):
        return self.digest().hex()

    @classmethod
    def hash(cls, data):
        return cls().update(data).digest()

    @classmethod
    def hexhash(cls, data):
        return cls().update(data).hexdigest()


# ============================================================
# HMAC
# ============================================================

class HMAC:
    """HMAC using SHA-256."""

    BLOCK_SIZE = 64

    def __init__(self, key, msg=None):
        if isinstance(key, str):
            key = key.encode('utf-8')
        if len(key) > self.BLOCK_SIZE:
            key = SHA256.hash(key)
        key = key.ljust(self.BLOCK_SIZE, b'\x00')
        self._o_key_pad = bytes(k ^ 0x5c for k in key)
        self._i_key_pad = bytes(k ^ 0x36 for k in key)
        self._inner = SHA256().update(self._i_key_pad)
        if msg is not None:
            self._inner.update(msg)

    def update(self, msg):
        self._inner.update(msg)
        return self

    def digest(self):
        inner_hash = self._inner.digest()
        return SHA256().update(self._o_key_pad).update(inner_hash).digest()

    def hexdigest(self):
        return self.digest().hex()

    @classmethod
    def mac(cls, key, msg):
        return cls(key, msg).digest()

    @classmethod
    def hexmac(cls, key, msg):
        return cls(key, msg).hexdigest()


# ============================================================
# AES (Advanced Encryption Standard)
# ============================================================

class AES:
    """AES-128/192/256 block cipher."""

    # S-box (substitution box)
    SBOX = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
    ]

    # Inverse S-box
    INV_SBOX = [0] * 256

    # Round constants
    RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]

    def __init__(self, key):
        if isinstance(key, str):
            key = key.encode('utf-8')
        if len(key) not in (16, 24, 32):
            raise ValueError(f"AES key must be 16, 24, or 32 bytes, got {len(key)}")
        self._key = key
        self._nk = len(key) // 4  # Number of 32-bit words in key
        self._nr = self._nk + 6    # Number of rounds
        self._round_keys = self._key_expansion(key)

    @classmethod
    def _init_inv_sbox(cls):
        for i in range(256):
            cls.INV_SBOX[cls.SBOX[i]] = i

    @staticmethod
    def _xtime(a):
        """Multiply by x in GF(2^8)."""
        return ((a << 1) ^ (0x1b if a & 0x80 else 0)) & 0xFF

    @classmethod
    def _gf_mul(cls, a, b):
        """Multiply two bytes in GF(2^8)."""
        result = 0
        for _ in range(8):
            if b & 1:
                result ^= a
            a = cls._xtime(a)
            b >>= 1
        return result

    def _key_expansion(self, key):
        """Expand key into round key schedule."""
        nk = self._nk
        nr = self._nr
        # Convert key to list of 4-byte words
        w = []
        for i in range(nk):
            w.append(list(key[4*i:4*i+4]))
        for i in range(nk, 4 * (nr + 1)):
            temp = list(w[i - 1])
            if i % nk == 0:
                # RotWord + SubWord + Rcon
                temp = temp[1:] + temp[:1]
                temp = [self.SBOX[b] for b in temp]
                temp[0] ^= self.RCON[i // nk - 1]
            elif nk > 6 and i % nk == 4:
                temp = [self.SBOX[b] for b in temp]
            w.append([w[i - nk][j] ^ temp[j] for j in range(4)])
        return w

    def _add_round_key(self, state, round_num):
        for c in range(4):
            for r in range(4):
                state[r][c] ^= self._round_keys[round_num * 4 + c][r]

    def _sub_bytes(self, state):
        for r in range(4):
            for c in range(4):
                state[r][c] = self.SBOX[state[r][c]]

    def _inv_sub_bytes(self, state):
        for r in range(4):
            for c in range(4):
                state[r][c] = self.INV_SBOX[state[r][c]]

    def _shift_rows(self, state):
        state[1] = state[1][1:] + state[1][:1]
        state[2] = state[2][2:] + state[2][:2]
        state[3] = state[3][3:] + state[3][:3]

    def _inv_shift_rows(self, state):
        state[1] = state[1][3:] + state[1][:3]
        state[2] = state[2][2:] + state[2][:2]
        state[3] = state[3][1:] + state[3][:1]

    def _mix_columns(self, state):
        for c in range(4):
            a = [state[r][c] for r in range(4)]
            state[0][c] = self._gf_mul(2, a[0]) ^ self._gf_mul(3, a[1]) ^ a[2] ^ a[3]
            state[1][c] = a[0] ^ self._gf_mul(2, a[1]) ^ self._gf_mul(3, a[2]) ^ a[3]
            state[2][c] = a[0] ^ a[1] ^ self._gf_mul(2, a[2]) ^ self._gf_mul(3, a[3])
            state[3][c] = self._gf_mul(3, a[0]) ^ a[1] ^ a[2] ^ self._gf_mul(2, a[3])

    def _inv_mix_columns(self, state):
        for c in range(4):
            a = [state[r][c] for r in range(4)]
            state[0][c] = self._gf_mul(14, a[0]) ^ self._gf_mul(11, a[1]) ^ self._gf_mul(13, a[2]) ^ self._gf_mul(9, a[3])
            state[1][c] = self._gf_mul(9, a[0]) ^ self._gf_mul(14, a[1]) ^ self._gf_mul(11, a[2]) ^ self._gf_mul(13, a[3])
            state[2][c] = self._gf_mul(13, a[0]) ^ self._gf_mul(9, a[1]) ^ self._gf_mul(14, a[2]) ^ self._gf_mul(11, a[3])
            state[3][c] = self._gf_mul(11, a[0]) ^ self._gf_mul(13, a[1]) ^ self._gf_mul(9, a[2]) ^ self._gf_mul(14, a[3])

    @staticmethod
    def _bytes_to_state(data):
        """Convert 16 bytes to 4x4 state matrix (column-major)."""
        state = [[0]*4 for _ in range(4)]
        for i in range(16):
            state[i % 4][i // 4] = data[i]
        return state

    @staticmethod
    def _state_to_bytes(state):
        """Convert 4x4 state matrix to 16 bytes."""
        result = []
        for c in range(4):
            for r in range(4):
                result.append(state[r][c])
        return bytes(result)

    def encrypt_block(self, plaintext):
        """Encrypt a single 16-byte block."""
        if len(plaintext) != 16:
            raise ValueError("Block must be 16 bytes")
        state = self._bytes_to_state(plaintext)
        self._add_round_key(state, 0)
        for rnd in range(1, self._nr):
            self._sub_bytes(state)
            self._shift_rows(state)
            self._mix_columns(state)
            self._add_round_key(state, rnd)
        self._sub_bytes(state)
        self._shift_rows(state)
        self._add_round_key(state, self._nr)
        return self._state_to_bytes(state)

    def decrypt_block(self, ciphertext):
        """Decrypt a single 16-byte block."""
        if len(ciphertext) != 16:
            raise ValueError("Block must be 16 bytes")
        state = self._bytes_to_state(ciphertext)
        self._add_round_key(state, self._nr)
        for rnd in range(self._nr - 1, 0, -1):
            self._inv_shift_rows(state)
            self._inv_sub_bytes(state)
            self._add_round_key(state, rnd)
            self._inv_mix_columns(state)
        self._inv_shift_rows(state)
        self._inv_sub_bytes(state)
        self._add_round_key(state, 0)
        return self._state_to_bytes(state)

# Initialize inverse S-box
AES._init_inv_sbox()


# ============================================================
# AES Modes of Operation
# ============================================================

def pkcs7_pad(data, block_size=16):
    """PKCS#7 padding."""
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)

def pkcs7_unpad(data):
    """Remove PKCS#7 padding."""
    pad_len = data[-1]
    if pad_len < 1 or pad_len > 16:
        raise ValueError("Invalid padding")
    if data[-pad_len:] != bytes([pad_len] * pad_len):
        raise ValueError("Invalid padding")
    return data[:-pad_len]

def xor_bytes(a, b):
    """XOR two byte strings."""
    return bytes(x ^ y for x, y in zip(a, b))


class AES_ECB:
    """AES in Electronic Codebook mode."""

    def __init__(self, key):
        self._aes = AES(key)

    def encrypt(self, plaintext):
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        plaintext = pkcs7_pad(plaintext)
        result = b""
        for i in range(0, len(plaintext), 16):
            result += self._aes.encrypt_block(plaintext[i:i+16])
        return result

    def decrypt(self, ciphertext):
        result = b""
        for i in range(0, len(ciphertext), 16):
            result += self._aes.decrypt_block(ciphertext[i:i+16])
        return pkcs7_unpad(result)


class AES_CBC:
    """AES in Cipher Block Chaining mode."""

    def __init__(self, key, iv=None):
        self._aes = AES(key)
        self._iv = iv if iv else os.urandom(16)

    @property
    def iv(self):
        return self._iv

    def encrypt(self, plaintext):
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        plaintext = pkcs7_pad(plaintext)
        result = b""
        prev = self._iv
        for i in range(0, len(plaintext), 16):
            block = xor_bytes(plaintext[i:i+16], prev)
            encrypted = self._aes.encrypt_block(block)
            result += encrypted
            prev = encrypted
        return result

    def decrypt(self, ciphertext):
        result = b""
        prev = self._iv
        for i in range(0, len(ciphertext), 16):
            block = ciphertext[i:i+16]
            decrypted = xor_bytes(self._aes.decrypt_block(block), prev)
            result += decrypted
            prev = block
        return pkcs7_unpad(result)


class AES_CTR:
    """AES in Counter mode (stream cipher)."""

    def __init__(self, key, nonce=None):
        self._aes = AES(key)
        self._nonce = nonce if nonce else os.urandom(8)

    @property
    def nonce(self):
        return self._nonce

    def _counter_block(self, counter):
        return self._nonce + struct.pack('>Q', counter)

    def encrypt(self, plaintext):
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        result = b""
        counter = 0
        for i in range(0, len(plaintext), 16):
            keystream = self._aes.encrypt_block(self._counter_block(counter))
            block = plaintext[i:i+16]
            result += xor_bytes(block, keystream[:len(block)])
            counter += 1
        return result

    def decrypt(self, ciphertext):
        """CTR mode: encryption and decryption are the same operation."""
        return self.encrypt(ciphertext)


# ============================================================
# RSA
# ============================================================

class RSAKeyPair:
    """RSA key pair."""

    def __init__(self, n, e, d, p=None, q=None):
        self.n = n
        self.e = e
        self.d = d
        self.p = p
        self.q = q
        self.key_size = n.bit_length()

    @classmethod
    def generate(cls, bits=1024):
        """Generate a new RSA key pair."""
        half = bits // 2
        p = generate_prime(half)
        q = generate_prime(half)
        while p == q:
            q = generate_prime(half)
        n = p * q
        phi = (p - 1) * (q - 1)
        e = 65537
        d = mod_inverse(e, phi)
        return cls(n, e, d, p, q)

    @property
    def public_key(self):
        return (self.n, self.e)

    @property
    def private_key(self):
        return (self.n, self.d)


class RSA:
    """RSA encryption/decryption and signing."""

    @staticmethod
    def encrypt(plaintext, public_key):
        """Encrypt with public key (textbook RSA on integer)."""
        n, e = public_key
        if isinstance(plaintext, (bytes, bytearray)):
            m = int.from_bytes(plaintext, 'big')
        elif isinstance(plaintext, int):
            m = plaintext
        else:
            m = int.from_bytes(plaintext.encode('utf-8'), 'big')
        if m >= n:
            raise ValueError("Message too large for key size")
        return mod_pow(m, e, n)

    @staticmethod
    def decrypt(ciphertext, private_key):
        """Decrypt with private key."""
        n, d = private_key
        return mod_pow(ciphertext, d, n)

    @staticmethod
    def decrypt_bytes(ciphertext, private_key):
        """Decrypt and return bytes."""
        m = RSA.decrypt(ciphertext, private_key)
        byte_len = (m.bit_length() + 7) // 8
        return m.to_bytes(byte_len, 'big')

    @staticmethod
    def sign(message, private_key):
        """Sign a message (hash then encrypt with private key)."""
        if isinstance(message, str):
            message = message.encode('utf-8')
        h = SHA256.hash(message)
        h_int = int.from_bytes(h, 'big')
        n, d = private_key
        return mod_pow(h_int, d, n)

    @staticmethod
    def verify(message, signature, public_key):
        """Verify a signature."""
        if isinstance(message, str):
            message = message.encode('utf-8')
        h = SHA256.hash(message)
        h_int = int.from_bytes(h, 'big')
        n, e = public_key
        decrypted = mod_pow(signature, e, n)
        return decrypted == h_int


# ============================================================
# OAEP Padding (simplified, for RSA)
# ============================================================

class OAEP:
    """Optimal Asymmetric Encryption Padding (simplified)."""

    @staticmethod
    def _mgf1(seed, length):
        """Mask Generation Function 1 (MGF1) using SHA-256."""
        result = b""
        counter = 0
        while len(result) < length:
            c = struct.pack('>I', counter)
            result += SHA256.hash(seed + c)
            counter += 1
        return result[:length]

    @classmethod
    def pad(cls, message, key_size_bytes, label=b""):
        """OAEP pad a message."""
        if isinstance(message, str):
            message = message.encode('utf-8')
        h_len = 32  # SHA-256 hash length
        max_msg_len = key_size_bytes - 2 * h_len - 2
        if len(message) > max_msg_len:
            raise ValueError(f"Message too long: {len(message)} > {max_msg_len}")

        l_hash = SHA256.hash(label)
        ps = b'\x00' * (max_msg_len - len(message))
        db = l_hash + ps + b'\x01' + message
        seed = os.urandom(h_len)
        db_mask = cls._mgf1(seed, len(db))
        masked_db = xor_bytes(db, db_mask)
        seed_mask = cls._mgf1(masked_db, h_len)
        masked_seed = xor_bytes(seed, seed_mask)
        return b'\x00' + masked_seed + masked_db

    @classmethod
    def unpad(cls, padded, label=b""):
        """Remove OAEP padding."""
        h_len = 32
        if padded[0:1] != b'\x00':
            raise ValueError("Decryption error")
        masked_seed = padded[1:1+h_len]
        masked_db = padded[1+h_len:]
        seed_mask = cls._mgf1(masked_db, h_len)
        seed = xor_bytes(masked_seed, seed_mask)
        db_mask = cls._mgf1(seed, len(masked_db))
        db = xor_bytes(masked_db, db_mask)
        l_hash = SHA256.hash(label)
        if db[:h_len] != l_hash:
            raise ValueError("Decryption error")
        # Find 0x01 separator
        i = h_len
        while i < len(db) and db[i] == 0:
            i += 1
        if i >= len(db) or db[i] != 1:
            raise ValueError("Decryption error")
        return db[i+1:]


# ============================================================
# Elliptic Curve Cryptography
# ============================================================

class EllipticCurve:
    """Elliptic curve over a prime field: y^2 = x^3 + ax + b (mod p)."""

    def __init__(self, a, b, p):
        self.a = a
        self.b = b
        self.p = p
        # Check discriminant
        disc = (4 * a * a * a + 27 * b * b) % p
        if disc == 0:
            raise ValueError("Singular curve (discriminant is zero)")

    def is_on_curve(self, point):
        if point is None:  # Point at infinity
            return True
        x, y = point
        return (y * y - x * x * x - self.a * x - self.b) % self.p == 0

    def add(self, P, Q):
        """Add two points on the curve."""
        if P is None:
            return Q
        if Q is None:
            return P
        x1, y1 = P
        x2, y2 = Q
        if x1 == x2:
            if (y1 + y2) % self.p == 0:
                return None  # Point at infinity
            # Point doubling
            lam = (3 * x1 * x1 + self.a) * mod_inverse(2 * y1, self.p) % self.p
        else:
            lam = (y2 - y1) * mod_inverse(x2 - x1, self.p) % self.p
        x3 = (lam * lam - x1 - x2) % self.p
        y3 = (lam * (x1 - x3) - y1) % self.p
        return (x3, y3)

    def multiply(self, k, P):
        """Scalar multiplication using double-and-add."""
        if k == 0 or P is None:
            return None
        if k < 0:
            P = (P[0], (-P[1]) % self.p)
            k = -k
        result = None
        addend = P
        while k:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result

    def negate(self, P):
        """Negate a point."""
        if P is None:
            return None
        return (P[0], (-P[1]) % self.p)


# secp256k1 curve parameters (used by Bitcoin)
SECP256K1 = EllipticCurve(
    a=0,
    b=7,
    p=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F,
)
SECP256K1_G = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)
SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


# Smaller curve for testing (P-192 style but custom small)
SMALL_CURVE = EllipticCurve(a=2, b=3, p=97)


class ECDSA:
    """Elliptic Curve Digital Signature Algorithm."""

    def __init__(self, curve=None, G=None, n=None):
        self.curve = curve or SECP256K1
        self.G = G or SECP256K1_G
        self.n = n or SECP256K1_N

    def generate_keypair(self):
        """Generate a private/public key pair."""
        private_key = int.from_bytes(os.urandom(32), 'big') % (self.n - 1) + 1
        public_key = self.curve.multiply(private_key, self.G)
        return private_key, public_key

    def sign(self, message, private_key, k=None):
        """Sign a message. k is the ephemeral key (random if not provided)."""
        if isinstance(message, str):
            message = message.encode('utf-8')
        z = int.from_bytes(SHA256.hash(message), 'big') % self.n
        while True:
            if k is None:
                k_val = int.from_bytes(os.urandom(32), 'big') % (self.n - 1) + 1
            else:
                k_val = k
            R = self.curve.multiply(k_val, self.G)
            r = R[0] % self.n
            if r == 0:
                if k is not None:
                    raise ValueError("Bad k value")
                continue
            s = (mod_inverse(k_val, self.n) * (z + r * private_key)) % self.n
            if s == 0:
                if k is not None:
                    raise ValueError("Bad k value")
                continue
            return (r, s)

    def verify(self, message, signature, public_key):
        """Verify an ECDSA signature."""
        if isinstance(message, str):
            message = message.encode('utf-8')
        r, s = signature
        if not (1 <= r < self.n and 1 <= s < self.n):
            return False
        z = int.from_bytes(SHA256.hash(message), 'big') % self.n
        w = mod_inverse(s, self.n)
        u1 = (z * w) % self.n
        u2 = (r * w) % self.n
        P = self.curve.add(
            self.curve.multiply(u1, self.G),
            self.curve.multiply(u2, public_key)
        )
        if P is None:
            return False
        return P[0] % self.n == r


class ECDH:
    """Elliptic Curve Diffie-Hellman key exchange."""

    def __init__(self, curve=None, G=None, n=None):
        self.curve = curve or SECP256K1
        self.G = G or SECP256K1_G
        self.n = n or SECP256K1_N

    def generate_keypair(self):
        """Generate a private/public key pair."""
        private_key = int.from_bytes(os.urandom(32), 'big') % (self.n - 1) + 1
        public_key = self.curve.multiply(private_key, self.G)
        return private_key, public_key

    def compute_shared_secret(self, private_key, other_public_key):
        """Compute the shared secret from private key and other's public key."""
        shared_point = self.curve.multiply(private_key, other_public_key)
        if shared_point is None:
            raise ValueError("Invalid shared secret (point at infinity)")
        # Hash the x-coordinate to get a uniform key
        x_bytes = shared_point[0].to_bytes(32, 'big')
        return SHA256.hash(x_bytes)


# ============================================================
# PBKDF2 (Password-Based Key Derivation Function 2)
# ============================================================

class PBKDF2:
    """PBKDF2-HMAC-SHA256."""

    @staticmethod
    def derive(password, salt, iterations=10000, key_length=32):
        """Derive a key from a password."""
        if isinstance(password, str):
            password = password.encode('utf-8')
        if isinstance(salt, str):
            salt = salt.encode('utf-8')

        dk = b""
        block_num = 1
        while len(dk) < key_length:
            u = HMAC.mac(password, salt + struct.pack('>I', block_num))
            result = u
            for _ in range(iterations - 1):
                u = HMAC.mac(password, u)
                result = xor_bytes(result, u)
            dk += result
            block_num += 1
        return dk[:key_length]


# ============================================================
# ChaCha20 Stream Cipher
# ============================================================

class ChaCha20:
    """ChaCha20 stream cipher (RFC 7539)."""

    def __init__(self, key, nonce, counter=0):
        if isinstance(key, str):
            key = key.encode('utf-8')
        if len(key) != 32:
            raise ValueError("ChaCha20 key must be 32 bytes")
        if isinstance(nonce, str):
            nonce = nonce.encode('utf-8')
        if len(nonce) != 12:
            raise ValueError("ChaCha20 nonce must be 12 bytes")
        self._key = key
        self._nonce = nonce
        self._counter = counter

    @staticmethod
    def _quarter_round(state, a, b, c, d):
        state[a] = (state[a] + state[b]) & 0xFFFFFFFF
        state[d] ^= state[a]
        state[d] = ((state[d] << 16) | (state[d] >> 16)) & 0xFFFFFFFF
        state[c] = (state[c] + state[d]) & 0xFFFFFFFF
        state[b] ^= state[c]
        state[b] = ((state[b] << 12) | (state[b] >> 20)) & 0xFFFFFFFF
        state[a] = (state[a] + state[b]) & 0xFFFFFFFF
        state[d] ^= state[a]
        state[d] = ((state[d] << 8) | (state[d] >> 24)) & 0xFFFFFFFF
        state[c] = (state[c] + state[d]) & 0xFFFFFFFF
        state[b] ^= state[c]
        state[b] = ((state[b] << 7) | (state[b] >> 25)) & 0xFFFFFFFF

    def _block(self, counter):
        # Constants "expand 32-byte k"
        state = [
            0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
        ]
        # Key (8 words)
        state.extend(struct.unpack('<8I', self._key))
        # Counter (1 word)
        state.append(counter & 0xFFFFFFFF)
        # Nonce (3 words)
        state.extend(struct.unpack('<3I', self._nonce))

        working = list(state)
        for _ in range(10):  # 20 rounds = 10 double rounds
            # Column rounds
            self._quarter_round(working, 0, 4, 8, 12)
            self._quarter_round(working, 1, 5, 9, 13)
            self._quarter_round(working, 2, 6, 10, 14)
            self._quarter_round(working, 3, 7, 11, 15)
            # Diagonal rounds
            self._quarter_round(working, 0, 5, 10, 15)
            self._quarter_round(working, 1, 6, 11, 12)
            self._quarter_round(working, 2, 7, 8, 13)
            self._quarter_round(working, 3, 4, 9, 14)

        output = [(working[i] + state[i]) & 0xFFFFFFFF for i in range(16)]
        return struct.pack('<16I', *output)

    def encrypt(self, plaintext):
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        result = b""
        counter = self._counter
        for i in range(0, len(plaintext), 64):
            keystream = self._block(counter)
            block = plaintext[i:i+64]
            result += xor_bytes(block, keystream[:len(block)])
            counter += 1
        return result

    def decrypt(self, ciphertext):
        return self.encrypt(ciphertext)


# ============================================================
# Poly1305 MAC
# ============================================================

class Poly1305:
    """Poly1305 message authentication code."""

    def __init__(self, key):
        if len(key) != 32:
            raise ValueError("Poly1305 key must be 32 bytes")
        # r (clamped)
        r = int.from_bytes(key[:16], 'little')
        r &= 0x0ffffffc0ffffffc0ffffffc0fffffff
        self._r = r
        # s
        self._s = int.from_bytes(key[16:], 'little')
        self._p = (1 << 130) - 5

    def mac(self, message):
        if isinstance(message, str):
            message = message.encode('utf-8')
        acc = 0
        for i in range(0, len(message), 16):
            block = message[i:i+16]
            n = int.from_bytes(block, 'little') + (1 << (8 * len(block)))
            acc = ((acc + n) * self._r) % self._p
        acc = (acc + self._s) & ((1 << 128) - 1)
        return acc.to_bytes(16, 'little')

    @classmethod
    def authenticate(cls, key, message):
        return cls(key).mac(message)
