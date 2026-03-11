"""Tests for C141: Cryptography."""
import pytest
import os
import struct
from crypto import (
    gcd, extended_gcd, mod_inverse, mod_pow, miller_rabin, generate_prime,
    SHA256, HMAC,
    AES, AES_ECB, AES_CBC, AES_CTR, pkcs7_pad, pkcs7_unpad, xor_bytes,
    RSAKeyPair, RSA, OAEP,
    EllipticCurve, SECP256K1, SECP256K1_G, SECP256K1_N, SMALL_CURVE,
    ECDSA, ECDH,
    PBKDF2, ChaCha20, Poly1305,
)


# ============================================================
# Modular Arithmetic
# ============================================================

class TestModularArithmetic:
    def test_gcd_basic(self):
        assert gcd(12, 8) == 4
        assert gcd(17, 13) == 1
        assert gcd(100, 75) == 25

    def test_gcd_with_zero(self):
        assert gcd(0, 5) == 5
        assert gcd(7, 0) == 7

    def test_gcd_coprime(self):
        assert gcd(13, 17) == 1
        assert gcd(35, 64) == 1

    def test_extended_gcd(self):
        g, x, y = extended_gcd(35, 15)
        assert g == 5
        assert 35 * x + 15 * y == 5

    def test_extended_gcd_coprime(self):
        g, x, y = extended_gcd(17, 13)
        assert g == 1
        assert 17 * x + 13 * y == 1

    def test_mod_inverse(self):
        assert (mod_inverse(3, 7) * 3) % 7 == 1
        assert (mod_inverse(17, 43) * 17) % 43 == 1

    def test_mod_inverse_no_inverse(self):
        with pytest.raises(ValueError):
            mod_inverse(6, 9)

    def test_mod_pow(self):
        assert mod_pow(2, 10, 1000) == 24  # 1024 mod 1000
        assert mod_pow(3, 13, 50) == 3**13 % 50

    def test_mod_pow_large(self):
        assert mod_pow(2, 100, 1000000007) == pow(2, 100, 1000000007)

    def test_miller_rabin_primes(self):
        primes = [2, 3, 5, 7, 11, 13, 97, 101, 1009, 104729]
        for p in primes:
            assert miller_rabin(p), f"{p} should be prime"

    def test_miller_rabin_composites(self):
        composites = [4, 6, 8, 9, 15, 21, 100, 1000, 104730]
        for c in composites:
            assert not miller_rabin(c), f"{c} should be composite"

    def test_miller_rabin_edge(self):
        assert not miller_rabin(0)
        assert not miller_rabin(1)

    def test_generate_prime(self):
        p = generate_prime(64)
        assert miller_rabin(p)
        assert p.bit_length() == 64


# ============================================================
# SHA-256
# ============================================================

class TestSHA256:
    def test_empty_string(self):
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert SHA256.hexhash(b"") == expected

    def test_abc(self):
        expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        assert SHA256.hexhash(b"abc") == expected

    def test_long_message(self):
        # "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        expected = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        assert SHA256.hexhash(msg) == expected

    def test_incremental_update(self):
        sha = SHA256()
        sha.update(b"abc")
        sha.update(b"def")
        full = SHA256.hexhash(b"abcdef")
        assert sha.hexdigest() == full

    def test_string_input(self):
        assert SHA256.hexhash("abc") == SHA256.hexhash(b"abc")

    def test_deterministic(self):
        h1 = SHA256.hexhash(b"test message")
        h2 = SHA256.hexhash(b"test message")
        assert h1 == h2

    def test_avalanche(self):
        h1 = SHA256.hash(b"test1")
        h2 = SHA256.hash(b"test2")
        # Different inputs -> different outputs
        assert h1 != h2
        # Count differing bits (should be roughly half)
        diff_bits = sum(bin(a ^ b).count('1') for a, b in zip(h1, h2))
        assert diff_bits > 50  # Should be ~128 out of 256

    def test_digest_length(self):
        assert len(SHA256.hash(b"anything")) == 32
        assert len(SHA256.hexhash(b"anything")) == 64


# ============================================================
# HMAC
# ============================================================

class TestHMAC:
    def test_basic(self):
        mac = HMAC.hexmac(b"key", b"message")
        assert len(mac) == 64  # 32 bytes hex

    def test_deterministic(self):
        mac1 = HMAC.mac(b"key", b"message")
        mac2 = HMAC.mac(b"key", b"message")
        assert mac1 == mac2

    def test_different_keys(self):
        mac1 = HMAC.mac(b"key1", b"message")
        mac2 = HMAC.mac(b"key2", b"message")
        assert mac1 != mac2

    def test_different_messages(self):
        mac1 = HMAC.mac(b"key", b"msg1")
        mac2 = HMAC.mac(b"key", b"msg2")
        assert mac1 != mac2

    def test_long_key(self):
        long_key = b"k" * 100  # > 64 bytes
        mac = HMAC.mac(long_key, b"message")
        assert len(mac) == 32

    def test_incremental(self):
        h = HMAC(b"key")
        h.update(b"hello ")
        h.update(b"world")
        assert h.digest() == HMAC.mac(b"key", b"hello world")

    def test_string_key(self):
        mac1 = HMAC.mac("key", b"msg")
        mac2 = HMAC.mac(b"key", b"msg")
        assert mac1 == mac2

    # RFC 4231 test vector 1
    def test_rfc4231_vector1(self):
        key = bytes([0x0b] * 20)
        data = b"Hi There"
        expected = "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        assert HMAC.hexmac(key, data) == expected

    # RFC 4231 test vector 2
    def test_rfc4231_vector2(self):
        key = b"Jefe"
        data = b"what do ya want for nothing?"
        expected = "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        assert HMAC.hexmac(key, data) == expected


# ============================================================
# AES Block Cipher
# ============================================================

class TestAES:
    def test_encrypt_decrypt_block_128(self):
        key = bytes(range(16))
        aes = AES(key)
        plaintext = bytes(range(16))
        ciphertext = aes.encrypt_block(plaintext)
        assert ciphertext != plaintext
        assert aes.decrypt_block(ciphertext) == plaintext

    def test_encrypt_decrypt_block_192(self):
        key = bytes(range(24))
        aes = AES(key)
        plaintext = b"0123456789abcdef"
        ciphertext = aes.encrypt_block(plaintext)
        assert aes.decrypt_block(ciphertext) == plaintext

    def test_encrypt_decrypt_block_256(self):
        key = bytes(range(32))
        aes = AES(key)
        plaintext = b"fedcba9876543210"
        ciphertext = aes.encrypt_block(plaintext)
        assert aes.decrypt_block(ciphertext) == plaintext

    def test_invalid_key_size(self):
        with pytest.raises(ValueError):
            AES(b"short")

    def test_invalid_block_size(self):
        aes = AES(bytes(16))
        with pytest.raises(ValueError):
            aes.encrypt_block(b"short")

    def test_different_keys_different_output(self):
        pt = bytes(16)
        c1 = AES(bytes(16)).encrypt_block(pt)
        c2 = AES(bytes([1]) + bytes(15)).encrypt_block(pt)
        assert c1 != c2

    # NIST AES-128 test vector (FIPS 197 Appendix B)
    def test_nist_aes128_vector(self):
        key = bytes([0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                     0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c])
        pt = bytes([0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
                    0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34])
        expected = bytes([0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
                         0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32])
        aes = AES(key)
        assert aes.encrypt_block(pt) == expected
        assert aes.decrypt_block(expected) == pt


# ============================================================
# AES Modes
# ============================================================

class TestAESModes:
    def test_ecb_encrypt_decrypt(self):
        key = bytes(range(16))
        ecb = AES_ECB(key)
        msg = b"Hello, World! This is a test message for ECB mode."
        ct = ecb.encrypt(msg)
        assert ecb.decrypt(ct) == msg

    def test_ecb_string_input(self):
        ecb = AES_ECB(bytes(16))
        ct = ecb.encrypt("hello")
        assert ecb.decrypt(ct) == b"hello"

    def test_cbc_encrypt_decrypt(self):
        key = bytes(range(16))
        iv = bytes(range(16))
        enc = AES_CBC(key, iv)
        dec = AES_CBC(key, iv)
        msg = b"CBC mode test with initialization vector"
        ct = enc.encrypt(msg)
        assert dec.decrypt(ct) == msg

    def test_cbc_iv_matters(self):
        key = bytes(16)
        msg = b"same plaintext!!"
        c1 = AES_CBC(key, bytes(16)).encrypt(msg)
        c2 = AES_CBC(key, bytes([1]) + bytes(15)).encrypt(msg)
        assert c1 != c2

    def test_cbc_chaining(self):
        """Identical plaintext blocks produce different ciphertext in CBC."""
        key = bytes(16)
        iv = bytes(16)
        msg = bytes(32)  # Two identical blocks
        ct = AES_CBC(key, iv).encrypt(msg)
        # In CBC, identical plaintext blocks should produce different ciphertext blocks
        # (first block XORed with IV, second with first ciphertext block)
        # Note: with padding, there are 3 blocks of ciphertext
        assert ct[:16] != ct[16:32]

    def test_ctr_encrypt_decrypt(self):
        key = bytes(range(16))
        nonce = bytes(range(8))
        ctr = AES_CTR(key, nonce)
        msg = b"CTR mode is a stream cipher mode"
        ct = ctr.encrypt(msg)
        assert AES_CTR(key, nonce).decrypt(ct) == msg

    def test_ctr_partial_block(self):
        key = bytes(16)
        nonce = bytes(8)
        ctr = AES_CTR(key, nonce)
        msg = b"short"  # Less than 16 bytes
        ct = ctr.encrypt(msg)
        assert len(ct) == len(msg)
        assert AES_CTR(key, nonce).decrypt(ct) == msg

    def test_ctr_stream_property(self):
        """CTR mode: encrypt(encrypt(x)) should NOT equal x (it's XOR with keystream)."""
        key = bytes(16)
        nonce = bytes(8)
        msg = b"test"
        ct = AES_CTR(key, nonce).encrypt(msg)
        # Decrypt with same key/nonce should give back plaintext
        assert AES_CTR(key, nonce).decrypt(ct) == msg


class TestPadding:
    def test_pkcs7_pad(self):
        assert pkcs7_pad(b"hello", 8) == b"hello\x03\x03\x03"
        assert len(pkcs7_pad(b"12345678", 8)) == 16  # Full block of padding

    def test_pkcs7_unpad(self):
        assert pkcs7_unpad(b"hello\x03\x03\x03") == b"hello"

    def test_pkcs7_roundtrip(self):
        for length in range(1, 33):
            data = os.urandom(length)
            assert pkcs7_unpad(pkcs7_pad(data)) == data

    def test_pkcs7_invalid_padding(self):
        with pytest.raises(ValueError):
            pkcs7_unpad(b"hello\x00")

    def test_xor_bytes(self):
        assert xor_bytes(b"\xff\x00", b"\x0f\xf0") == b"\xf0\xf0"


# ============================================================
# RSA
# ============================================================

class TestRSA:
    @pytest.fixture
    def keypair(self):
        return RSAKeyPair.generate(512)  # Small for testing speed

    def test_keygen(self, keypair):
        assert keypair.n > 0
        assert keypair.e == 65537
        assert keypair.d > 0
        assert keypair.p is not None
        assert keypair.q is not None
        assert keypair.p * keypair.q == keypair.n

    def test_encrypt_decrypt_int(self, keypair):
        msg = 42
        ct = RSA.encrypt(msg, keypair.public_key)
        pt = RSA.decrypt(ct, keypair.private_key)
        assert pt == msg

    def test_encrypt_decrypt_bytes(self, keypair):
        msg = b"Hello RSA"
        ct = RSA.encrypt(msg, keypair.public_key)
        pt = RSA.decrypt_bytes(ct, keypair.private_key)
        assert pt == msg

    def test_sign_verify(self, keypair):
        msg = b"Sign this message"
        sig = RSA.sign(msg, keypair.private_key)
        assert RSA.verify(msg, sig, keypair.public_key)

    def test_sign_verify_wrong_message(self, keypair):
        sig = RSA.sign(b"original", keypair.private_key)
        assert not RSA.verify(b"tampered", sig, keypair.public_key)

    def test_sign_verify_string(self, keypair):
        sig = RSA.sign("string message", keypair.private_key)
        assert RSA.verify("string message", sig, keypair.public_key)

    def test_different_keys_cant_decrypt(self):
        kp1 = RSAKeyPair.generate(512)
        kp2 = RSAKeyPair.generate(512)
        ct = RSA.encrypt(42, kp1.public_key)
        pt = RSA.decrypt(ct, kp2.private_key)
        assert pt != 42  # Should not decrypt correctly

    def test_message_too_large(self, keypair):
        msg = keypair.n + 1  # Larger than modulus
        with pytest.raises(ValueError):
            RSA.encrypt(msg, keypair.public_key)

    def test_key_size(self, keypair):
        assert keypair.key_size >= 500  # ~512, slight variance from prime generation


# ============================================================
# OAEP
# ============================================================

class TestOAEP:
    def test_pad_unpad(self):
        msg = b"test OAEP"
        padded = OAEP.pad(msg, 128)  # 1024-bit key
        assert len(padded) == 128
        assert OAEP.unpad(padded) == msg

    def test_pad_unpad_empty(self):
        padded = OAEP.pad(b"", 128)
        assert OAEP.unpad(padded) == b""

    def test_message_too_long(self):
        with pytest.raises(ValueError):
            OAEP.pad(b"x" * 100, 64)  # Too long for key size

    def test_randomized(self):
        """OAEP padding should be randomized."""
        msg = b"same"
        p1 = OAEP.pad(msg, 128)
        p2 = OAEP.pad(msg, 128)
        assert p1 != p2  # Different random seed
        assert OAEP.unpad(p1) == OAEP.unpad(p2) == msg

    def test_rsa_with_oaep(self):
        kp = RSAKeyPair.generate(1024)
        key_bytes = (kp.key_size + 7) // 8
        msg = b"OAEP + RSA"
        padded = OAEP.pad(msg, key_bytes)
        ct = RSA.encrypt(padded, kp.public_key)
        pt_padded = RSA.decrypt(ct, kp.private_key)
        pt_bytes = pt_padded.to_bytes(key_bytes, 'big')
        assert OAEP.unpad(pt_bytes) == msg


# ============================================================
# Elliptic Curves
# ============================================================

class TestEllipticCurve:
    def test_small_curve_point_on_curve(self):
        # y^2 = x^3 + 2x + 3 (mod 97)
        # Find a point: try x=3: y^2 = 27+6+3 = 36, y=6 or 91
        curve = SMALL_CURVE
        assert curve.is_on_curve((3, 6))
        assert curve.is_on_curve((3, 91))  # 97 - 6 = 91

    def test_point_at_infinity(self):
        assert SMALL_CURVE.is_on_curve(None)

    def test_add_to_infinity(self):
        P = (3, 6)
        assert SMALL_CURVE.add(P, None) == P
        assert SMALL_CURVE.add(None, P) == P

    def test_add_inverse(self):
        P = (3, 6)
        neg_P = (3, 91)  # Negation on curve mod 97
        assert SMALL_CURVE.add(P, neg_P) is None  # Should be point at infinity

    def test_point_doubling(self):
        P = (3, 6)
        R = SMALL_CURVE.add(P, P)
        assert R is not None
        assert SMALL_CURVE.is_on_curve(R)

    def test_scalar_multiply(self):
        P = (3, 6)
        R = SMALL_CURVE.multiply(2, P)
        assert R == SMALL_CURVE.add(P, P)

    def test_scalar_multiply_zero(self):
        assert SMALL_CURVE.multiply(0, (3, 6)) is None

    def test_scalar_multiply_one(self):
        P = (3, 6)
        assert SMALL_CURVE.multiply(1, P) == P

    def test_negate(self):
        P = (3, 6)
        neg = SMALL_CURVE.negate(P)
        assert neg == (3, 91)
        assert SMALL_CURVE.add(P, neg) is None

    def test_negate_infinity(self):
        assert SMALL_CURVE.negate(None) is None

    def test_singular_curve_rejected(self):
        with pytest.raises(ValueError):
            EllipticCurve(0, 0, 97)  # Discriminant 0

    def test_secp256k1_generator_on_curve(self):
        assert SECP256K1.is_on_curve(SECP256K1_G)

    def test_secp256k1_double(self):
        G2 = SECP256K1.add(SECP256K1_G, SECP256K1_G)
        assert G2 is not None
        assert SECP256K1.is_on_curve(G2)

    def test_secp256k1_multiply(self):
        P = SECP256K1.multiply(7, SECP256K1_G)
        assert P is not None
        assert SECP256K1.is_on_curve(P)

    def test_secp256k1_order(self):
        """n*G should be point at infinity."""
        result = SECP256K1.multiply(SECP256K1_N, SECP256K1_G)
        assert result is None

    def test_associativity(self):
        """(P+Q)+R == P+(Q+R)"""
        P = (3, 6)
        Q = SMALL_CURVE.multiply(2, P)
        R = SMALL_CURVE.multiply(3, P)
        lhs = SMALL_CURVE.add(SMALL_CURVE.add(P, Q), R)
        rhs = SMALL_CURVE.add(P, SMALL_CURVE.add(Q, R))
        assert lhs == rhs

    def test_commutativity(self):
        P = (3, 6)
        Q = SMALL_CURVE.multiply(5, P)
        assert SMALL_CURVE.add(P, Q) == SMALL_CURVE.add(Q, P)


# ============================================================
# ECDSA
# ============================================================

class TestECDSA:
    def test_sign_verify(self):
        ecdsa = ECDSA()
        priv, pub = ecdsa.generate_keypair()
        msg = b"Test message for ECDSA"
        sig = ecdsa.sign(msg, priv)
        assert ecdsa.verify(msg, sig, pub)

    def test_wrong_message_fails(self):
        ecdsa = ECDSA()
        priv, pub = ecdsa.generate_keypair()
        sig = ecdsa.sign(b"original", priv)
        assert not ecdsa.verify(b"tampered", sig, pub)

    def test_wrong_key_fails(self):
        ecdsa = ECDSA()
        priv1, _ = ecdsa.generate_keypair()
        _, pub2 = ecdsa.generate_keypair()
        sig = ecdsa.sign(b"message", priv1)
        assert not ecdsa.verify(b"message", sig, pub2)

    def test_deterministic_with_k(self):
        ecdsa = ECDSA()
        priv, pub = ecdsa.generate_keypair()
        k = 12345678901234567890
        sig1 = ecdsa.sign(b"msg", priv, k=k)
        sig2 = ecdsa.sign(b"msg", priv, k=k)
        assert sig1 == sig2

    def test_string_message(self):
        ecdsa = ECDSA()
        priv, pub = ecdsa.generate_keypair()
        sig = ecdsa.sign("string msg", priv)
        assert ecdsa.verify("string msg", sig, pub)

    def test_invalid_signature_range(self):
        ecdsa = ECDSA()
        _, pub = ecdsa.generate_keypair()
        assert not ecdsa.verify(b"msg", (0, 1), pub)
        assert not ecdsa.verify(b"msg", (1, 0), pub)

    def test_public_key_on_curve(self):
        ecdsa = ECDSA()
        _, pub = ecdsa.generate_keypair()
        assert SECP256K1.is_on_curve(pub)


# ============================================================
# ECDH
# ============================================================

class TestECDH:
    def test_shared_secret(self):
        ecdh = ECDH()
        priv_a, pub_a = ecdh.generate_keypair()
        priv_b, pub_b = ecdh.generate_keypair()
        secret_a = ecdh.compute_shared_secret(priv_a, pub_b)
        secret_b = ecdh.compute_shared_secret(priv_b, pub_a)
        assert secret_a == secret_b

    def test_different_pairs_different_secrets(self):
        ecdh = ECDH()
        priv_a, pub_a = ecdh.generate_keypair()
        priv_b, pub_b = ecdh.generate_keypair()
        priv_c, pub_c = ecdh.generate_keypair()
        s_ab = ecdh.compute_shared_secret(priv_a, pub_b)
        s_ac = ecdh.compute_shared_secret(priv_a, pub_c)
        assert s_ab != s_ac

    def test_shared_secret_length(self):
        ecdh = ECDH()
        priv_a, pub_a = ecdh.generate_keypair()
        priv_b, pub_b = ecdh.generate_keypair()
        secret = ecdh.compute_shared_secret(priv_a, pub_b)
        assert len(secret) == 32  # SHA-256 output


# ============================================================
# PBKDF2
# ============================================================

class TestPBKDF2:
    def test_basic_derivation(self):
        key = PBKDF2.derive("password", "salt", iterations=1)
        assert len(key) == 32

    def test_deterministic(self):
        k1 = PBKDF2.derive("pass", "salt", iterations=10)
        k2 = PBKDF2.derive("pass", "salt", iterations=10)
        assert k1 == k2

    def test_different_passwords(self):
        k1 = PBKDF2.derive("pass1", "salt", iterations=10)
        k2 = PBKDF2.derive("pass2", "salt", iterations=10)
        assert k1 != k2

    def test_different_salts(self):
        k1 = PBKDF2.derive("pass", "salt1", iterations=10)
        k2 = PBKDF2.derive("pass", "salt2", iterations=10)
        assert k1 != k2

    def test_more_iterations_different(self):
        k1 = PBKDF2.derive("pass", "salt", iterations=1)
        k2 = PBKDF2.derive("pass", "salt", iterations=2)
        assert k1 != k2

    def test_custom_key_length(self):
        key = PBKDF2.derive("pass", "salt", iterations=1, key_length=16)
        assert len(key) == 16

    # RFC 6070 test vector 1 (PBKDF2-HMAC-SHA256 is not in RFC 6070,
    # but we test determinism and properties)
    def test_rfc_style(self):
        key = PBKDF2.derive(b"password", b"salt", iterations=1, key_length=32)
        assert len(key) == 32
        # Verify it's deterministic
        key2 = PBKDF2.derive(b"password", b"salt", iterations=1, key_length=32)
        assert key == key2


# ============================================================
# ChaCha20
# ============================================================

class TestChaCha20:
    def test_encrypt_decrypt(self):
        key = bytes(range(32))
        nonce = bytes(range(12))
        msg = b"ChaCha20 test message!"
        ct = ChaCha20(key, nonce).encrypt(msg)
        pt = ChaCha20(key, nonce).decrypt(ct)
        assert pt == msg

    def test_string_input(self):
        key = bytes(32)
        nonce = bytes(12)
        ct = ChaCha20(key, nonce).encrypt("hello")
        assert ChaCha20(key, nonce).decrypt(ct) == b"hello"

    def test_stream_property(self):
        """Ciphertext length equals plaintext length."""
        key = bytes(32)
        nonce = bytes(12)
        for length in [1, 15, 16, 17, 63, 64, 65, 100]:
            msg = os.urandom(length)
            ct = ChaCha20(key, nonce).encrypt(msg)
            assert len(ct) == length

    def test_different_nonce(self):
        key = bytes(32)
        msg = b"same plaintext"
        c1 = ChaCha20(key, bytes(12)).encrypt(msg)
        c2 = ChaCha20(key, b"\x01" + bytes(11)).encrypt(msg)
        assert c1 != c2

    def test_different_key(self):
        nonce = bytes(12)
        msg = b"same plaintext"
        c1 = ChaCha20(bytes(32), nonce).encrypt(msg)
        c2 = ChaCha20(b"\x01" + bytes(31), nonce).encrypt(msg)
        assert c1 != c2

    def test_counter(self):
        key = bytes(32)
        nonce = bytes(12)
        msg = b"x" * 128
        c1 = ChaCha20(key, nonce, counter=0).encrypt(msg)
        c2 = ChaCha20(key, nonce, counter=1).encrypt(msg)
        assert c1 != c2

    def test_invalid_key_size(self):
        with pytest.raises(ValueError):
            ChaCha20(bytes(16), bytes(12))

    def test_invalid_nonce_size(self):
        with pytest.raises(ValueError):
            ChaCha20(bytes(32), bytes(8))

    # RFC 7539 Section 2.4.2 test vector
    def test_rfc7539_vector(self):
        key = bytes([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
            0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
        ])
        nonce = bytes([
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4a,
            0x00, 0x00, 0x00, 0x00,
        ])
        counter = 1
        plaintext = (
            b"Ladies and Gentlemen of the class of '99: "
            b"If I could offer you only one tip for the future, sunscreen would be it."
        )
        expected_ct = bytes([
            0x6e, 0x2e, 0x35, 0x9a, 0x25, 0x68, 0xf9, 0x80,
            0x41, 0xba, 0x07, 0x28, 0xdd, 0x0d, 0x69, 0x81,
            0xe9, 0x7e, 0x7a, 0xec, 0x1d, 0x43, 0x60, 0xc2,
            0x0a, 0x27, 0xaf, 0xcc, 0xfd, 0x9f, 0xae, 0x0b,
            0xf9, 0x1b, 0x65, 0xc5, 0x52, 0x47, 0x33, 0xab,
            0x8f, 0x59, 0x3d, 0xab, 0xcd, 0x62, 0xb3, 0x57,
            0x16, 0x39, 0xd6, 0x24, 0xe6, 0x51, 0x52, 0xab,
            0x8f, 0x53, 0x0c, 0x35, 0x9f, 0x08, 0x61, 0xd8,
            0x07, 0xca, 0x0d, 0xbf, 0x50, 0x0d, 0x6a, 0x61,
            0x56, 0xa3, 0x8e, 0x08, 0x8a, 0x22, 0xb6, 0x5e,
            0x52, 0xbc, 0x51, 0x4d, 0x16, 0xcc, 0xf8, 0x06,
            0x81, 0x8c, 0xe9, 0x1a, 0xb7, 0x79, 0x37, 0x36,
            0x5a, 0xf9, 0x0b, 0xbf, 0x74, 0xa3, 0x5b, 0xe6,
            0xb4, 0x0b, 0x8e, 0xed, 0xf2, 0x78, 0x5e, 0x42,
            0x87, 0x4d,
        ])
        ct = ChaCha20(key, nonce, counter=counter).encrypt(plaintext)
        assert ct == expected_ct


# ============================================================
# Poly1305
# ============================================================

class TestPoly1305:
    def test_basic(self):
        key = os.urandom(32)
        mac = Poly1305.authenticate(key, b"test message")
        assert len(mac) == 16

    def test_deterministic(self):
        key = bytes(range(32))
        m1 = Poly1305.authenticate(key, b"msg")
        m2 = Poly1305.authenticate(key, b"msg")
        assert m1 == m2

    def test_different_keys(self):
        m1 = Poly1305.authenticate(bytes(32), b"msg")
        m2 = Poly1305.authenticate(b"\x01" + bytes(31), b"msg")
        assert m1 != m2

    def test_different_messages(self):
        key = bytes(range(32))
        m1 = Poly1305.authenticate(key, b"msg1")
        m2 = Poly1305.authenticate(key, b"msg2")
        assert m1 != m2

    def test_invalid_key_size(self):
        with pytest.raises(ValueError):
            Poly1305(bytes(16))

    # RFC 7539 Section 2.5.2 test vector
    def test_rfc7539_vector(self):
        key = bytes([
            0x85, 0xd6, 0xbe, 0x78, 0x57, 0x55, 0x6d, 0x33,
            0x7f, 0x44, 0x52, 0xfe, 0x42, 0xd5, 0x06, 0xa8,
            0x01, 0x03, 0x80, 0x8a, 0xfb, 0x0d, 0xb2, 0xfd,
            0x4a, 0xbf, 0xf6, 0xaf, 0x41, 0x49, 0xf5, 0x1b,
        ])
        msg = b"Cryptographic Forum Research Group"
        expected = bytes([
            0xa8, 0x06, 0x1d, 0xc1, 0x30, 0x51, 0x36, 0xc6,
            0xc2, 0x2b, 0x8b, 0xaf, 0x0c, 0x01, 0x27, 0xa9,
        ])
        assert Poly1305.authenticate(key, msg) == expected


# ============================================================
# Integration / Cross-Component Tests
# ============================================================

class TestIntegration:
    def test_aes_cbc_with_pbkdf2_key(self):
        """Derive key from password, encrypt with AES-CBC."""
        key = PBKDF2.derive("my password", "salt123", iterations=100, key_length=16)
        iv = bytes(16)
        msg = b"Secret message encrypted with derived key"
        ct = AES_CBC(key, iv).encrypt(msg)
        pt = AES_CBC(key, iv).decrypt(ct)
        assert pt == msg

    def test_rsa_sign_verify_roundtrip(self):
        """Full RSA sign-verify cycle with SHA-256."""
        kp = RSAKeyPair.generate(1024)
        for msg in [b"short", b"x" * 1000, b""]:
            sig = RSA.sign(msg, kp.private_key)
            assert RSA.verify(msg, sig, kp.public_key)

    def test_ecdsa_multiple_messages(self):
        """Sign and verify multiple messages with same key."""
        ecdsa = ECDSA()
        priv, pub = ecdsa.generate_keypair()
        messages = [b"msg1", b"msg2", b"msg3", b"" * 100]
        for msg in messages:
            sig = ecdsa.sign(msg, priv)
            assert ecdsa.verify(msg, sig, pub)

    def test_chacha20_large_message(self):
        """Encrypt a large message with ChaCha20."""
        key = os.urandom(32)
        nonce = os.urandom(12)
        msg = os.urandom(10000)
        ct = ChaCha20(key, nonce).encrypt(msg)
        pt = ChaCha20(key, nonce).decrypt(ct)
        assert pt == msg

    def test_hmac_verify_integrity(self):
        """Use HMAC to verify message integrity."""
        key = os.urandom(32)
        msg = b"important data"
        tag = HMAC.mac(key, msg)
        # Verify
        assert HMAC.mac(key, msg) == tag
        # Tampered message fails
        assert HMAC.mac(key, b"tampered data") != tag

    def test_encrypt_then_mac(self):
        """Encrypt-then-MAC pattern."""
        enc_key = os.urandom(16)
        mac_key = os.urandom(32)
        nonce = bytes(8)
        msg = b"authenticated encryption"
        ct = AES_CTR(enc_key, nonce).encrypt(msg)
        tag = HMAC.mac(mac_key, ct)
        # Verify and decrypt
        assert HMAC.mac(mac_key, ct) == tag
        pt = AES_CTR(enc_key, nonce).decrypt(ct)
        assert pt == msg

    def test_ecdh_aes_key_agreement(self):
        """ECDH key exchange then AES encryption."""
        ecdh = ECDH()
        priv_a, pub_a = ecdh.generate_keypair()
        priv_b, pub_b = ecdh.generate_keypair()
        # Both derive same shared secret
        shared = ecdh.compute_shared_secret(priv_a, pub_b)
        # Use first 16 bytes as AES key
        aes_key = shared[:16]
        nonce = bytes(8)
        msg = b"Secure channel established!"
        ct = AES_CTR(aes_key, nonce).encrypt(msg)
        pt = AES_CTR(aes_key, nonce).decrypt(ct)
        assert pt == msg


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_sha256_empty(self):
        assert len(SHA256.hash(b"")) == 32

    def test_aes_all_zeros(self):
        aes = AES(bytes(16))
        ct = aes.encrypt_block(bytes(16))
        assert aes.decrypt_block(ct) == bytes(16)

    def test_aes_all_ones(self):
        aes = AES(bytes([0xff] * 16))
        ct = aes.encrypt_block(bytes([0xff] * 16))
        assert aes.decrypt_block(ct) == bytes([0xff] * 16)

    def test_chacha20_empty(self):
        key = bytes(32)
        nonce = bytes(12)
        assert ChaCha20(key, nonce).encrypt(b"") == b""

    def test_rsa_encrypt_zero(self):
        kp = RSAKeyPair.generate(512)
        ct = RSA.encrypt(0, kp.public_key)
        assert RSA.decrypt(ct, kp.private_key) == 0

    def test_rsa_encrypt_one(self):
        kp = RSAKeyPair.generate(512)
        ct = RSA.encrypt(1, kp.public_key)
        assert RSA.decrypt(ct, kp.private_key) == 1

    def test_ec_multiply_large_scalar(self):
        P = (3, 6)
        # k*P where k > order should wrap around
        R = SMALL_CURVE.multiply(1000, P)
        assert R is None or SMALL_CURVE.is_on_curve(R)

    def test_gcd_negative(self):
        assert gcd(-12, 8) == 4
        assert gcd(12, -8) == 4
