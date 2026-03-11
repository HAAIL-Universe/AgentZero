"""Tests for C110: Wavelet Tree"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from wavelet_tree import (
    BitVector, WaveletTree, WaveletMatrix, HuffmanWaveletTree,
    RangeWaveletTree, wavelet_tree_from_string, wavelet_matrix_from_string
)


# ============================================================
# BitVector tests
# ============================================================

class TestBitVector:
    def test_rank1_basic(self):
        bv = BitVector([1, 0, 1, 1, 0, 0, 1])
        assert bv.rank1(0) == 0
        assert bv.rank1(1) == 1
        assert bv.rank1(3) == 2
        assert bv.rank1(4) == 3
        assert bv.rank1(7) == 4

    def test_rank0_basic(self):
        bv = BitVector([1, 0, 1, 1, 0, 0, 1])
        assert bv.rank0(0) == 0
        assert bv.rank0(1) == 0
        assert bv.rank0(2) == 1
        assert bv.rank0(5) == 2
        assert bv.rank0(7) == 3

    def test_select1(self):
        bv = BitVector([1, 0, 1, 1, 0, 0, 1])
        assert bv.select1(1) == 0
        assert bv.select1(2) == 2
        assert bv.select1(3) == 3
        assert bv.select1(4) == 6
        assert bv.select1(5) == -1

    def test_select0(self):
        bv = BitVector([1, 0, 1, 1, 0, 0, 1])
        assert bv.select0(1) == 1
        assert bv.select0(2) == 4
        assert bv.select0(3) == 5
        assert bv.select0(4) == -1

    def test_empty(self):
        bv = BitVector([])
        assert bv.rank1(0) == 0
        assert bv.rank0(0) == 0
        assert bv.select1(1) == -1
        assert bv.select0(1) == -1

    def test_all_ones(self):
        bv = BitVector([1, 1, 1, 1])
        assert bv.rank1(4) == 4
        assert bv.rank0(4) == 0
        assert bv.select1(3) == 2
        assert bv.select0(1) == -1

    def test_all_zeros(self):
        bv = BitVector([0, 0, 0])
        assert bv.rank1(3) == 0
        assert bv.rank0(3) == 3
        assert bv.select0(2) == 1
        assert bv.select1(1) == -1

    def test_access(self):
        bv = BitVector([1, 0, 1, 0])
        assert bv.access(0) == 1
        assert bv.access(1) == 0
        assert bv.access(2) == 1
        assert bv.access(3) == 0


# ============================================================
# WaveletTree tests
# ============================================================

class TestWaveletTree:
    def setup_method(self):
        self.seq = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        self.wt = WaveletTree(self.seq)

    def test_access_all(self):
        for i, v in enumerate(self.seq):
            assert self.wt.access(i) == v

    def test_access_out_of_range(self):
        with pytest.raises(IndexError):
            self.wt.access(len(self.seq))
        with pytest.raises(IndexError):
            self.wt.access(-1)

    def test_rank_basic(self):
        # rank(1, 4) = count of 1 in seq[0..4) = [3,1,4,1] = 1
        # Wait: seq[0..4) = [3,1,4,1], count of 1 = 2
        assert self.wt.rank(1, 4) == 2
        assert self.wt.rank(5, 11) == 3
        assert self.wt.rank(9, 11) == 1
        assert self.wt.rank(7, 11) == 0  # 7 not in seq

    def test_rank_zero_pos(self):
        assert self.wt.rank(3, 0) == 0

    def test_rank_full(self):
        for v in set(self.seq):
            assert self.wt.rank(v, len(self.seq)) == self.seq.count(v)

    def test_select_basic(self):
        # select(1, 1) = position of 1st occurrence of 1 = index 1
        assert self.wt.select(1, 1) == 1
        assert self.wt.select(1, 2) == 3
        assert self.wt.select(5, 1) == 4
        assert self.wt.select(5, 3) == 10
        assert self.wt.select(9, 1) == 5

    def test_select_not_found(self):
        assert self.wt.select(7, 1) == -1
        assert self.wt.select(1, 3) == -1  # only 2 ones

    def test_quantile(self):
        # seq = [3,1,4,1,5,9,2,6,5,3,5]
        # sorted: [1,1,2,3,3,4,5,5,5,6,9]
        assert self.wt.quantile(0, 11, 0) == 1
        assert self.wt.quantile(0, 11, 1) == 1
        assert self.wt.quantile(0, 11, 2) == 2
        assert self.wt.quantile(0, 11, 5) == 4
        assert self.wt.quantile(0, 11, 10) == 9

    def test_quantile_subrange(self):
        # seq[2..7) = [4,1,5,9,2], sorted = [1,2,4,5,9]
        assert self.wt.quantile(2, 7, 0) == 1
        assert self.wt.quantile(2, 7, 2) == 4
        assert self.wt.quantile(2, 7, 4) == 9

    def test_quantile_invalid(self):
        with pytest.raises(ValueError):
            self.wt.quantile(0, 11, 11)
        with pytest.raises(ValueError):
            self.wt.quantile(5, 5, 0)

    def test_range_freq(self):
        # Count elements in [0,11) with 2 <= val < 6
        # seq: [3,1,4,1,5,9,2,6,5,3,5] -> {3,4,5,2,5,3,5} = 7
        assert self.wt.range_freq(0, 11, 2, 6) == 7

    def test_range_freq_subrange(self):
        # seq[0..5) = [3,1,4,1,5], 1 <= val < 4 -> {3,1,1} = 3
        assert self.wt.range_freq(0, 5, 1, 4) == 3

    def test_range_freq_empty(self):
        assert self.wt.range_freq(0, 11, 10, 20) == 0
        assert self.wt.range_freq(3, 3, 0, 10) == 0

    def test_range_list(self):
        result = self.wt.range_list(0, 11, 1, 6)
        result_dict = dict(result)
        assert result_dict[1] == 2
        assert result_dict[3] == 2
        assert result_dict[5] == 3
        assert result_dict[4] == 1
        assert result_dict[2] == 1

    def test_top_k(self):
        result = self.wt.top_k(0, 11, 2)
        assert result[0] == (5, 3)
        assert result[1][1] == 2  # 1 or 3 with freq 2

    def test_prev_value(self):
        assert self.wt.prev_value(0, 11, 5) == 4
        assert self.wt.prev_value(0, 11, 1) is None
        assert self.wt.prev_value(0, 11, 10) == 9

    def test_next_value(self):
        assert self.wt.next_value(0, 11, 5) == 6
        assert self.wt.next_value(0, 11, 9) is None
        assert self.wt.next_value(0, 11, 0) == 1

    def test_swap(self):
        self.wt.swap(0, 1)
        assert self.wt.access(0) == 1
        assert self.wt.access(1) == 3

    def test_empty_tree(self):
        wt = WaveletTree([])
        assert wt.n == 0
        assert wt.rank(1, 0) == 0
        assert wt.select(1, 1) == -1

    def test_single_element(self):
        wt = WaveletTree([42])
        assert wt.access(0) == 42
        assert wt.rank(42, 1) == 1
        assert wt.select(42, 1) == 0

    def test_uniform_sequence(self):
        wt = WaveletTree([5, 5, 5, 5])
        assert wt.access(2) == 5
        assert wt.rank(5, 4) == 4
        assert wt.select(5, 3) == 2
        assert wt.quantile(0, 4, 2) == 5

    def test_two_element_alphabet(self):
        wt = WaveletTree([0, 1, 0, 1, 0])
        assert wt.rank(0, 5) == 3
        assert wt.rank(1, 5) == 2
        assert wt.select(0, 2) == 2
        assert wt.select(1, 2) == 3

    def test_rank_select_consistency(self):
        """rank and select are inverses."""
        for sym in set(self.seq):
            count = 0
            for i in range(self.wt.n):
                if self.seq[i] == sym:
                    count += 1
                    pos = self.wt.select(sym, count)
                    assert pos == i
                    assert self.wt.rank(sym, pos + 1) == count

    def test_large_sequence(self):
        import random
        random.seed(42)
        seq = [random.randint(0, 20) for _ in range(200)]
        wt = WaveletTree(seq)
        for i in range(len(seq)):
            assert wt.access(i) == seq[i]
        for sym in set(seq):
            assert wt.rank(sym, len(seq)) == seq.count(sym)


# ============================================================
# WaveletMatrix tests
# ============================================================

class TestWaveletMatrix:
    def setup_method(self):
        self.seq = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        self.wm = WaveletMatrix(self.seq)

    def test_access_all(self):
        for i, v in enumerate(self.seq):
            assert self.wm.access(i) == v

    def test_access_out_of_range(self):
        with pytest.raises(IndexError):
            self.wm.access(len(self.seq))

    def test_rank_basic(self):
        assert self.wm.rank(1, 4) == 2
        assert self.wm.rank(5, 11) == 3
        assert self.wm.rank(9, 11) == 1
        assert self.wm.rank(7, 11) == 0

    def test_rank_full(self):
        for v in set(self.seq):
            assert self.wm.rank(v, len(self.seq)) == self.seq.count(v)

    def test_select_basic(self):
        assert self.wm.select(1, 1) == 1
        assert self.wm.select(1, 2) == 3
        assert self.wm.select(5, 1) == 4
        assert self.wm.select(9, 1) == 5

    def test_select_not_found(self):
        assert self.wm.select(7, 1) == -1
        assert self.wm.select(1, 3) == -1

    def test_quantile(self):
        assert self.wm.quantile(0, 11, 0) == 1
        assert self.wm.quantile(0, 11, 2) == 2
        assert self.wm.quantile(0, 11, 10) == 9

    def test_quantile_subrange(self):
        assert self.wm.quantile(2, 7, 0) == 1
        assert self.wm.quantile(2, 7, 4) == 9

    def test_range_freq(self):
        assert self.wm.range_freq(0, 11, 2, 6) == 7

    def test_range_freq_subrange(self):
        assert self.wm.range_freq(0, 5, 1, 4) == 3

    def test_empty(self):
        wm = WaveletMatrix([])
        assert wm.rank(1, 0) == 0
        assert wm.select(1, 1) == -1

    def test_single(self):
        wm = WaveletMatrix([42])
        assert wm.access(0) == 42
        assert wm.rank(42, 1) == 1
        assert wm.select(42, 1) == 0

    def test_uniform(self):
        wm = WaveletMatrix([7, 7, 7])
        assert wm.access(1) == 7
        assert wm.rank(7, 3) == 3
        assert wm.select(7, 2) == 1
        assert wm.quantile(0, 3, 1) == 7

    def test_negative_values(self):
        wm = WaveletMatrix([-3, -1, -2, 0, 1])
        for i, v in enumerate([-3, -1, -2, 0, 1]):
            assert wm.access(i) == v
        assert wm.rank(-1, 5) == 1
        assert wm.rank(-3, 5) == 1
        assert wm.select(-2, 1) == 2
        assert wm.quantile(0, 5, 0) == -3

    def test_rank_select_consistency(self):
        for sym in set(self.seq):
            count = 0
            for i in range(self.wm.n):
                if self.seq[i] == sym:
                    count += 1
                    pos = self.wm.select(sym, count)
                    assert pos == i
                    assert self.wm.rank(sym, pos + 1) == count

    def test_large_random(self):
        import random
        random.seed(123)
        seq = [random.randint(0, 50) for _ in range(300)]
        wm = WaveletMatrix(seq)
        for i in range(len(seq)):
            assert wm.access(i) == seq[i]
        for sym in set(seq):
            assert wm.rank(sym, len(seq)) == seq.count(sym)

    def test_range_freq_boundary(self):
        assert self.wm.range_freq(0, 11, 1, 10) == 11
        assert self.wm.range_freq(0, 11, 10, 20) == 0


# ============================================================
# HuffmanWaveletTree tests
# ============================================================

class TestHuffmanWaveletTree:
    def setup_method(self):
        self.seq = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        self.hwt = HuffmanWaveletTree(self.seq)

    def test_access_all(self):
        for i, v in enumerate(self.seq):
            assert self.hwt.access(i) == v

    def test_access_out_of_range(self):
        with pytest.raises(IndexError):
            self.hwt.access(len(self.seq))

    def test_rank_basic(self):
        assert self.hwt.rank(1, 4) == 2
        assert self.hwt.rank(5, 11) == 3
        assert self.hwt.rank(9, 11) == 1

    def test_rank_missing(self):
        assert self.hwt.rank(7, 11) == 0

    def test_rank_full(self):
        for v in set(self.seq):
            assert self.hwt.rank(v, len(self.seq)) == self.seq.count(v)

    def test_select_basic(self):
        assert self.hwt.select(1, 1) == 1
        assert self.hwt.select(1, 2) == 3
        assert self.hwt.select(5, 1) == 4
        assert self.hwt.select(9, 1) == 5

    def test_select_not_found(self):
        assert self.hwt.select(7, 1) == -1
        assert self.hwt.select(1, 3) == -1

    def test_empty(self):
        hwt = HuffmanWaveletTree([])
        assert hwt.rank(1, 0) == 0
        assert hwt.select(1, 1) == -1

    def test_single(self):
        hwt = HuffmanWaveletTree([99])
        assert hwt.access(0) == 99
        assert hwt.rank(99, 1) == 1
        assert hwt.select(99, 1) == 0

    def test_skewed_alphabet(self):
        # Many a's, few b's -- Huffman should give shorter codes to 'a'
        seq = [1]*100 + [2]*10 + [3]*1
        hwt = HuffmanWaveletTree(seq)
        for i, v in enumerate(seq):
            assert hwt.access(i) == v
        assert hwt.rank(1, 111) == 100
        assert hwt.rank(2, 111) == 10
        assert hwt.rank(3, 111) == 1
        # Huffman should give 1-bit code to most frequent
        assert hwt.codes[1][1] == 1  # 1 bit for symbol 1

    def test_avg_bits(self):
        seq = [1]*100 + [2]*10 + [3]*1
        hwt = HuffmanWaveletTree(seq)
        # Should be less than log2(3) ~= 1.58 for skewed distribution
        assert hwt.avg_bits_per_symbol() < 1.5

    def test_rank_select_consistency(self):
        for sym in set(self.seq):
            count = 0
            for i in range(self.hwt.n):
                if self.seq[i] == sym:
                    count += 1
                    pos = self.hwt.select(sym, count)
                    assert pos == i
                    assert self.hwt.rank(sym, pos + 1) == count

    def test_two_symbols(self):
        seq = [10, 20, 10, 20, 10]
        hwt = HuffmanWaveletTree(seq)
        assert hwt.rank(10, 5) == 3
        assert hwt.rank(20, 5) == 2
        assert hwt.select(10, 2) == 2
        assert hwt.select(20, 1) == 1


# ============================================================
# RangeWaveletTree tests
# ============================================================

class TestRangeWaveletTree:
    def test_count_points(self):
        points = [(1, 3), (2, 1), (3, 4), (4, 1), (5, 5)]
        rwt = RangeWaveletTree(points)
        # All points in [1,5] x [1,5]
        assert rwt.count_points(1, 5, 1, 5) == 5
        # Points with y >= 3
        assert rwt.count_points(1, 5, 3, 5) == 3
        # Points with x in [2,4], y in [1,2]
        assert rwt.count_points(2, 4, 1, 2) == 2

    def test_report_points(self):
        points = [(1, 3), (2, 1), (3, 4), (4, 1), (5, 5)]
        rwt = RangeWaveletTree(points)
        result = rwt.report_points(2, 4, 1, 2)
        assert set(result) == {(2, 1), (4, 1)}

    def test_kth_by_y(self):
        points = [(1, 10), (2, 5), (3, 8), (4, 3), (5, 7)]
        rwt = RangeWaveletTree(points)
        assert rwt.kth_point_by_y(1, 5, 0) == 3   # smallest y
        assert rwt.kth_point_by_y(1, 5, 4) == 10  # largest y

    def test_empty_range(self):
        points = [(1, 1), (5, 5)]
        rwt = RangeWaveletTree(points)
        assert rwt.count_points(2, 4, 1, 5) == 0

    def test_empty_tree(self):
        rwt = RangeWaveletTree([])
        assert rwt.count_points(0, 10, 0, 10) == 0

    def test_single_point(self):
        rwt = RangeWaveletTree([(3, 7)])
        assert rwt.count_points(3, 3, 7, 7) == 1
        assert rwt.count_points(0, 2, 0, 10) == 0

    def test_duplicate_x(self):
        points = [(1, 1), (1, 2), (1, 3)]
        rwt = RangeWaveletTree(points)
        assert rwt.count_points(1, 1, 1, 3) == 3
        assert rwt.count_points(1, 1, 2, 2) == 1

    def test_large_grid(self):
        import random
        random.seed(99)
        points = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(50)]
        rwt = RangeWaveletTree(points)
        # Brute-force verify
        x1, x2, y1, y2 = 20, 60, 30, 70
        expected = sum(1 for x, y in points if x1 <= x <= x2 and y1 <= y <= y2)
        assert rwt.count_points(x1, x2, y1, y2) == expected


# ============================================================
# String wavelet tree tests
# ============================================================

class TestStringWaveletTree:
    def test_from_string(self):
        wt = wavelet_tree_from_string("abracadabra")
        assert wt.access(0) == ord('a')
        assert wt.rank(ord('a'), 11) == 5
        assert wt.rank(ord('b'), 11) == 2

    def test_from_string_matrix(self):
        wm = wavelet_matrix_from_string("abracadabra")
        assert wm.access(0) == ord('a')
        assert wm.rank(ord('a'), 11) == 5

    def test_select_char(self):
        wt = wavelet_tree_from_string("mississippi")
        # 's' = 115, positions: 2,3,5,6
        assert wt.select(ord('s'), 1) == 2
        assert wt.select(ord('s'), 4) == 6

    def test_quantile_string(self):
        wt = wavelet_tree_from_string("dcba")
        # sorted by ord: a,b,c,d
        assert wt.quantile(0, 4, 0) == ord('a')
        assert wt.quantile(0, 4, 3) == ord('d')


# ============================================================
# Cross-variant consistency tests
# ============================================================

class TestCrossVariant:
    def test_all_variants_agree_on_rank(self):
        seq = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        wt = WaveletTree(seq)
        wm = WaveletMatrix(seq)
        hwt = HuffmanWaveletTree(seq)
        for sym in set(seq):
            for pos in range(len(seq) + 1):
                r_wt = wt.rank(sym, pos)
                r_wm = wm.rank(sym, pos)
                r_hwt = hwt.rank(sym, pos)
                assert r_wt == r_wm == r_hwt, f"rank({sym}, {pos}): wt={r_wt} wm={r_wm} hwt={r_hwt}"

    def test_all_variants_agree_on_select(self):
        seq = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        wt = WaveletTree(seq)
        wm = WaveletMatrix(seq)
        hwt = HuffmanWaveletTree(seq)
        for sym in set(seq):
            for k in range(1, seq.count(sym) + 2):
                s_wt = wt.select(sym, k)
                s_wm = wm.select(sym, k)
                s_hwt = hwt.select(sym, k)
                assert s_wt == s_wm == s_hwt, f"select({sym}, {k}): wt={s_wt} wm={s_wm} hwt={s_hwt}"

    def test_all_variants_agree_on_access(self):
        seq = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        wt = WaveletTree(seq)
        wm = WaveletMatrix(seq)
        hwt = HuffmanWaveletTree(seq)
        for i in range(len(seq)):
            assert wt.access(i) == wm.access(i) == hwt.access(i)

    def test_wt_wm_agree_on_quantile(self):
        seq = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        wt = WaveletTree(seq)
        wm = WaveletMatrix(seq)
        for l in range(len(seq)):
            for r in range(l + 1, min(l + 5, len(seq) + 1)):
                for k in range(r - l):
                    q_wt = wt.quantile(l, r, k)
                    q_wm = wm.quantile(l, r, k)
                    assert q_wt == q_wm, f"quantile({l},{r},{k}): wt={q_wt} wm={q_wm}"

    def test_wt_wm_agree_on_range_freq(self):
        seq = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        wt = WaveletTree(seq)
        wm = WaveletMatrix(seq)
        for l in range(len(seq)):
            for r in range(l + 1, min(l + 4, len(seq) + 1)):
                for lo in range(1, 10):
                    for hi in range(lo + 1, 11):
                        f_wt = wt.range_freq(l, r, lo, hi)
                        f_wm = wm.range_freq(l, r, lo, hi)
                        assert f_wt == f_wm, f"range_freq({l},{r},{lo},{hi}): wt={f_wt} wm={f_wm}"


# ============================================================
# Edge cases and stress tests
# ============================================================

class TestEdgeCases:
    def test_large_alphabet(self):
        seq = list(range(100))
        wt = WaveletTree(seq)
        for i in range(100):
            assert wt.access(i) == i
            assert wt.rank(i, 100) == 1
            assert wt.select(i, 1) == i

    def test_large_alphabet_matrix(self):
        seq = list(range(100))
        wm = WaveletMatrix(seq)
        for i in range(100):
            assert wm.access(i) == i
            assert wm.rank(i, 100) == 1
            assert wm.select(i, 1) == i

    def test_repeated_element(self):
        seq = [42] * 50
        wt = WaveletTree(seq)
        assert wt.rank(42, 25) == 25
        assert wt.select(42, 10) == 9
        assert wt.quantile(0, 50, 25) == 42

    def test_two_distinct(self):
        seq = [0, 1] * 50
        wt = WaveletTree(seq)
        wm = WaveletMatrix(seq)
        assert wt.rank(0, 100) == 50
        assert wm.rank(1, 100) == 50
        assert wt.select(0, 25) == 48
        assert wm.select(1, 25) == 49

    def test_descending(self):
        seq = list(range(20, 0, -1))
        wt = WaveletTree(seq)
        for i in range(20):
            assert wt.access(i) == 20 - i
        assert wt.quantile(0, 20, 0) == 1
        assert wt.quantile(0, 20, 19) == 20

    def test_random_stress(self):
        import random
        random.seed(7)
        seq = [random.randint(0, 10) for _ in range(500)]
        wt = WaveletTree(seq)
        wm = WaveletMatrix(seq)
        # Spot-check rank
        for _ in range(50):
            sym = random.choice(seq)
            pos = random.randint(0, len(seq))
            expected = sum(1 for v in seq[:pos] if v == sym)
            assert wt.rank(sym, pos) == expected
            assert wm.rank(sym, pos) == expected

    def test_quantile_median(self):
        seq = [5, 3, 8, 1, 9, 2, 7]
        wt = WaveletTree(seq)
        # sorted: [1,2,3,5,7,8,9], median (k=3) = 5
        assert wt.quantile(0, 7, 3) == 5

    def test_range_list_empty(self):
        wt = WaveletTree([1, 2, 3])
        result = wt.range_list(0, 3, 10, 20)
        assert result == []

    def test_prev_next_single(self):
        wt = WaveletTree([5])
        assert wt.prev_value(0, 1, 5) is None
        assert wt.next_value(0, 1, 5) is None
        assert wt.prev_value(0, 1, 6) == 5
        assert wt.next_value(0, 1, 4) == 5

    def test_matrix_range_freq_exact(self):
        seq = [1, 2, 3, 4, 5]
        wm = WaveletMatrix(seq)
        assert wm.range_freq(0, 5, 2, 5) == 3  # 2,3,4
        assert wm.range_freq(0, 5, 1, 6) == 5  # all

    def test_quantile_all_same_subrange(self):
        wt = WaveletTree([3, 3, 3, 7, 7])
        assert wt.quantile(0, 3, 0) == 3
        assert wt.quantile(0, 3, 2) == 3
        assert wt.quantile(3, 5, 0) == 7


# ============================================================
# Additional coverage tests
# ============================================================

class TestAdditional:
    def test_rank_beyond_length(self):
        wt = WaveletTree([1, 2, 3])
        assert wt.rank(1, 100) == 1  # clamped to n

    def test_select_zero_k(self):
        wt = WaveletTree([1, 2, 3])
        assert wt.select(1, 0) == -1

    def test_range_freq_inverted(self):
        wt = WaveletTree([1, 2, 3])
        assert wt.range_freq(0, 3, 5, 2) == 0  # lo >= hi

    def test_huffman_single_symbol(self):
        hwt = HuffmanWaveletTree([4, 4, 4])
        assert hwt.access(0) == 4
        assert hwt.rank(4, 3) == 3
        assert hwt.select(4, 2) == 1

    def test_matrix_select_last(self):
        wm = WaveletMatrix([1, 2, 1, 2, 1])
        assert wm.select(1, 3) == 4
        assert wm.select(2, 2) == 3

    def test_range_wavelet_kth_invalid(self):
        rwt = RangeWaveletTree([(1, 1)])
        with pytest.raises(ValueError):
            rwt.kth_point_by_y(1, 1, 1)  # only 1 point, k=1 is out of range

    def test_wt_range_list_full(self):
        seq = [1, 1, 2, 2, 3]
        wt = WaveletTree(seq)
        result = dict(wt.range_list(0, 5, 1, 4))
        assert result == {1: 2, 2: 2, 3: 1}

    def test_matrix_quantile_invalid(self):
        wm = WaveletMatrix([1, 2, 3])
        with pytest.raises(ValueError):
            wm.quantile(0, 3, 3)

    def test_huffman_many_symbols(self):
        seq = list(range(50)) + list(range(50))
        hwt = HuffmanWaveletTree(seq)
        for i in range(len(seq)):
            assert hwt.access(i) == seq[i]
        for sym in range(50):
            assert hwt.rank(sym, 100) == 2

    def test_top_k_all(self):
        wt = WaveletTree([1, 2, 3])
        result = wt.top_k(0, 3, 10)
        assert len(result) == 3
        assert all(f == 1 for _, f in result)

    def test_prev_value_at_boundary(self):
        wt = WaveletTree([1, 5, 10])
        assert wt.prev_value(0, 3, 10) == 5
        assert wt.prev_value(0, 3, 5) == 1

    def test_next_value_at_boundary(self):
        wt = WaveletTree([1, 5, 10])
        assert wt.next_value(0, 3, 1) == 5
        assert wt.next_value(0, 3, 5) == 10

    def test_swap_and_query(self):
        wt = WaveletTree([1, 2, 3])
        wt.swap(0, 2)
        assert wt.access(0) == 3
        assert wt.access(2) == 1
        assert wt.rank(3, 3) == 1
        assert wt.rank(1, 3) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
