"""
C110: Wavelet Tree

A wavelet tree is a succinct data structure for sequences over an alphabet,
supporting rank, select, quantile, range counting, and range frequency queries
in O(log sigma) time where sigma is the alphabet size.

Variants:
1. WaveletTree -- basic wavelet tree over integer alphabet
2. WaveletMatrix -- space-efficient variant using level-wise bitvectors
3. HuffmanWaveletTree -- alphabet-adaptive using Huffman-shaped tree
4. RangeWaveletTree -- 2D range queries (points in rectangles)

All variants support:
- rank(symbol, pos) -- count occurrences of symbol in seq[0..pos)
- select(symbol, k) -- position of k-th occurrence of symbol (1-based)
- quantile(l, r, k) -- k-th smallest element in seq[l..r) (0-based k)
- range_freq(l, r, lo, hi) -- count elements in seq[l..r) with lo <= val < hi
- access(pos) -- retrieve element at position
"""


# -- Bitvector with rank/select support --

class BitVector:
    """Bitvector with O(1) rank and O(log n) select via sampling."""

    def __init__(self, bits):
        self.bits = list(bits)
        self.n = len(self.bits)
        # Prefix sum for rank
        self._rank_prefix = [0] * (self.n + 1)
        for i in range(self.n):
            self._rank_prefix[i + 1] = self._rank_prefix[i] + self.bits[i]

    def rank1(self, pos):
        """Count 1-bits in bits[0..pos)."""
        if pos <= 0:
            return 0
        if pos > self.n:
            pos = self.n
        return self._rank_prefix[pos]

    def rank0(self, pos):
        """Count 0-bits in bits[0..pos)."""
        return pos - self.rank1(pos)

    def select1(self, k):
        """Position of k-th 1-bit (1-based). Returns -1 if not found."""
        if k <= 0 or k > self._rank_prefix[self.n]:
            return -1
        lo, hi = 0, self.n
        while lo < hi:
            mid = (lo + hi) // 2
            if self._rank_prefix[mid + 1] < k:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def select0(self, k):
        """Position of k-th 0-bit (1-based). Returns -1 if not found."""
        total_zeros = self.n - self._rank_prefix[self.n]
        if k <= 0 or k > total_zeros:
            return -1
        lo, hi = 0, self.n
        while lo < hi:
            mid = (lo + hi) // 2
            zeros_at_mid = (mid + 1) - self._rank_prefix[mid + 1]
            if zeros_at_mid < k:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def access(self, pos):
        return self.bits[pos]


# -- Wavelet Tree (pointer-based, recursive) --

class WaveletTreeNode:
    __slots__ = ('lo', 'hi', 'bv', 'left', 'right')

    def __init__(self):
        self.lo = 0
        self.hi = 0
        self.bv = None
        self.left = None
        self.right = None


class WaveletTree:
    """Wavelet tree over integer sequences. Alphabet [min_val, max_val]."""

    def __init__(self, seq):
        if not seq:
            self.root = None
            self.n = 0
            self.seq = []
            return
        self.seq = list(seq)
        self.n = len(self.seq)
        lo = min(self.seq)
        hi = max(self.seq)
        self.root = self._build(self.seq, lo, hi)

    def _build(self, seq, lo, hi):
        node = WaveletTreeNode()
        node.lo = lo
        node.hi = hi
        if lo >= hi or not seq:
            return node
        mid = (lo + hi) // 2
        bits = [1 if v > mid else 0 for v in seq]
        node.bv = BitVector(bits)
        left_seq = [v for v in seq if v <= mid]
        right_seq = [v for v in seq if v > mid]
        node.left = self._build(left_seq, lo, mid)
        node.right = self._build(right_seq, mid + 1, hi)
        return node

    def access(self, pos):
        """Retrieve element at position pos."""
        if pos < 0 or pos >= self.n:
            raise IndexError(f"Position {pos} out of range [0, {self.n})")
        return self._access(self.root, pos)

    def _access(self, node, pos):
        if node.lo >= node.hi:
            return node.lo
        if node.bv.access(pos) == 0:
            new_pos = node.bv.rank0(pos)
            return self._access(node.left, new_pos)
        else:
            new_pos = node.bv.rank1(pos)
            return self._access(node.right, new_pos)

    def rank(self, symbol, pos):
        """Count occurrences of symbol in seq[0..pos)."""
        if self.root is None or pos <= 0:
            return 0
        if pos > self.n:
            pos = self.n
        return self._rank(self.root, symbol, pos)

    def _rank(self, node, symbol, pos):
        if node.lo >= node.hi:
            return pos
        mid = (node.lo + node.hi) // 2
        if symbol <= mid:
            new_pos = node.bv.rank0(pos)
            return self._rank(node.left, symbol, new_pos)
        else:
            new_pos = node.bv.rank1(pos)
            return self._rank(node.right, symbol, new_pos)

    def select(self, symbol, k):
        """Position of k-th occurrence of symbol (1-based). Returns -1 if not found."""
        if self.root is None or k <= 0:
            return -1
        return self._select(self.root, symbol, k)

    def _select(self, node, symbol, k):
        if node.lo >= node.hi:
            if k > 0:
                return k - 1  # k-th element in leaf = position k-1 in this node's range
            return -1
        mid = (node.lo + node.hi) // 2
        if symbol <= mid:
            child_pos = self._select(node.left, symbol, k)
            if child_pos < 0:
                return -1
            # Map back: child_pos is position in left child, find in parent
            return node.bv.select0(child_pos + 1)
        else:
            child_pos = self._select(node.right, symbol, k)
            if child_pos < 0:
                return -1
            return node.bv.select1(child_pos + 1)

    def quantile(self, l, r, k):
        """k-th smallest element in seq[l..r) (0-based k). Returns (value, count_less)."""
        if self.root is None or l >= r or k < 0 or k >= (r - l):
            raise ValueError("Invalid quantile query")
        return self._quantile(self.root, l, r, k)

    def _quantile(self, node, l, r, k):
        if node.lo >= node.hi:
            return node.lo
        zeros_l = node.bv.rank0(l)
        zeros_r = node.bv.rank0(r)
        left_count = zeros_r - zeros_l
        if k < left_count:
            return self._quantile(node.left, zeros_l, zeros_r, k)
        else:
            ones_l = node.bv.rank1(l)
            ones_r = node.bv.rank1(r)
            return self._quantile(node.right, ones_l, ones_r, k - left_count)

    def range_freq(self, l, r, lo, hi):
        """Count elements in seq[l..r) with lo <= val < hi."""
        if self.root is None or l >= r or lo >= hi:
            return 0
        return self._range_freq(self.root, l, r, lo, hi)

    def _range_freq(self, node, l, r, lo, hi):
        if l >= r or node.lo >= hi or node.hi < lo:
            return 0
        if lo <= node.lo and node.hi < hi:
            return r - l
        mid = (node.lo + node.hi) // 2
        zeros_l = node.bv.rank0(l)
        zeros_r = node.bv.rank0(r)
        ones_l = node.bv.rank1(l)
        ones_r = node.bv.rank1(r)
        count = 0
        if lo <= mid:
            count += self._range_freq(node.left, zeros_l, zeros_r, lo, hi)
        if hi > mid + 1:
            count += self._range_freq(node.right, ones_l, ones_r, lo, hi)
        return count

    def range_list(self, l, r, lo, hi):
        """List distinct elements in seq[l..r) with lo <= val < hi, with frequencies."""
        if self.root is None or l >= r or lo >= hi:
            return []
        result = []
        self._range_list(self.root, l, r, lo, hi, result)
        return result

    def _range_list(self, node, l, r, lo, hi, result):
        if l >= r or node.lo >= hi or node.hi < lo:
            return
        if node.lo >= node.hi:
            freq = r - l
            if freq > 0 and lo <= node.lo < hi:
                result.append((node.lo, freq))
            return
        mid = (node.lo + node.hi) // 2
        zeros_l = node.bv.rank0(l)
        zeros_r = node.bv.rank0(r)
        ones_l = node.bv.rank1(l)
        ones_r = node.bv.rank1(r)
        self._range_list(node.left, zeros_l, zeros_r, lo, hi, result)
        self._range_list(node.right, ones_l, ones_r, lo, hi, result)

    def top_k(self, l, r, k):
        """Return k most frequent elements in seq[l..r), sorted by frequency desc."""
        elements = self.range_list(l, r, min(self.seq), max(self.seq) + 1)
        elements.sort(key=lambda x: (-x[1], x[0]))
        return elements[:k]

    def prev_value(self, l, r, val):
        """Largest value < val in seq[l..r), or None."""
        if self.root is None or l >= r:
            return None
        lo = self.root.lo
        if val <= lo:
            return None
        # Count elements < val
        count = self.range_freq(l, r, lo, val)
        if count == 0:
            return None
        # The (count-1)-th smallest is the largest < val
        return self.quantile(l, r, count - 1)

    def next_value(self, l, r, val):
        """Smallest value > val in seq[l..r), or None."""
        if self.root is None or l >= r:
            return None
        hi = self.root.hi
        if val >= hi:
            return None
        # Count elements <= val
        count = self.range_freq(l, r, self.root.lo, val + 1)
        if count >= r - l:
            return None
        # The count-th smallest is the smallest > val
        return self.quantile(l, r, count)

    def swap(self, pos1, pos2):
        """Swap elements at pos1 and pos2. Rebuilds the tree (O(n log sigma))."""
        self.seq[pos1], self.seq[pos2] = self.seq[pos2], self.seq[pos1]
        lo = min(self.seq)
        hi = max(self.seq)
        self.root = self._build(self.seq, lo, hi)


# -- Wavelet Matrix (level-wise bitvectors, more cache-friendly) --

class WaveletMatrix:
    """Wavelet matrix: space-efficient wavelet tree using level-wise bitvectors."""

    def __init__(self, seq):
        if not seq:
            self.n = 0
            self.levels = 0
            self.bvs = []
            self.zeros = []
            self.seq = []
            return
        self.seq = list(seq)
        self.n = len(self.seq)
        self.max_val = max(self.seq)
        self.min_val = min(self.seq)
        # Shift to non-negative
        shifted = [v - self.min_val for v in self.seq]
        range_val = self.max_val - self.min_val
        self.levels = max(1, range_val.bit_length()) if range_val > 0 else 1
        self.bvs = []
        self.zeros = []
        self._build(shifted)

    def _build(self, seq):
        cur = list(seq)
        for level in range(self.levels - 1, -1, -1):
            bits = [(v >> level) & 1 for v in cur]
            bv = BitVector(bits)
            self.bvs.append(bv)
            z = bv.rank0(self.n)
            self.zeros.append(z)
            # Stable partition: 0-bits first, then 1-bits
            zeros_list = [v for i, v in enumerate(cur) if bits[i] == 0]
            ones_list = [v for i, v in enumerate(cur) if bits[i] == 1]
            cur = zeros_list + ones_list

    def access(self, pos):
        """Retrieve element at position pos."""
        if pos < 0 or pos >= self.n:
            raise IndexError(f"Position {pos} out of range")
        val = 0
        p = pos
        for level_idx in range(self.levels):
            bit_pos = self.levels - 1 - level_idx
            bv = self.bvs[level_idx]
            z = self.zeros[level_idx]
            if bv.access(p) == 0:
                p = bv.rank0(p)
            else:
                val |= (1 << bit_pos)
                p = z + bv.rank1(p)
        return val + self.min_val

    def rank(self, symbol, pos):
        """Count occurrences of symbol in seq[0..pos)."""
        if pos <= 0 or self.n == 0:
            return 0
        if pos > self.n:
            pos = self.n
        shifted = symbol - self.min_val
        if shifted < 0 or shifted > self.max_val - self.min_val:
            return 0
        l, r = 0, pos
        for level_idx in range(self.levels):
            bit_pos = self.levels - 1 - level_idx
            bv = self.bvs[level_idx]
            z = self.zeros[level_idx]
            bit = (shifted >> bit_pos) & 1
            if bit == 0:
                l = bv.rank0(l)
                r = bv.rank0(r)
            else:
                l = z + bv.rank1(l)
                r = z + bv.rank1(r)
        return r - l

    def select(self, symbol, k):
        """Position of k-th occurrence of symbol (1-based). Returns -1 if not found."""
        if k <= 0 or self.n == 0:
            return -1
        shifted = symbol - self.min_val
        if shifted < 0 or shifted > self.max_val - self.min_val:
            return -1
        # Track the path down
        path = []  # (level_idx, bit, bv, z)
        pos = 0
        for level_idx in range(self.levels):
            bit_pos = self.levels - 1 - level_idx
            bv = self.bvs[level_idx]
            z = self.zeros[level_idx]
            bit = (shifted >> bit_pos) & 1
            path.append((level_idx, bit, bv, z))
            if bit == 0:
                pos = bv.rank0(pos)
            else:
                pos = z + bv.rank1(pos)
        # pos is now the start of this symbol's range at the bottom level
        # k-th occurrence is at position pos + k - 1 at bottom level
        target = pos + k - 1
        # Walk back up
        for level_idx, bit, bv, z in reversed(path):
            if bit == 0:
                # target is in zeros section of this level
                target = bv.select0(target + 1)
            else:
                # target is in ones section (offset by z)
                target = bv.select1(target - z + 1)
            if target < 0:
                return -1
        return target

    def quantile(self, l, r, k):
        """k-th smallest in seq[l..r), 0-based k."""
        if self.n == 0 or l >= r or k < 0 or k >= (r - l):
            raise ValueError("Invalid quantile query")
        val = 0
        for level_idx in range(self.levels):
            bit_pos = self.levels - 1 - level_idx
            bv = self.bvs[level_idx]
            z = self.zeros[level_idx]
            zeros_l = bv.rank0(l)
            zeros_r = bv.rank0(r)
            left_count = zeros_r - zeros_l
            if k < left_count:
                l = zeros_l
                r = zeros_r
            else:
                val |= (1 << bit_pos)
                l = z + bv.rank1(l)
                r = z + bv.rank1(r)
                k -= left_count
        return val + self.min_val

    def range_freq(self, l, r, lo, hi):
        """Count elements in seq[l..r) with lo <= val < hi."""
        if self.n == 0 or l >= r or lo >= hi:
            return 0
        return self._range_freq_le(l, r, hi - 1) - self._range_freq_le(l, r, lo - 1)

    def _range_freq_le(self, l, r, val):
        """Count elements in seq[l..r) <= val."""
        shifted = val - self.min_val
        if shifted < 0:
            return 0
        if shifted >= (1 << self.levels):
            return r - l
        count = 0
        for level_idx in range(self.levels):
            bit_pos = self.levels - 1 - level_idx
            bv = self.bvs[level_idx]
            z = self.zeros[level_idx]
            bit = (shifted >> bit_pos) & 1
            zeros_l = bv.rank0(l)
            zeros_r = bv.rank0(r)
            if bit == 1:
                count += zeros_r - zeros_l
                l = z + bv.rank1(l)
                r = z + bv.rank1(r)
            else:
                l = zeros_l
                r = zeros_r
        count += r - l
        return count


# -- Huffman Wavelet Tree (alphabet-adaptive shape) --

class HuffmanNode:
    __slots__ = ('symbol', 'freq', 'left', 'right', 'code', 'code_len')

    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
        self.code = 0
        self.code_len = 0


def _build_huffman(freq_map):
    """Build Huffman tree from frequency map. Returns root node and code table."""
    import heapq
    if len(freq_map) == 1:
        sym = next(iter(freq_map))
        root = HuffmanNode(symbol=sym, freq=freq_map[sym])
        codes = {sym: (0, 1)}
        return root, codes

    counter = 0
    heap = []
    for sym, freq in freq_map.items():
        heapq.heappush(heap, (freq, counter, HuffmanNode(symbol=sym, freq=freq)))
        counter += 1

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        merged = HuffmanNode(freq=f1 + f2, left=n1, right=n2)
        heapq.heappush(heap, (f1 + f2, counter, merged))
        counter += 1

    root = heap[0][2]
    codes = {}
    _assign_codes(root, 0, 0, codes)
    return root, codes


def _assign_codes(node, code, depth, codes):
    if node.symbol is not None:
        node.code = code
        node.code_len = max(depth, 1)
        codes[node.symbol] = (code, max(depth, 1))
        return
    if node.left:
        _assign_codes(node.left, code << 1, depth + 1, codes)
    if node.right:
        _assign_codes(node.right, (code << 1) | 1, depth + 1, codes)


class HuffmanWaveletTree:
    """Wavelet tree with Huffman-shaped tree for better compression on skewed alphabets."""

    def __init__(self, seq):
        if not seq:
            self.n = 0
            self.root = None
            self.seq = []
            self.codes = {}
            return
        self.seq = list(seq)
        self.n = len(self.seq)
        freq = {}
        for v in self.seq:
            freq[v] = freq.get(v, 0) + 1
        self.huffman_root, self.codes = _build_huffman(freq)
        self.root = self._build(self.huffman_root, self.seq)

    def _build(self, hnode, seq):
        node = WaveletTreeNode()
        if hnode.symbol is not None:
            node.lo = hnode.symbol
            node.hi = hnode.symbol
            return node
        # Determine which symbols go left vs right
        left_syms = set()
        self._collect_symbols(hnode.left, left_syms)
        bits = [0 if v in left_syms else 1 for v in seq]
        node.bv = BitVector(bits)
        left_seq = [v for v in seq if v in left_syms]
        right_seq = [v for v in seq if v not in left_syms]
        node.left = self._build(hnode.left, left_seq)
        node.right = self._build(hnode.right, right_seq)
        return node

    def _collect_symbols(self, hnode, syms):
        if hnode.symbol is not None:
            syms.add(hnode.symbol)
            return
        if hnode.left:
            self._collect_symbols(hnode.left, syms)
        if hnode.right:
            self._collect_symbols(hnode.right, syms)

    def access(self, pos):
        if pos < 0 or pos >= self.n:
            raise IndexError(f"Position {pos} out of range")
        return self._access(self.root, pos)

    def _access(self, node, pos):
        if node.bv is None:
            return node.lo
        if node.bv.access(pos) == 0:
            return self._access(node.left, node.bv.rank0(pos))
        else:
            return self._access(node.right, node.bv.rank1(pos))

    def rank(self, symbol, pos):
        if pos <= 0 or self.n == 0:
            return 0
        if pos > self.n:
            pos = self.n
        if symbol not in self.codes:
            return 0
        return self._rank(self.root, symbol, pos)

    def _rank(self, node, symbol, pos):
        if node.bv is None:
            return pos
        left_syms = set()
        self._collect_symbols_from_wt(node.left, left_syms)
        if symbol in left_syms:
            return self._rank(node.left, symbol, node.bv.rank0(pos))
        else:
            return self._rank(node.right, symbol, node.bv.rank1(pos))

    def _collect_symbols_from_wt(self, node, syms):
        if node.bv is None:
            syms.add(node.lo)
            return
        self._collect_symbols_from_wt(node.left, syms)
        self._collect_symbols_from_wt(node.right, syms)

    def select(self, symbol, k):
        if k <= 0 or self.n == 0 or symbol not in self.codes:
            return -1
        return self._select(self.root, symbol, k)

    def _select(self, node, symbol, k):
        if node.bv is None:
            if node.lo == symbol and k >= 1:
                return k - 1
            return -1
        left_syms = set()
        self._collect_symbols_from_wt(node.left, left_syms)
        if symbol in left_syms:
            child_pos = self._select(node.left, symbol, k)
            if child_pos < 0:
                return -1
            return node.bv.select0(child_pos + 1)
        else:
            child_pos = self._select(node.right, symbol, k)
            if child_pos < 0:
                return -1
            return node.bv.select1(child_pos + 1)

    def avg_bits_per_symbol(self):
        """Average bits per symbol (should approach entropy for Huffman shape)."""
        if not self.codes:
            return 0
        total = sum(self.codes[s][1] * self.seq.count(s) for s in self.codes)
        return total / self.n


# -- Range Wavelet Tree (2D point queries) --

class RangeWaveletTree:
    """Wavelet tree supporting 2D range counting and reporting.

    Given points (x, y), supports:
    - count_points(x1, x2, y1, y2) -- count points in rectangle
    - report_points(x1, x2, y1, y2) -- list points in rectangle
    """

    def __init__(self, points):
        """points: list of (x, y) tuples."""
        if not points:
            self.n = 0
            self.wt = None
            self.sorted_x = []
            self.y_at_pos = []
            return
        # Sort by x, build wavelet tree over y-values
        sorted_pts = sorted(points, key=lambda p: (p[0], p[1]))
        self.sorted_x = [p[0] for p in sorted_pts]
        self.y_at_pos = [p[1] for p in sorted_pts]
        self.n = len(sorted_pts)
        self.wt = WaveletTree(self.y_at_pos)

    def _x_range(self, x1, x2):
        """Find index range [l, r) for x1 <= x <= x2 using binary search."""
        import bisect
        l = bisect.bisect_left(self.sorted_x, x1)
        r = bisect.bisect_right(self.sorted_x, x2)
        return l, r

    def count_points(self, x1, x2, y1, y2):
        """Count points (x, y) with x1 <= x <= x2 and y1 <= y <= y2."""
        if self.wt is None:
            return 0
        l, r = self._x_range(x1, x2)
        if l >= r:
            return 0
        return self.wt.range_freq(l, r, y1, y2 + 1)

    def report_points(self, x1, x2, y1, y2):
        """List points (x, y) in rectangle."""
        if self.wt is None:
            return []
        l, r = self._x_range(x1, x2)
        if l >= r:
            return []
        result = []
        for i in range(l, r):
            y = self.y_at_pos[i]
            if y1 <= y <= y2:
                result.append((self.sorted_x[i], y))
        return result

    def kth_point_by_y(self, x1, x2, k):
        """k-th smallest y among points with x1 <= x <= x2 (0-based)."""
        if self.wt is None:
            raise ValueError("Empty")
        l, r = self._x_range(x1, x2)
        if l >= r or k < 0 or k >= (r - l):
            raise ValueError("Invalid query")
        return self.wt.quantile(l, r, k)


# -- Utility: string wavelet tree --

def wavelet_tree_from_string(s):
    """Build a wavelet tree from a string (using ord values)."""
    return WaveletTree([ord(c) for c in s])


def wavelet_matrix_from_string(s):
    """Build a wavelet matrix from a string (using ord values)."""
    return WaveletMatrix([ord(c) for c in s])
