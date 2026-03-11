"""
C087: Suffix Array -- SA-IS algorithm with LCP array and pattern search.

Components:
1. SuffixArray -- SA-IS (Suffix Array by Induced Sorting) O(n) construction
2. LCPArray -- Kasai's algorithm for LCP (Longest Common Prefix) array
3. SuffixArraySearcher -- Binary search pattern matching on suffix arrays
4. EnhancedSuffixArray -- Combined SA + LCP with advanced queries
5. MultiStringSuffixArray -- Generalized suffix array for multiple strings
"""


# =============================================================================
# Component 1: SuffixArray -- SA-IS O(n) construction
# =============================================================================

def _is_lms(types, i):
    """Check if position i is a leftmost S-type suffix."""
    return i > 0 and types[i] == 'S' and types[i - 1] == 'L'


def _classify_types(s):
    """Classify each suffix as S-type or L-type."""
    n = len(s)
    types = [''] * n
    types[n - 1] = 'S'  # sentinel is S-type
    for i in range(n - 2, -1, -1):
        if s[i] > s[i + 1]:
            types[i] = 'L'
        elif s[i] < s[i + 1]:
            types[i] = 'S'
        else:
            types[i] = types[i + 1]
    return types


def _get_bucket_sizes(s, alphabet_size):
    """Count occurrences of each character."""
    buckets = [0] * alphabet_size
    for c in s:
        buckets[c] += 1
    return buckets


def _get_bucket_heads(buckets):
    """Get the head (start) position of each bucket."""
    heads = [0] * len(buckets)
    total = 0
    for i in range(len(buckets)):
        heads[i] = total
        total += buckets[i]
    return heads


def _get_bucket_tails(buckets):
    """Get the tail (end) position of each bucket."""
    tails = [0] * len(buckets)
    total = 0
    for i in range(len(buckets)):
        total += buckets[i]
        tails[i] = total - 1
    return tails


def _induced_sort(s, sa, types, buckets, lms_positions):
    """Induced sorting: place LMS suffixes, then induce L and S types."""
    n = len(s)
    alphabet_size = len(buckets)

    # Initialize SA
    for i in range(n):
        sa[i] = -1

    # Place LMS suffixes at bucket tails (right to left)
    tails = _get_bucket_tails(buckets)
    for i in reversed(lms_positions):
        sa[tails[s[i]]] = i
        tails[s[i]] -= 1

    # Induce L-type suffixes from bucket heads (left to right)
    heads = _get_bucket_heads(buckets)
    for i in range(n):
        if sa[i] > 0 and types[sa[i] - 1] == 'L':
            sa[heads[s[sa[i] - 1]]] = sa[i] - 1
            heads[s[sa[i] - 1]] += 1

    # Induce S-type suffixes from bucket tails (right to left)
    tails = _get_bucket_tails(buckets)
    for i in range(n - 1, -1, -1):
        if sa[i] > 0 and types[sa[i] - 1] == 'S':
            sa[tails[s[sa[i] - 1]]] = sa[i] - 1
            tails[s[sa[i] - 1]] -= 1


def _lms_substrings_equal(s, types, i, j):
    """Check if two LMS substrings starting at i and j are equal."""
    if i == -1 or j == -1:
        return False
    k = 0
    while True:
        i_is_lms = _is_lms(types, i + k)
        j_is_lms = _is_lms(types, j + k)
        if k > 0 and i_is_lms and j_is_lms:
            return True  # both reached next LMS and matched throughout
        if i_is_lms != j_is_lms:
            return False
        if s[i + k] != s[j + k]:
            return False
        k += 1


def sa_is(s, alphabet_size=None):
    """
    SA-IS algorithm: build suffix array in O(n) time.

    Args:
        s: list of non-negative integers (text + sentinel 0 at end)
        alphabet_size: size of alphabet (max value + 1)

    Returns:
        list: suffix array
    """
    n = len(s)

    if n == 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return [1, 0] if s[0] > s[1] else [0, 1]

    if alphabet_size is None:
        alphabet_size = max(s) + 1

    # Step 1: Classify suffixes
    types = _classify_types(s)

    # Step 2: Find LMS positions
    lms_positions = [i for i in range(n) if _is_lms(types, i)]

    # Step 3: Initial induced sort with LMS suffixes
    buckets = _get_bucket_sizes(s, alphabet_size)
    sa = [-1] * n
    _induced_sort(s, sa, types, buckets, lms_positions)

    # Step 4: Name LMS substrings
    name = 0
    prev = -1
    names = [-1] * n
    for i in range(n):
        pos = sa[i]
        if _is_lms(types, pos):
            if not _lms_substrings_equal(s, types, prev, pos):
                name += 1
            names[pos] = name - 1
            prev = pos

    # Compact named LMS positions
    lms_names = [names[i] for i in range(n) if names[i] != -1]

    # Step 5: If names not unique, recurse
    if name < len(lms_positions):
        sub_sa = sa_is(lms_names, name)
        # Map back to original positions
        sorted_lms = [lms_positions[sub_sa[i]] for i in range(len(sub_sa))]
    else:
        # Names are unique -- direct sort
        sorted_lms = [0] * len(lms_positions)
        for i, nm in enumerate(lms_names):
            sorted_lms[nm] = lms_positions[i]

    # Step 6: Final induced sort with correctly ordered LMS
    sa = [-1] * n
    _induced_sort(s, sa, types, buckets, sorted_lms)

    return sa


class SuffixArray:
    """Suffix array with O(n) SA-IS construction."""

    def __init__(self, text):
        if isinstance(text, str):
            self.text = text
            # Convert to integer array with sentinel
            self._int_text = [ord(c) + 1 for c in text] + [0]
        elif isinstance(text, (list, tuple)):
            self.text = text
            self._int_text = [v + 1 for v in text] + [0]
        else:
            raise TypeError("text must be str, list, or tuple")

        self.n = len(self._int_text)
        self.sa = sa_is(self._int_text)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.sa[i]

    def suffix(self, i):
        """Get the suffix starting at position sa[i]."""
        pos = self.sa[i]
        if isinstance(self.text, str):
            return self.text[pos:]
        return self.text[pos:]

    def suffixes_sorted(self):
        """Return all suffixes in sorted order (excluding sentinel)."""
        result = []
        for i in range(self.n):
            pos = self.sa[i]
            if pos < len(self.text):
                if isinstance(self.text, str):
                    result.append(self.text[pos:])
                else:
                    result.append(tuple(self.text[pos:]))
        return result


# =============================================================================
# Component 2: LCPArray -- Kasai's algorithm
# =============================================================================

class LCPArray:
    """LCP (Longest Common Prefix) array using Kasai's O(n) algorithm."""

    def __init__(self, text, sa=None):
        if isinstance(text, str):
            self.text = text
        elif isinstance(text, (list, tuple)):
            self.text = list(text)
        else:
            raise TypeError("text must be str, list, or tuple")

        if sa is None:
            sa_obj = SuffixArray(text)
            self.sa = sa_obj.sa
            self.n = sa_obj.n
        else:
            if isinstance(sa, SuffixArray):
                self.sa = sa.sa
                self.n = sa.n
            else:
                self.sa = sa
                self.n = len(sa)

        self.lcp = self._kasai()

    def _kasai(self):
        """Kasai's algorithm: compute LCP array in O(n)."""
        n = self.n
        sa = self.sa
        rank = [0] * n
        for i in range(n):
            rank[sa[i]] = i

        lcp = [0] * n
        k = 0
        text = self.text
        text_len = len(text)

        for i in range(n):
            if rank[i] == 0:
                k = 0
                continue
            j = sa[rank[i] - 1]
            while i + k < text_len and j + k < text_len and text[i + k] == text[j + k]:
                k += 1
            lcp[rank[i]] = k
            if k > 0:
                k -= 1

        return lcp

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.lcp[i]

    def longest_repeated_substring(self):
        """Find the longest substring that appears at least twice."""
        if not self.lcp:
            return "" if isinstance(self.text, str) else ()

        max_lcp = 0
        max_idx = 0
        for i in range(1, self.n):
            if self.lcp[i] > max_lcp:
                max_lcp = self.lcp[i]
                max_idx = i

        if max_lcp == 0:
            return "" if isinstance(self.text, str) else ()

        pos = self.sa[max_idx]
        if isinstance(self.text, str):
            return self.text[pos:pos + max_lcp]
        return tuple(self.text[pos:pos + max_lcp])

    def count_distinct_substrings(self):
        """Count the number of distinct non-empty substrings."""
        text_len = len(self.text)
        # Total substrings: n*(n+1)/2, minus LCP overlaps
        total = text_len * (text_len + 1) // 2
        for i in range(self.n):
            total -= self.lcp[i]
        return total


# =============================================================================
# Component 3: SuffixArraySearcher -- Pattern matching
# =============================================================================

class SuffixArraySearcher:
    """Binary search-based pattern matching on suffix arrays."""

    def __init__(self, text, sa=None):
        if isinstance(text, str):
            self.text = text
        else:
            self.text = list(text)

        if sa is None:
            sa_obj = SuffixArray(text)
            self.sa = sa_obj.sa
            self.n = sa_obj.n
        else:
            if isinstance(sa, SuffixArray):
                self.sa = sa.sa
                self.n = sa.n
            else:
                self.sa = sa
                self.n = len(sa)

    def _compare(self, pos, pattern):
        """Compare pattern with suffix starting at pos."""
        text = self.text
        m = len(pattern)
        text_len = len(text)
        for i in range(m):
            if pos + i >= text_len:
                return -1  # suffix is shorter
            if text[pos + i] < pattern[i]:
                return -1
            if text[pos + i] > pattern[i]:
                return 1
        return 0

    def _lower_bound(self, pattern):
        """Find the first SA index where pattern could appear."""
        lo, hi = 0, self.n
        while lo < hi:
            mid = (lo + hi) // 2
            if self._compare(self.sa[mid], pattern) < 0:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _upper_bound(self, pattern):
        """Find one past the last SA index where pattern appears."""
        lo, hi = 0, self.n
        while lo < hi:
            mid = (lo + hi) // 2
            if self._compare(self.sa[mid], pattern) <= 0:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def search(self, pattern):
        """Find all occurrences of pattern. Returns sorted list of positions."""
        lb = self._lower_bound(pattern)
        ub = self._upper_bound(pattern)
        return sorted(self.sa[i] for i in range(lb, ub))

    def count(self, pattern):
        """Count occurrences of pattern."""
        lb = self._lower_bound(pattern)
        ub = self._upper_bound(pattern)
        return ub - lb

    def contains(self, pattern):
        """Check if pattern exists in text."""
        lb = self._lower_bound(pattern)
        if lb >= self.n:
            return False
        return self._compare(self.sa[lb], pattern) == 0

    def longest_common_prefix_of(self, p1, p2):
        """Find the longest common prefix between two patterns in the text."""
        positions1 = self.search(p1)
        positions2 = self.search(p2)
        if not positions1 or not positions2:
            return "" if isinstance(self.text, str) else ()

        best = 0
        text = self.text
        text_len = len(text)
        for pos1 in positions1:
            for pos2 in positions2:
                k = 0
                while (pos1 + k < text_len and pos2 + k < text_len
                       and text[pos1 + k] == text[pos2 + k]):
                    k += 1
                best = max(best, k)

        if isinstance(self.text, str):
            pos = positions1[0]
            return self.text[pos:pos + best]
        pos = positions1[0]
        return tuple(self.text[pos:pos + best])


# =============================================================================
# Component 4: EnhancedSuffixArray -- SA + LCP combined queries
# =============================================================================

class EnhancedSuffixArray:
    """Combined suffix array + LCP array with advanced queries."""

    def __init__(self, text):
        self.text = text
        self._sa_obj = SuffixArray(text)
        self.sa = self._sa_obj.sa
        self.n = self._sa_obj.n
        self._lcp_obj = LCPArray(text, self._sa_obj)
        self.lcp = self._lcp_obj.lcp
        self._searcher = SuffixArraySearcher(text, self._sa_obj)

        # Build inverse SA (rank array)
        self.rank = [0] * self.n
        for i in range(self.n):
            self.rank[self.sa[i]] = i

    def search(self, pattern):
        """Find all occurrences of pattern."""
        return self._searcher.search(pattern)

    def count(self, pattern):
        """Count occurrences of pattern."""
        return self._searcher.count(pattern)

    def contains(self, pattern):
        """Check if pattern exists."""
        return self._searcher.contains(pattern)

    def longest_repeated_substring(self):
        """Find longest substring appearing at least twice."""
        return self._lcp_obj.longest_repeated_substring()

    def count_distinct_substrings(self):
        """Count distinct non-empty substrings."""
        return self._lcp_obj.count_distinct_substrings()

    def longest_common_extension(self, i, j):
        """
        Compute the longest common extension (LCE) between positions i and j.
        Uses the LCP array with a naive range minimum query.
        """
        if i == j:
            return len(self.text) - i

        ri = self.rank[i]
        rj = self.rank[j]
        if ri > rj:
            ri, rj = rj, ri

        # LCE = min LCP in range [ri+1, rj]
        min_lcp = float('inf')
        for k in range(ri + 1, rj + 1):
            if self.lcp[k] < min_lcp:
                min_lcp = self.lcp[k]
        return min_lcp

    def top_k_repeated(self, k):
        """Find the top-k longest repeated substrings."""
        results = []
        seen = set()
        # Collect all (lcp_value, position) pairs
        lcp_entries = []
        for i in range(1, self.n):
            if self.lcp[i] > 0:
                lcp_entries.append((self.lcp[i], self.sa[i]))

        # Sort by LCP descending
        lcp_entries.sort(key=lambda x: -x[0])

        for length, pos in lcp_entries:
            if isinstance(self.text, str):
                substr = self.text[pos:pos + length]
            else:
                substr = tuple(self.text[pos:pos + length])
            if substr not in seen:
                seen.add(substr)
                results.append((substr, length))
                if len(results) >= k:
                    break

        return results

    def all_repeated_substrings(self, min_length=1):
        """Find all substrings appearing at least twice with given minimum length."""
        results = set()
        for i in range(1, self.n):
            if self.lcp[i] >= min_length:
                pos = self.sa[i]
                for length in range(min_length, self.lcp[i] + 1):
                    if isinstance(self.text, str):
                        results.add(self.text[pos:pos + length])
                    else:
                        results.add(tuple(self.text[pos:pos + length]))
        return results

    def kth_substring(self, k):
        """Find the k-th lexicographically smallest distinct substring (1-indexed)."""
        text_len = len(self.text)
        count = 0
        for i in range(self.n):
            pos = self.sa[i]
            if pos >= text_len:
                continue  # skip sentinel
            # Number of new substrings from this suffix
            start = self.lcp[i] + 1 if i > 0 else 1
            num_new = (text_len - pos) - (self.lcp[i] if i > 0 else 0)
            if num_new <= 0:
                continue
            if count + num_new >= k:
                # k-th falls in this suffix
                length = (self.lcp[i] if i > 0 else 0) + (k - count)
                if isinstance(self.text, str):
                    return self.text[pos:pos + length]
                return tuple(self.text[pos:pos + length])
            count += num_new

        return None  # k exceeds total distinct substrings


# =============================================================================
# Component 5: MultiStringSuffixArray -- Generalized suffix array
# =============================================================================

class MultiStringSuffixArray:
    """Generalized suffix array for multiple strings."""

    def __init__(self, texts):
        if not texts:
            raise ValueError("texts must be non-empty")

        self.texts = texts
        self.num_texts = len(texts)
        self._is_str = isinstance(texts[0], str)

        # Build concatenated text with unique separators
        # Use values beyond the alphabet range as separators
        if self._is_str:
            max_val = max(ord(c) for t in texts for c in t) if any(texts) else 0
        else:
            max_val = max(v for t in texts for v in t) if any(texts) else 0

        self._separators = list(range(max_val + 2, max_val + 2 + self.num_texts))

        # Build concatenated array with separators
        concat = []
        self._boundaries = []  # (start, end) for each text
        pos = 0
        for i, t in enumerate(texts):
            start = pos
            if self._is_str:
                concat.extend(ord(c) + 1 for c in t)
            else:
                concat.extend(v + 1 for v in t)
            pos += len(t)
            self._boundaries.append((start, pos))
            # Add unique separator (use i+1 so sentinel 0 is smallest)
            # Separators should sort between nothing and text chars
            # Use special encoding: separators are mapped to values 1..num_texts
            # and text chars are shifted up by num_texts
            concat.append(0)  # sentinel-like separator
            pos += 1

        # Re-encode: separators to unique small values, text chars shifted
        # Actually let's use a simpler approach: just make unique separators
        # that are smaller than any text character
        encoded = []
        pos = 0
        sep_idx = 1
        for i, t in enumerate(texts):
            for ch in t:
                if self._is_str:
                    encoded.append(ord(ch) + self.num_texts + 1)
                else:
                    encoded.append(ch + self.num_texts + 1)
            encoded.append(sep_idx)  # unique separator
            sep_idx += 1
            pos += len(t) + 1

        # Replace last separator with 0 (sentinel)
        if encoded:
            encoded[-1] = 0

        self._concat = encoded
        self._total_len = len(encoded)

        # Build suffix array on concatenated text
        alpha_size = max(encoded) + 1 if encoded else 1
        self.sa = sa_is(encoded, alpha_size)

        # Map each position to its source text index and track concat starts
        self._pos_to_text = [-1] * self._total_len
        self._concat_starts = []  # concat start position of each text
        pos = 0
        for i, t in enumerate(texts):
            self._concat_starts.append(pos)
            for j in range(len(t)):
                self._pos_to_text[pos + j] = i
            pos += len(t) + 1  # +1 for separator

    def _get_text_index(self, pos):
        """Get which text a position belongs to."""
        if 0 <= pos < self._total_len:
            return self._pos_to_text[pos]
        return -1

    def _get_text_offset(self, pos):
        """Get the offset within the source text."""
        text_idx = self._get_text_index(pos)
        if text_idx < 0:
            return -1
        return pos - self._concat_starts[text_idx]

    def search(self, pattern):
        """
        Search for pattern across all texts.
        Returns list of (text_index, position_in_text).
        """
        if self._is_str:
            int_pattern = [ord(c) + self.num_texts + 1 for c in pattern]
        else:
            int_pattern = [v + self.num_texts + 1 for v in pattern]

        # Binary search
        lo, hi = 0, self._total_len
        while lo < hi:
            mid = (lo + hi) // 2
            pos = self.sa[mid]
            cmp = self._compare_at(pos, int_pattern)
            if cmp < 0:
                lo = mid + 1
            else:
                hi = mid
        lb = lo

        lo, hi = lb, self._total_len
        while lo < hi:
            mid = (lo + hi) // 2
            pos = self.sa[mid]
            cmp = self._compare_at(pos, int_pattern)
            if cmp <= 0:
                lo = mid + 1
            else:
                hi = mid
        ub = lo

        results = []
        for i in range(lb, ub):
            pos = self.sa[i]
            text_idx = self._get_text_index(pos)
            if text_idx >= 0:
                offset = self._get_text_offset(pos)
                results.append((text_idx, offset))

        return sorted(results)

    def _compare_at(self, pos, pattern):
        """Compare pattern with suffix at pos in concatenated text."""
        concat = self._concat
        n = self._total_len
        m = len(pattern)
        for i in range(m):
            if pos + i >= n:
                return -1
            if concat[pos + i] < pattern[i]:
                return -1
            if concat[pos + i] > pattern[i]:
                return 1
        return 0

    def longest_common_substring(self):
        """Find the longest common substring across all texts."""
        if self.num_texts < 2:
            return "" if self._is_str else ()

        # Build LCP array on concatenated text
        n = self._total_len
        sa = self.sa
        rank = [0] * n
        for i in range(n):
            rank[sa[i]] = i

        lcp = [0] * n
        k = 0
        concat = self._concat
        for i in range(n):
            if rank[i] == 0:
                k = 0
                continue
            j = sa[rank[i] - 1]
            while i + k < n and j + k < n and concat[i + k] == concat[j + k]:
                k += 1
            lcp[rank[i]] = k
            if k > 0:
                k -= 1

        # Find max LCP where adjacent SA entries come from different texts
        best_len = 0
        best_pos = 0
        for i in range(1, n):
            if lcp[i] > best_len:
                t1 = self._get_text_index(sa[i])
                t2 = self._get_text_index(sa[i - 1])
                if t1 >= 0 and t2 >= 0 and t1 != t2:
                    best_len = lcp[i]
                    best_pos = sa[i]

        if best_len == 0:
            return "" if self._is_str else ()

        # Decode back to original characters
        if self._is_str:
            chars = []
            for j in range(best_len):
                val = self._concat[best_pos + j] - self.num_texts - 1
                chars.append(chr(val))
            return ''.join(chars)
        else:
            return tuple(self._concat[best_pos + j] - self.num_texts - 1
                         for j in range(best_len))

    def search_in_text(self, pattern, text_index):
        """Search for pattern only in a specific text."""
        all_results = self.search(pattern)
        return [(ti, pos) for ti, pos in all_results if ti == text_index]

    def common_substrings(self, min_length=1):
        """Find substrings common to all texts with minimum length."""
        if self.num_texts < 2:
            return set()

        # Build LCP on concat
        n = self._total_len
        sa = self.sa
        rank = [0] * n
        for i in range(n):
            rank[sa[i]] = i

        lcp_arr = [0] * n
        k = 0
        concat = self._concat
        for i in range(n):
            if rank[i] == 0:
                k = 0
                continue
            j = sa[rank[i] - 1]
            while i + k < n and j + k < n and concat[i + k] == concat[j + k]:
                k += 1
            lcp_arr[rank[i]] = k
            if k > 0:
                k -= 1

        # Sliding window: find ranges where all texts are represented
        results = set()
        for i in range(1, n):
            # Check window [i-1, i] and expand
            # Simple approach: for each position with sufficient LCP,
            # check if nearby entries cover all texts
            if lcp_arr[i] >= min_length:
                # Expand window to find all texts
                window_min_lcp = lcp_arr[i]
                texts_seen = set()
                t = self._get_text_index(sa[i])
                if t >= 0:
                    texts_seen.add(t)
                # Look backwards
                j = i - 1
                while j >= 0:
                    t = self._get_text_index(sa[j])
                    if t >= 0:
                        texts_seen.add(t)
                    if len(texts_seen) == self.num_texts:
                        break
                    if j > 0 and lcp_arr[j] < min_length:
                        break
                    if j > 0:
                        window_min_lcp = min(window_min_lcp, lcp_arr[j])
                    j -= 1

                if len(texts_seen) == self.num_texts and window_min_lcp >= min_length:
                    pos = sa[i]
                    for length in range(min_length, window_min_lcp + 1):
                        if self._is_str:
                            substr = ''.join(
                                chr(self._concat[pos + x] - self.num_texts - 1)
                                for x in range(length)
                            )
                        else:
                            substr = tuple(
                                self._concat[pos + x] - self.num_texts - 1
                                for x in range(length)
                            )
                        results.add(substr)

        return results
