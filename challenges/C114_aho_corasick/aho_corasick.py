"""
C114: Aho-Corasick Multi-Pattern String Matching

Variants:
1. AhoCorasick -- core automaton (trie + failure links + output links)
2. StreamingAhoCorasick -- process text incrementally, maintain state
3. WeightedAhoCorasick -- patterns with weights/priorities, best-match
4. WildcardAhoCorasick -- patterns with '?' single-char wildcard
5. AhoCorasickReplacer -- find-and-replace with pattern priority
6. AhoCorasickCounter -- count occurrences efficiently
"""

from collections import deque


# ---------------------------------------------------------------------------
# Variant 1: Core Aho-Corasick Automaton
# ---------------------------------------------------------------------------

class TrieNode:
    __slots__ = ('children', 'fail', 'output', 'depth', 'pattern_id')

    def __init__(self):
        self.children = {}      # char -> TrieNode
        self.fail = None        # failure link
        self.output = []        # list of (pattern_id, pattern_string)
        self.depth = 0
        self.pattern_id = -1    # if this node terminates a pattern


class AhoCorasick:
    """Multi-pattern string matching using Aho-Corasick automaton.

    Build a trie of patterns, compute failure links via BFS,
    then scan text in O(n + m + z) where n=text length, m=total pattern length, z=matches.
    """

    def __init__(self, patterns=None):
        self.root = TrieNode()
        self.patterns = []
        self._built = False
        if patterns:
            for p in patterns:
                self.add_pattern(p)
            self.build()

    def add_pattern(self, pattern):
        """Add a pattern to the automaton. Must call build() after all patterns added."""
        if self._built:
            raise RuntimeError("Cannot add patterns after build()")
        pid = len(self.patterns)
        self.patterns.append(pattern)
        node = self.root
        for ch in pattern:
            if ch not in node.children:
                child = TrieNode()
                child.depth = node.depth + 1
                node.children[ch] = child
            node = node.children[ch]
        node.pattern_id = pid
        return pid

    def build(self):
        """Compute failure links and output links via BFS."""
        if self._built:
            return
        self.root.fail = self.root
        queue = deque()

        # Initialize depth-1 nodes: fail -> root
        for ch, child in self.root.children.items():
            child.fail = self.root
            if child.pattern_id >= 0:
                child.output = [(child.pattern_id, self.patterns[child.pattern_id])]
            queue.append(child)

        # BFS to set failure links
        while queue:
            node = queue.popleft()
            for ch, child in node.children.items():
                queue.append(child)
                # Follow failure links to find longest proper suffix that is a prefix
                fail = node.fail
                while fail is not self.root and ch not in fail.children:
                    fail = fail.fail
                child.fail = fail.children[ch] if ch in fail.children else self.root
                if child.fail is child:
                    child.fail = self.root
                # Merge output: this node's pattern (if any) + failure's output
                child.output = []
                if child.pattern_id >= 0:
                    child.output.append((child.pattern_id, self.patterns[child.pattern_id]))
                child.output.extend(child.fail.output)

        self._built = True

    def search(self, text):
        """Search text for all pattern occurrences.

        Returns list of (position, pattern_id, pattern_string) sorted by position.
        Position is the start index in text where the pattern begins.
        """
        if not self._built:
            raise RuntimeError("Must call build() before search()")
        results = []
        node = self.root
        for i, ch in enumerate(text):
            while node is not self.root and ch not in node.children:
                node = node.fail
            if ch in node.children:
                node = node.children[ch]
            # else: stay at root
            for pid, pat in node.output:
                start = i - len(pat) + 1
                results.append((start, pid, pat))
        results.sort(key=lambda x: (x[0], x[1]))
        return results

    def search_first(self, text):
        """Return the first match or None."""
        if not self._built:
            raise RuntimeError("Must call build() before search()")
        node = self.root
        for i, ch in enumerate(text):
            while node is not self.root and ch not in node.children:
                node = node.fail
            if ch in node.children:
                node = node.children[ch]
            if node.output:
                pid, pat = node.output[0]
                start = i - len(pat) + 1
                return (start, pid, pat)
        return None

    def contains_any(self, text):
        """Check if text contains any pattern."""
        return self.search_first(text) is not None

    def __len__(self):
        return len(self.patterns)

    def __repr__(self):
        return f"AhoCorasick(patterns={len(self.patterns)}, built={self._built})"


# ---------------------------------------------------------------------------
# Variant 2: Streaming Aho-Corasick
# ---------------------------------------------------------------------------

class StreamingAhoCorasick:
    """Process text incrementally, maintaining automaton state between calls.

    Useful for processing large files or network streams chunk by chunk.
    """

    def __init__(self, patterns=None):
        self.ac = AhoCorasick()
        self._state = None
        self._offset = 0
        if patterns:
            for p in patterns:
                self.ac.add_pattern(p)
            self.ac.build()
            self._state = self.ac.root

    def add_pattern(self, pattern):
        return self.ac.add_pattern(pattern)

    def build(self):
        self.ac.build()
        self._state = self.ac.root

    def feed(self, chunk):
        """Feed a chunk of text. Returns matches found in this chunk.

        Positions are global (accounting for all previous chunks).
        """
        if not self.ac._built:
            raise RuntimeError("Must call build() before feed()")
        results = []
        node = self._state
        for i, ch in enumerate(chunk):
            while node is not self.ac.root and ch not in node.children:
                node = node.fail
            if ch in node.children:
                node = node.children[ch]
            for pid, pat in node.output:
                start = self._offset + i - len(pat) + 1
                results.append((start, pid, pat))
        self._state = node
        self._offset += len(chunk)
        return results

    def reset(self):
        """Reset state to process a new text."""
        self._state = self.ac.root
        self._offset = 0

    @property
    def patterns(self):
        return self.ac.patterns

    @property
    def total_bytes_processed(self):
        return self._offset


# ---------------------------------------------------------------------------
# Variant 3: Weighted Aho-Corasick
# ---------------------------------------------------------------------------

class WeightedAhoCorasick:
    """Patterns with weights/priorities. Supports best-match queries.

    Higher weight = higher priority.
    """

    def __init__(self):
        self.ac = AhoCorasick()
        self.weights = {}  # pattern_id -> weight

    def add_pattern(self, pattern, weight=1.0):
        pid = self.ac.add_pattern(pattern)
        self.weights[pid] = weight
        return pid

    def build(self):
        self.ac.build()

    def search(self, text):
        """Return all matches with weights: (position, pattern_id, pattern, weight)."""
        raw = self.ac.search(text)
        return [(pos, pid, pat, self.weights.get(pid, 1.0)) for pos, pid, pat in raw]

    def search_best(self, text):
        """Return only the highest-weight match at each position."""
        raw = self.ac.search(text)
        best_at = {}  # position -> (pos, pid, pat, weight)
        for pos, pid, pat in raw:
            w = self.weights.get(pid, 1.0)
            if pos not in best_at or w > best_at[pos][3]:
                best_at[pos] = (pos, pid, pat, w)
        result = sorted(best_at.values(), key=lambda x: x[0])
        return result

    def search_top_k(self, text, k=5):
        """Return top-k matches by weight."""
        matches = self.search(text)
        matches.sort(key=lambda x: -x[3])
        return matches[:k]

    @property
    def patterns(self):
        return self.ac.patterns


# ---------------------------------------------------------------------------
# Variant 4: Wildcard Aho-Corasick
# ---------------------------------------------------------------------------

class WildcardAhoCorasick:
    """Supports '?' as a single-character wildcard in patterns.

    Uses the shift-count approach: decompose each wildcard pattern into
    non-wildcard fragments with offsets, then combine fragment matches
    to detect full pattern matches.
    """

    def __init__(self, wildcard='?'):
        self.wildcard = wildcard
        self._patterns = []      # (original_pattern, fragments_info)
        self._ac = AhoCorasick()
        self._fragment_map = {}  # ac_pattern_id -> (pattern_idx, fragment_offset)
        self._built = False

    def add_pattern(self, pattern):
        pid = len(self._patterns)
        # Split pattern by wildcard into fragments with their offsets
        fragments = []
        current = []
        offset = 0
        start = 0
        for i, ch in enumerate(pattern):
            if ch == self.wildcard:
                if current:
                    frag = ''.join(current)
                    fragments.append((frag, start))
                    current = []
                start = i + 1
            else:
                if not current:
                    start = i
                current.append(ch)
        if current:
            frag = ''.join(current)
            fragments.append((frag, start))

        self._patterns.append((pattern, fragments, len(pattern)))

        # Add each fragment to the AC automaton
        for frag, frag_offset in fragments:
            ac_pid = self._ac.add_pattern(frag)
            self._fragment_map[ac_pid] = (pid, frag_offset)

        return pid

    def build(self):
        self._ac.build()
        self._built = True

    def search(self, text):
        """Search for wildcard patterns. Returns (position, pattern_id, original_pattern)."""
        if not self._built:
            raise RuntimeError("Must call build() before search()")

        raw_matches = self._ac.search(text)

        # For each pattern, count how many fragments matched at each potential start position
        # A full match requires ALL fragments to match
        from collections import defaultdict
        # pattern_id -> {start_pos -> set of fragment_offsets matched}
        candidates = defaultdict(lambda: defaultdict(set))

        for pos, ac_pid, frag in raw_matches:
            if ac_pid in self._fragment_map:
                pat_id, frag_offset = self._fragment_map[ac_pid]
                # The pattern would start at pos - frag_offset
                pat_start = pos - frag_offset
                if pat_start >= 0:
                    candidates[pat_id][pat_start].add(frag_offset)

        results = []
        for pat_id, starts in candidates.items():
            pattern, fragments, pat_len = self._patterns[pat_id]
            num_fragments = len(fragments)
            if num_fragments == 0:
                # All wildcards -- match everywhere
                for i in range(len(text) - pat_len + 1):
                    results.append((i, pat_id, pattern))
            else:
                for start_pos, matched_offsets in starts.items():
                    if start_pos + pat_len > len(text):
                        continue
                    if len(matched_offsets) == num_fragments:
                        # Verify all fragment offsets are accounted for
                        expected = {fo for _, fo in fragments}
                        if matched_offsets == expected:
                            results.append((start_pos, pat_id, pattern))

        results.sort(key=lambda x: (x[0], x[1]))
        return results

    @property
    def patterns(self):
        return [p[0] for p in self._patterns]


# ---------------------------------------------------------------------------
# Variant 5: Aho-Corasick Replacer
# ---------------------------------------------------------------------------

class AhoCorasickReplacer:
    """Find-and-replace with multiple patterns simultaneously.

    Supports priority-based replacement (longer match or higher priority wins)
    and non-overlapping replacement modes.
    """

    def __init__(self):
        self.ac = AhoCorasick()
        self._replacements = {}  # pattern_id -> replacement_string
        self._priorities = {}    # pattern_id -> priority (higher wins)

    def add_rule(self, pattern, replacement, priority=None):
        """Add a replacement rule. Priority defaults to pattern length."""
        pid = self.ac.add_pattern(pattern)
        self._replacements[pid] = replacement
        self._priorities[pid] = priority if priority is not None else len(pattern)
        return pid

    def build(self):
        self.ac.build()

    def replace(self, text, mode='longest'):
        """Replace all non-overlapping matches in text.

        Modes:
        - 'longest': prefer longer matches at each position
        - 'first': prefer first-added pattern at each position
        - 'priority': prefer highest priority
        """
        matches = self.ac.search(text)
        if not matches:
            return text

        # Group by start position, pick best match
        from collections import defaultdict
        by_pos = defaultdict(list)
        for pos, pid, pat in matches:
            by_pos[pos].append((pos, pid, pat))

        # Select best match at each position
        selected = {}
        for pos, candidates in by_pos.items():
            if mode == 'longest':
                best = max(candidates, key=lambda x: len(x[2]))
            elif mode == 'first':
                best = min(candidates, key=lambda x: x[1])
            elif mode == 'priority':
                best = max(candidates, key=lambda x: self._priorities.get(x[1], 0))
            else:
                best = candidates[0]
            selected[pos] = best

        # Greedy non-overlapping: scan left to right
        sorted_positions = sorted(selected.keys())
        result_parts = []
        i = 0
        for pos in sorted_positions:
            if pos < i:
                continue  # skip overlapping
            _, pid, pat = selected[pos]
            result_parts.append(text[i:pos])
            result_parts.append(self._replacements[pid])
            i = pos + len(pat)
        result_parts.append(text[i:])
        return ''.join(result_parts)

    def replace_all(self, text):
        """Replace with all overlapping matches reported (last replacement wins)."""
        return self.replace(text, mode='longest')

    @property
    def patterns(self):
        return self.ac.patterns


# ---------------------------------------------------------------------------
# Variant 6: Aho-Corasick Counter
# ---------------------------------------------------------------------------

class AhoCorasickCounter:
    """Efficiently count pattern occurrences without storing all match positions.

    Also supports frequency analysis and pattern ranking.
    """

    def __init__(self, patterns=None):
        self.ac = AhoCorasick()
        self._counts = {}
        if patterns:
            for p in patterns:
                self.ac.add_pattern(p)
            self.ac.build()

    def add_pattern(self, pattern):
        return self.ac.add_pattern(pattern)

    def build(self):
        self.ac.build()

    def count(self, text):
        """Count occurrences of each pattern. Returns dict: pattern -> count."""
        if not self.ac._built:
            raise RuntimeError("Must call build() before count()")
        counts = {}
        for pat in self.ac.patterns:
            counts[pat] = 0
        node = self.ac.root
        for ch in text:
            while node is not self.ac.root and ch not in node.children:
                node = node.fail
            if ch in node.children:
                node = node.children[ch]
            for pid, pat in node.output:
                counts[pat] = counts.get(pat, 0) + 1
        return counts

    def total_matches(self, text):
        """Return total number of matches across all patterns."""
        counts = self.count(text)
        return sum(counts.values())

    def most_common(self, text, k=None):
        """Return patterns sorted by frequency (most common first)."""
        counts = self.count(text)
        items = sorted(counts.items(), key=lambda x: -x[1])
        if k is not None:
            items = items[:k]
        return items

    def matched_patterns(self, text):
        """Return set of patterns that matched at least once."""
        counts = self.count(text)
        return {pat for pat, c in counts.items() if c > 0}

    @property
    def patterns(self):
        return self.ac.patterns
