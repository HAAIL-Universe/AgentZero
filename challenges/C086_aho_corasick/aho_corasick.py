"""C086: Aho-Corasick -- Multi-pattern string matching automaton.

Composes the trie concept from C085 with failure links (KMP generalized
to a trie) for linear-time simultaneous multi-pattern search.

Four variants:
1. AhoCorasick -- classic automaton (build, search, find_all, find_first)
2. AhoCorasickStream -- streaming/incremental matching (feed chunks)
3. AhoCorasickReplacer -- simultaneous multi-pattern replacement
4. WildcardAC -- patterns with single-char wildcards (?)
"""

from collections import deque


# =============================================================================
# Variant 1: Classic Aho-Corasick
# =============================================================================

class ACNode:
    """Node in the Aho-Corasick automaton."""
    __slots__ = ('children', 'fail', 'output', 'depth', 'dict_suffix')

    def __init__(self):
        self.children = {}      # char -> ACNode
        self.fail = None        # failure link (suffix pointer)
        self.output = []        # pattern indices that end here
        self.depth = 0          # depth in trie (= prefix length)
        self.dict_suffix = None # dictionary suffix link (shortcut to next output)


class Match:
    """A match result: position in text, pattern index, matched text."""
    __slots__ = ('start', 'end', 'pattern_idx', 'pattern')

    def __init__(self, start, end, pattern_idx, pattern):
        self.start = start
        self.end = end
        self.pattern_idx = pattern_idx
        self.pattern = pattern

    def __repr__(self):
        return f"Match(start={self.start}, end={self.end}, pattern={self.pattern!r})"

    def __eq__(self, other):
        if not isinstance(other, Match):
            return NotImplemented
        return (self.start == other.start and self.end == other.end
                and self.pattern_idx == other.pattern_idx)

    def __hash__(self):
        return hash((self.start, self.end, self.pattern_idx))


class AhoCorasick:
    """Classic Aho-Corasick multi-pattern string matching.

    Build phase: O(sum of pattern lengths)
    Search phase: O(text length + number of matches)
    """

    def __init__(self, patterns=None, case_sensitive=True):
        self.root = ACNode()
        self._patterns = []
        self._built = False
        self._case_sensitive = case_sensitive
        if patterns is not None:
            for p in patterns:
                self.add_pattern(p)
            self.build()

    def _normalize(self, s):
        """Normalize string based on case sensitivity."""
        return s if self._case_sensitive else s.lower()

    @property
    def patterns(self):
        return list(self._patterns)

    @property
    def pattern_count(self):
        return len(self._patterns)

    def add_pattern(self, pattern):
        """Add a pattern to the automaton. Must call build() after adding all patterns."""
        if self._built:
            raise RuntimeError("Cannot add patterns after build()")
        idx = len(self._patterns)
        self._patterns.append(pattern)
        node = self.root
        for ch in self._normalize(pattern):
            if ch not in node.children:
                child = ACNode()
                child.depth = node.depth + 1
                node.children[ch] = child
            node = node.children[ch]
        node.output.append(idx)
        return idx

    def build(self):
        """Build failure links using BFS. Must be called after adding all patterns."""
        if self._built:
            return
        root = self.root
        root.fail = root
        root.dict_suffix = None

        # BFS to build failure links
        queue = deque()
        for ch, child in root.children.items():
            child.fail = root
            child.dict_suffix = None
            queue.append(child)

        while queue:
            node = queue.popleft()
            for ch, child in node.children.items():
                queue.append(child)
                # Follow failure links to find longest proper suffix
                fail = node.fail
                while fail is not root and ch not in fail.children:
                    fail = fail.fail
                child.fail = fail.children[ch] if ch in fail.children else root
                if child.fail is child:
                    child.fail = root  # prevent self-loop
                # Dictionary suffix link: shortcut to next node with output
                if child.fail.output:
                    child.dict_suffix = child.fail
                else:
                    child.dict_suffix = child.fail.dict_suffix

        self._built = True

    def _check_built(self):
        if not self._built:
            raise RuntimeError("Must call build() before searching")

    def find_all(self, text):
        """Find all occurrences of all patterns in text. Returns list of Match."""
        self._check_built()
        results = []
        node = self.root
        text_norm = self._normalize(text)

        for i, ch in enumerate(text_norm):
            while node is not self.root and ch not in node.children:
                node = node.fail
            if ch in node.children:
                node = node.children[ch]
            # else: stay at root

            # Collect outputs from this node and dict suffix chain
            temp = node
            while temp is not None and temp is not self.root:
                for idx in temp.output:
                    pat = self._patterns[idx]
                    start = i - len(pat) + 1
                    results.append(Match(start, i + 1, idx, pat))
                temp = temp.dict_suffix

        return results

    def find_first(self, text):
        """Find first occurrence of any pattern. Returns Match or None."""
        self._check_built()
        node = self.root
        text_norm = self._normalize(text)

        for i, ch in enumerate(text_norm):
            while node is not self.root and ch not in node.children:
                node = node.fail
            if ch in node.children:
                node = node.children[ch]

            temp = node
            while temp is not None and temp is not self.root:
                if temp.output:
                    idx = temp.output[0]
                    pat = self._patterns[idx]
                    start = i - len(pat) + 1
                    return Match(start, i + 1, idx, pat)
                temp = temp.dict_suffix

        return None

    def find_non_overlapping(self, text):
        """Find non-overlapping matches (leftmost, then longest)."""
        self._check_built()
        all_matches = self.find_all(text)
        if not all_matches:
            return []
        # Sort by start, then by length descending (longest first at same start)
        all_matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
        result = []
        last_end = -1
        for m in all_matches:
            if m.start >= last_end:
                result.append(m)
                last_end = m.end
        return result

    def search(self, text):
        """Return True if any pattern is found in text."""
        return self.find_first(text) is not None

    def count_matches(self, text):
        """Count total number of (overlapping) matches."""
        return len(self.find_all(text))

    def match_at(self, text, position):
        """Check if any pattern matches starting at the given position."""
        self._check_built()
        node = self.root
        text_norm = self._normalize(text)
        for i in range(position, len(text_norm)):
            ch = text_norm[i]
            if ch not in node.children:
                return None
            node = node.children[ch]
            if node.output:
                idx = node.output[0]
                pat = self._patterns[idx]
                return Match(position, i + 1, idx, pat)
        return None

    def which_patterns(self, text):
        """Return set of pattern indices found in text."""
        matches = self.find_all(text)
        return {m.pattern_idx for m in matches}

    def pattern_positions(self, text):
        """Return dict mapping pattern index -> list of start positions."""
        matches = self.find_all(text)
        result = {}
        for m in matches:
            if m.pattern_idx not in result:
                result[m.pattern_idx] = []
            result[m.pattern_idx].append(m.start)
        return result


# =============================================================================
# Variant 2: Streaming Aho-Corasick
# =============================================================================

class AhoCorasickStream:
    """Streaming/incremental Aho-Corasick for processing text in chunks.

    Maintains automaton state between feed() calls, so text can be
    processed incrementally without buffering the entire input.
    """

    def __init__(self, patterns=None, case_sensitive=True):
        self._ac = AhoCorasick(patterns, case_sensitive)
        self._state = self._ac.root
        self._offset = 0  # total characters processed
        self._matches = []

    @property
    def ac(self):
        return self._ac

    @property
    def total_processed(self):
        return self._offset

    @property
    def matches(self):
        return list(self._matches)

    def add_pattern(self, pattern):
        return self._ac.add_pattern(pattern)

    def build(self):
        self._ac.build()
        self._state = self._ac.root

    def reset(self):
        """Reset streaming state (keep automaton)."""
        self._state = self._ac.root
        self._offset = 0
        self._matches = []

    def feed(self, chunk):
        """Process a chunk of text. Returns new matches found in this chunk."""
        self._ac._check_built()
        node = self._state
        root = self._ac.root
        new_matches = []
        norm = self._ac._normalize(chunk)

        for i, ch in enumerate(norm):
            global_pos = self._offset + i
            while node is not root and ch not in node.children:
                node = node.fail
            if ch in node.children:
                node = node.children[ch]

            temp = node
            while temp is not None and temp is not root:
                for idx in temp.output:
                    pat = self._ac._patterns[idx]
                    start = global_pos - len(pat) + 1
                    m = Match(start, global_pos + 1, idx, pat)
                    new_matches.append(m)
                    self._matches.append(m)
                temp = temp.dict_suffix

        self._state = node
        self._offset += len(chunk)
        return new_matches

    def feed_all(self, chunks):
        """Process multiple chunks. Returns all new matches."""
        all_new = []
        for chunk in chunks:
            all_new.extend(self.feed(chunk))
        return all_new


# =============================================================================
# Variant 3: Aho-Corasick Replacer
# =============================================================================

class AhoCorasickReplacer:
    """Simultaneous multi-pattern find-and-replace using Aho-Corasick.

    Given a mapping of patterns to replacements, replaces all occurrences
    in a single pass through the text.
    """

    def __init__(self, replacements=None, case_sensitive=True):
        """replacements: dict mapping pattern -> replacement string."""
        self._ac = AhoCorasick(case_sensitive=case_sensitive)
        self._replacements = {}  # pattern_idx -> replacement
        self._case_sensitive = case_sensitive
        if replacements:
            for pattern, replacement in replacements.items():
                self.add_replacement(pattern, replacement)
            self.build()

    def add_replacement(self, pattern, replacement):
        """Add a pattern -> replacement mapping."""
        idx = self._ac.add_pattern(pattern)
        self._replacements[idx] = replacement
        return idx

    def build(self):
        self._ac.build()

    def replace(self, text):
        """Replace all non-overlapping pattern occurrences. Returns new string."""
        self._ac._check_built()
        matches = self._ac.find_non_overlapping(text)
        if not matches:
            return text

        parts = []
        last = 0
        for m in matches:
            if m.start > last:
                parts.append(text[last:m.start])
            parts.append(self._replacements[m.pattern_idx])
            last = m.end
        if last < len(text):
            parts.append(text[last:])
        return ''.join(parts)

    def replace_with(self, text, fn):
        """Replace matches using a callback function fn(match) -> str."""
        self._ac._check_built()
        matches = self._ac.find_non_overlapping(text)
        if not matches:
            return text

        parts = []
        last = 0
        for m in matches:
            if m.start > last:
                parts.append(text[last:m.start])
            parts.append(fn(m))
            last = m.end
        if last < len(text):
            parts.append(text[last:])
        return ''.join(parts)

    def replace_count(self, text):
        """Replace and return (new_text, count)."""
        self._ac._check_built()
        matches = self._ac.find_non_overlapping(text)
        if not matches:
            return text, 0

        parts = []
        last = 0
        for m in matches:
            if m.start > last:
                parts.append(text[last:m.start])
            parts.append(self._replacements[m.pattern_idx])
            last = m.end
        if last < len(text):
            parts.append(text[last:])
        return ''.join(parts), len(matches)


# =============================================================================
# Variant 4: Wildcard Aho-Corasick
# =============================================================================

class WildcardAC:
    """Aho-Corasick with single-character wildcard support.

    Patterns may contain '?' which matches any single character.
    Uses a fragment-based approach: split each pattern on '?' chars,
    search for fragments, then verify that fragments align with wildcards.
    """

    def __init__(self, patterns=None, wildcard='?', case_sensitive=True):
        self._patterns = []
        self._wildcard = wildcard
        self._case_sensitive = case_sensitive
        self._fragments_info = []  # per pattern: [(fragment, offset_in_pattern), ...]
        self._ac = None
        self._built = False
        if patterns:
            for p in patterns:
                self.add_pattern(p)
            self.build()

    def add_pattern(self, pattern):
        if self._built:
            raise RuntimeError("Cannot add patterns after build()")
        idx = len(self._patterns)
        self._patterns.append(pattern)
        return idx

    def _split_fragments(self, pattern):
        """Split pattern on wildcard chars, tracking offsets."""
        fragments = []
        start = 0
        for i, ch in enumerate(pattern):
            if ch == self._wildcard:
                if i > start:
                    fragments.append((pattern[start:i], start))
                start = i + 1
        if start < len(pattern):
            fragments.append((pattern[start:], start))
        return fragments

    def build(self):
        if self._built:
            return

        # Collect unique fragments and build AC automaton
        all_fragments = set()
        self._fragments_info = []
        for pat in self._patterns:
            frags = self._split_fragments(pat)
            self._fragments_info.append(frags)
            for frag_text, _ in frags:
                all_fragments.add(frag_text if self._case_sensitive else frag_text.lower())

        self._ac = AhoCorasick(case_sensitive=self._case_sensitive)
        self._frag_to_idx = {}
        for frag in sorted(all_fragments):
            idx = self._ac.add_pattern(frag)
            self._frag_to_idx[frag if self._case_sensitive else frag.lower()] = idx
        self._ac.build()
        self._built = True

    def find_all(self, text):
        """Find all matches, including wildcards."""
        if not self._built:
            raise RuntimeError("Must call build() before searching")

        # Find all fragment matches
        frag_matches = self._ac.find_all(text)
        # Group fragment matches by their fragment pattern index
        frag_positions = {}  # frag_idx -> set of start positions
        for m in frag_matches:
            if m.pattern_idx not in frag_positions:
                frag_positions[m.pattern_idx] = set()
            frag_positions[m.pattern_idx].add(m.start)

        results = []
        norm_text = text if self._case_sensitive else text.lower()

        for pat_idx, pattern in enumerate(self._patterns):
            pat_len = len(pattern)
            frags = self._fragments_info[pat_idx]

            if not frags:
                # Pattern is all wildcards (e.g., "???")
                for start in range(len(text) - pat_len + 1):
                    results.append(Match(start, start + pat_len, pat_idx, pattern))
                continue

            # Use first fragment to get candidate start positions
            first_frag_text, first_frag_offset = frags[0]
            frag_key = first_frag_text if self._case_sensitive else first_frag_text.lower()
            frag_idx = self._frag_to_idx.get(frag_key)
            if frag_idx is None or frag_idx not in frag_positions:
                continue

            for frag_start in frag_positions[frag_idx]:
                # Candidate pattern start
                pat_start = frag_start - first_frag_offset
                if pat_start < 0 or pat_start + pat_len > len(text):
                    continue

                # Verify all fragments are at their expected positions
                valid = True
                for frag_text, frag_offset in frags[1:]:
                    expected_start = pat_start + frag_offset
                    fk = frag_text if self._case_sensitive else frag_text.lower()
                    fi = self._frag_to_idx.get(fk)
                    if fi is None or fi not in frag_positions:
                        valid = False
                        break
                    if expected_start not in frag_positions[fi]:
                        valid = False
                        break

                if valid:
                    results.append(Match(pat_start, pat_start + pat_len, pat_idx, pattern))

        # Sort by position
        results.sort(key=lambda m: (m.start, m.pattern_idx))
        return results

    def search(self, text):
        """Return True if any pattern matches."""
        return len(self.find_all(text)) > 0

    def find_first(self, text):
        """Find first match."""
        matches = self.find_all(text)
        return matches[0] if matches else None


# =============================================================================
# Utility: Pattern Set Operations
# =============================================================================

class ACPatternSet:
    """A set-like interface backed by Aho-Corasick for fast multi-membership testing."""

    def __init__(self, patterns=None, case_sensitive=True):
        self._ac = AhoCorasick(case_sensitive=case_sensitive)
        self._built = False
        if patterns is not None:
            for p in patterns:
                self._ac.add_pattern(p)
            self._ac.build()
            self._built = True

    def add(self, pattern):
        if self._built:
            raise RuntimeError("Cannot add to a built pattern set")
        self._ac.add_pattern(pattern)

    def build(self):
        self._ac.build()
        self._built = True

    def contains_any(self, text):
        """Check if text contains any pattern from the set."""
        return self._ac.search(text)

    def contains_all(self, text):
        """Check if text contains all patterns from the set."""
        found = self._ac.which_patterns(text)
        return len(found) == self._ac.pattern_count

    def which_found(self, text):
        """Return list of patterns found in text."""
        found = self._ac.which_patterns(text)
        return [self._ac._patterns[i] for i in sorted(found)]

    def filter_texts(self, texts):
        """Filter texts that contain at least one pattern."""
        return [t for t in texts if self._ac.search(t)]

    @property
    def pattern_count(self):
        return self._ac.pattern_count
