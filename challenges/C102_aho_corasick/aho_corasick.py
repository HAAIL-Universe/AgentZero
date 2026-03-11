"""
C102: Aho-Corasick Multi-Pattern String Matching

Aho-Corasick automaton for simultaneous search of multiple patterns in text.
Time: O(n + m + z) where n = text length, m = total pattern length, z = matches.

Features:
- Trie construction from pattern set
- BFS failure link computation (KMP generalized to trie)
- Dictionary links for efficient output collection
- Overlapping and non-overlapping match modes
- Pattern replacement with custom substitutions
- Streaming/incremental search
- Case-insensitive matching
- Wildcard patterns (single-char '?')
- Match callback interface
- Pattern group/category support
"""

from collections import deque


class AhoCorasickNode:
    """A node in the Aho-Corasick trie/automaton."""
    __slots__ = ('children', 'fail', 'dict_link', 'output', 'depth', 'char')

    def __init__(self, char='', depth=0):
        self.children = {}      # char -> AhoCorasickNode
        self.fail = None        # failure link (to longest proper suffix that is a prefix)
        self.dict_link = None   # dictionary suffix link (next output state via fail chain)
        self.output = []        # list of (pattern_index, pattern) for patterns ending here
        self.depth = depth      # depth in trie
        self.char = char        # character on edge to this node


class Match:
    """Represents a match found during search."""
    __slots__ = ('start', 'end', 'pattern', 'pattern_index', 'label')

    def __init__(self, start, end, pattern, pattern_index, label=None):
        self.start = start          # start position in text (inclusive)
        self.end = end              # end position in text (exclusive)
        self.pattern = pattern      # the matched pattern string
        self.pattern_index = pattern_index  # index in original pattern list
        self.label = label          # optional category/group label

    def __repr__(self):
        return f"Match(start={self.start}, end={self.end}, pattern={self.pattern!r})"

    def __eq__(self, other):
        if not isinstance(other, Match):
            return NotImplemented
        return (self.start == other.start and self.end == other.end and
                self.pattern == other.pattern and self.pattern_index == other.pattern_index)

    def __hash__(self):
        return hash((self.start, self.end, self.pattern, self.pattern_index))


class AhoCorasick:
    """
    Aho-Corasick multi-pattern string matcher.

    Usage:
        ac = AhoCorasick()
        ac.add_pattern("he")
        ac.add_pattern("she")
        ac.add_pattern("his")
        ac.add_pattern("hers")
        ac.build()
        matches = ac.search("ushers")
    """

    def __init__(self, case_sensitive=True):
        self.root = AhoCorasickNode()
        self.root.fail = self.root
        self.patterns = []          # list of (original_pattern, label)
        self.case_sensitive = case_sensitive
        self._built = False

    def _normalize(self, text):
        """Normalize text for case-insensitive matching."""
        if self.case_sensitive:
            return text
        return text.lower()

    def add_pattern(self, pattern, label=None):
        """Add a pattern to the automaton. Must call build() after all patterns added."""
        if self._built:
            raise RuntimeError("Cannot add patterns after build(). Create a new automaton.")
        if not pattern:
            raise ValueError("Pattern cannot be empty")

        pattern_index = len(self.patterns)
        self.patterns.append((pattern, label))

        normalized = self._normalize(pattern)
        node = self.root
        for i, ch in enumerate(normalized):
            if ch not in node.children:
                node.children[ch] = AhoCorasickNode(ch, i + 1)
            node = node.children[ch]

        node.output.append((pattern_index, pattern))
        return pattern_index

    def add_patterns(self, patterns, label=None):
        """Add multiple patterns at once. Each can be a string or (string, label) tuple."""
        indices = []
        for p in patterns:
            if isinstance(p, tuple):
                indices.append(self.add_pattern(p[0], p[1]))
            else:
                indices.append(self.add_pattern(p, label))
        return indices

    def build(self):
        """Build failure and dictionary links via BFS. Must be called before search."""
        if self._built:
            return

        queue = deque()

        # Initialize: depth-1 nodes fail to root
        for ch, child in self.root.children.items():
            child.fail = self.root
            child.dict_link = self.root if self.root.output else None
            queue.append(child)

        # BFS to build failure links
        while queue:
            node = queue.popleft()

            for ch, child in node.children.items():
                queue.append(child)

                # Follow failure links to find longest proper suffix
                fail = node.fail
                while fail is not self.root and ch not in fail.children:
                    fail = fail.fail

                child.fail = fail.children[ch] if ch in fail.children else self.root
                if child.fail is child:
                    child.fail = self.root

                # Dictionary link: nearest ancestor (via fail) with output
                if child.fail.output:
                    child.dict_link = child.fail
                else:
                    child.dict_link = child.fail.dict_link

        self._built = True

    def _collect_outputs(self, node, position):
        """Collect all pattern matches at a given node position."""
        matches = []
        # Direct outputs at this node
        for pidx, pat in node.output:
            label = self.patterns[pidx][1]
            start = position - len(pat) + 1
            matches.append(Match(start, position + 1, pat, pidx, label))

        # Follow dictionary links for shorter patterns
        dlink = node.dict_link
        while dlink and dlink is not self.root:
            for pidx, pat in dlink.output:
                label = self.patterns[pidx][1]
                start = position - len(pat) + 1
                matches.append(Match(start, position + 1, pat, pidx, label))
            dlink = dlink.dict_link

        return matches

    def search(self, text, overlapping=True):
        """
        Search text for all pattern occurrences.

        Args:
            text: The text to search in.
            overlapping: If True, return all overlapping matches.
                        If False, return only non-overlapping matches (leftmost-longest).

        Returns:
            List of Match objects sorted by position.
        """
        if not self._built:
            raise RuntimeError("Must call build() before search()")

        normalized = self._normalize(text)
        node = self.root
        all_matches = []

        for i, ch in enumerate(normalized):
            while node is not self.root and ch not in node.children:
                node = node.fail

            if ch in node.children:
                node = node.children[ch]
            # else: stay at root

            all_matches.extend(self._collect_outputs(node, i))

        all_matches.sort(key=lambda m: (m.start, -len(m.pattern)))

        if not overlapping:
            return self._filter_non_overlapping(all_matches)

        return all_matches

    def _filter_non_overlapping(self, matches):
        """Filter to non-overlapping matches (leftmost, then longest)."""
        result = []
        last_end = -1
        for m in matches:
            if m.start >= last_end:
                result.append(m)
                last_end = m.end
        return result

    def search_callback(self, text, callback):
        """
        Search text and call callback(match) for each match found.
        If callback returns False, stop searching.
        """
        if not self._built:
            raise RuntimeError("Must call build() before search()")

        normalized = self._normalize(text)
        node = self.root

        for i, ch in enumerate(normalized):
            while node is not self.root and ch not in node.children:
                node = node.fail

            if ch in node.children:
                node = node.children[ch]

            outputs = self._collect_outputs(node, i)
            for m in outputs:
                if callback(m) is False:
                    return

    def replace(self, text, replacements):
        """
        Replace matched patterns in text.

        Args:
            text: Input text.
            replacements: Dict mapping pattern string -> replacement string,
                         or a callable(Match) -> str.

        Returns:
            Text with replacements applied (non-overlapping, leftmost-longest).
        """
        matches = self.search(text, overlapping=False)

        if callable(replacements):
            replace_fn = replacements
        else:
            replace_fn = lambda m: replacements.get(m.pattern, m.pattern)

        result = []
        last_pos = 0
        for m in matches:
            result.append(text[last_pos:m.start])
            result.append(replace_fn(m))
            last_pos = m.end

        result.append(text[last_pos:])
        return ''.join(result)

    def contains_any(self, text):
        """Return True if text contains any of the patterns."""
        if not self._built:
            raise RuntimeError("Must call build() before search()")

        normalized = self._normalize(text)
        node = self.root

        for ch in normalized:
            while node is not self.root and ch not in node.children:
                node = node.fail

            if ch in node.children:
                node = node.children[ch]

            if node.output:
                return True
            if node.dict_link and node.dict_link is not self.root:
                return True

        return False

    def match_count(self, text):
        """Count total number of pattern occurrences (including overlapping)."""
        return len(self.search(text, overlapping=True))

    def matched_patterns(self, text):
        """Return set of pattern strings that appear in text."""
        matches = self.search(text, overlapping=True)
        return {m.pattern for m in matches}


class StreamingAhoCorasick:
    """
    Streaming/incremental Aho-Corasick matcher.

    Allows feeding text in chunks while maintaining automaton state.
    """

    def __init__(self, automaton):
        if not automaton._built:
            raise RuntimeError("Automaton must be built before streaming")
        self.automaton = automaton
        self.node = automaton.root
        self.position = 0       # global position across all fed chunks
        self.matches = []

    def feed(self, text):
        """
        Feed a chunk of text. Returns matches found in this chunk.
        Matches may span chunk boundaries.
        """
        normalized = self.automaton._normalize(text)
        chunk_matches = []

        for ch in normalized:
            node = self.node
            while node is not self.automaton.root and ch not in node.children:
                node = node.fail

            if ch in node.children:
                node = node.children[ch]

            self.node = node
            outputs = self.automaton._collect_outputs(node, self.position)
            chunk_matches.extend(outputs)
            self.position += 1

        self.matches.extend(chunk_matches)
        return chunk_matches

    def reset(self):
        """Reset streaming state."""
        self.node = self.automaton.root
        self.position = 0
        self.matches = []

    def get_all_matches(self):
        """Return all matches found so far across all chunks."""
        return list(self.matches)


class WildcardAhoCorasick:
    """
    Aho-Corasick variant supporting single-character wildcard '?' in patterns.

    Strategy: Split pattern by '?' into fragments, search for fragments,
    then verify that fragments appear at correct relative positions.
    """

    def __init__(self, case_sensitive=True):
        self.case_sensitive = case_sensitive
        self.patterns = []      # (original_pattern, label, fragments_info)
        self._built = False
        self._ac = None

    def _normalize(self, text):
        if self.case_sensitive:
            return text
        return text.lower()

    def add_pattern(self, pattern, label=None):
        """Add a pattern that may contain '?' wildcards."""
        if not pattern:
            raise ValueError("Pattern cannot be empty")

        # Parse pattern into fragments with positions
        fragments = []
        current = []
        pos = 0
        for ch in pattern:
            if ch == '?':
                if current:
                    frag = ''.join(current)
                    fragments.append((frag, pos - len(current)))
                    current = []
                pos += 1
            else:
                current.append(ch)
                pos += 1
        if current:
            frag = ''.join(current)
            fragments.append((frag, pos - len(current)))

        pidx = len(self.patterns)
        self.patterns.append((pattern, label, fragments, len(pattern)))
        return pidx

    def build(self):
        """Build the underlying automaton from pattern fragments."""
        if self._built:
            return

        self._ac = AhoCorasick(self.case_sensitive)

        # Collect all unique fragments
        self._frag_to_patterns = {}  # (frag_text) -> [(pidx, frag_offset)]
        for pidx, (pattern, label, fragments, plen) in enumerate(self.patterns):
            if not fragments:
                # Pattern is all wildcards -- handled separately
                continue
            for frag_text, frag_offset in fragments:
                key = self._normalize(frag_text)
                if key not in self._frag_to_patterns:
                    self._frag_to_patterns[key] = []
                self._frag_to_patterns[key].append((pidx, frag_offset))
                self._ac.add_pattern(frag_text)

        self._ac.build()
        self._built = True

    def search(self, text):
        """Search for wildcard patterns in text."""
        if not self._built:
            raise RuntimeError("Must call build() before search()")

        normalized = self._normalize(text)
        text_len = len(text)

        # For each pattern, track how many fragments have been found at each start position
        # pattern_hits[pidx][text_start] = set of fragment offsets found
        pattern_hits = {}
        for pidx, (pattern, label, fragments, plen) in enumerate(self.patterns):
            if not fragments:
                continue
            pattern_hits[pidx] = {}

        # Search for all fragments
        frag_matches = self._ac.search(text, overlapping=True)

        for fm in frag_matches:
            frag_key = self._normalize(fm.pattern)
            if frag_key not in self._frag_to_patterns:
                continue
            for pidx, frag_offset in self._frag_to_patterns[frag_key]:
                # This fragment at position fm.start means the pattern
                # would start at fm.start - frag_offset
                pattern_start = fm.start - frag_offset
                plen = self.patterns[pidx][3]
                if pattern_start < 0 or pattern_start + plen > text_len:
                    continue

                if pattern_start not in pattern_hits[pidx]:
                    pattern_hits[pidx][pattern_start] = set()
                pattern_hits[pidx][pattern_start].add(frag_offset)

        # Check which pattern starts have all fragments found
        matches = []
        for pidx, (pattern, label, fragments, plen) in enumerate(self.patterns):
            if not fragments:
                # All-wildcard pattern: matches everywhere it fits
                for start in range(text_len - plen + 1):
                    matches.append(Match(start, start + plen, pattern, pidx, label))
                continue

            needed = {frag_offset for _, frag_offset in fragments}
            if pidx not in pattern_hits:
                continue
            for start, found_offsets in pattern_hits[pidx].items():
                if needed <= found_offsets:
                    matches.append(Match(start, start + plen, pattern, pidx, label))

        matches.sort(key=lambda m: (m.start, -len(m.pattern)))
        return matches


class AhoCorasickSet:
    """
    Pattern set manager -- build once, query many times.
    Supports adding/removing patterns by rebuilding efficiently.
    """

    def __init__(self, case_sensitive=True):
        self.case_sensitive = case_sensitive
        self._patterns = {}     # name -> (pattern_string, label)
        self._ac = None
        self._dirty = True

    def add(self, name, pattern, label=None):
        """Add a named pattern."""
        self._patterns[name] = (pattern, label)
        self._dirty = True

    def remove(self, name):
        """Remove a named pattern."""
        if name in self._patterns:
            del self._patterns[name]
            self._dirty = True

    def _rebuild(self):
        """Rebuild automaton from current pattern set."""
        self._ac = AhoCorasick(self.case_sensitive)
        self._name_map = {}  # pattern_index -> name
        for name, (pattern, label) in self._patterns.items():
            idx = self._ac.add_pattern(pattern, label)
            self._name_map[idx] = name
        self._ac.build()
        self._dirty = False

    def search(self, text, overlapping=True):
        """Search text, returning matches with pattern names."""
        if self._dirty:
            self._rebuild()
        matches = self._ac.search(text, overlapping=overlapping)
        # Annotate with names
        for m in matches:
            m.label = self._name_map.get(m.pattern_index, m.label)
        return matches

    def contains_any(self, text):
        """Check if text contains any pattern."""
        if self._dirty:
            self._rebuild()
        return self._ac.contains_any(text)

    @property
    def pattern_count(self):
        return len(self._patterns)
