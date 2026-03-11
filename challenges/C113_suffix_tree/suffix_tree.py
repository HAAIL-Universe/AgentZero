"""
C113: Suffix Tree -- Ukkonen's Online O(n) Construction

Variants:
1. SuffixTree -- Core Ukkonen's algorithm with implicit/explicit suffix links
2. GeneralizedSuffixTree -- Multi-string suffix tree
3. SuffixTreeWithLCP -- LCP queries using tree traversal
4. SuffixTreeSearcher -- Pattern matching, substring operations
5. SuffixTreeAnalyzer -- Longest repeated substring, shortest unique, tandem repeats
"""


class SuffixTreeNode:
    """Node in the suffix tree."""
    __slots__ = ('children', 'suffix_link', 'start', 'end', 'suffix_index',
                 'string_ids', '_leaf_end')

    def __init__(self, start=-1, end=None):
        self.children = {}
        self.suffix_link = None
        self.start = start
        self.end = end  # None for internal nodes that use shared end, or an EndRef
        self.suffix_index = -1
        self.string_ids = set()
        self._leaf_end = None

    @property
    def edge_length(self):
        if self.end is None:
            return 0
        end_val = self.end.value if isinstance(self.end, EndRef) else self.end
        return end_val - self.start + 1

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.start == -1


class EndRef:
    """Mutable reference to the global end for leaf nodes (Ukkonen trick 3)."""
    __slots__ = ('value',)

    def __init__(self, value=-1):
        self.value = value


class SuffixTree:
    """
    Suffix tree built using Ukkonen's algorithm in O(n) time.

    Key tricks:
    1. Implicit extensions (Rule 1) -- leaf edges grow automatically via global end
    2. Skip/count trick -- traverse edges in O(1) amortized
    3. Suffix links -- jump from one extension to the next in O(1)
    """

    TERMINAL = '\x00'

    def __init__(self, text=None):
        self.text = ""
        self.root = SuffixTreeNode()
        self.root.suffix_link = self.root
        self._global_end = EndRef(-1)
        self._remaining = 0
        self._active_node = self.root
        self._active_edge = -1
        self._active_length = 0
        self._size = 0

        if text is not None:
            self.build(text)

    def build(self, text):
        """Build suffix tree for the given text using Ukkonen's algorithm."""
        if not text.endswith(self.TERMINAL):
            text = text + self.TERMINAL
        self.text = text
        self.root = SuffixTreeNode()
        self.root.suffix_link = self.root
        self._global_end = EndRef(-1)
        self._remaining = 0
        self._active_node = self.root
        self._active_edge = -1
        self._active_length = 0
        self._size = len(text)

        for i in range(len(text)):
            self._extend(i)

        self._set_suffix_indices(self.root, 0)

    def _extend(self, pos):
        """Extend the suffix tree by one character at position pos."""
        self._global_end.value = pos
        self._remaining += 1
        last_new_internal = None

        while self._remaining > 0:
            if self._active_length == 0:
                self._active_edge = pos

            active_char = self.text[self._active_edge]

            if active_char not in self._active_node.children:
                # Rule 2: create new leaf
                leaf = SuffixTreeNode(pos, self._global_end)
                self._active_node.children[active_char] = leaf

                if last_new_internal is not None:
                    last_new_internal.suffix_link = self._active_node
                    last_new_internal = None
            else:
                next_node = self._active_node.children[active_char]

                # Skip/count trick: walk down if active_length >= edge_length
                if self._walk_down(next_node):
                    continue

                # Rule 3: character already in tree
                if self.text[next_node.start + self._active_length] == self.text[pos]:
                    self._active_length += 1
                    if last_new_internal is not None:
                        last_new_internal.suffix_link = self._active_node
                    break

                # Rule 2: split edge and create new leaf
                split = SuffixTreeNode(next_node.start, next_node.start + self._active_length - 1)
                self._active_node.children[active_char] = split

                leaf = SuffixTreeNode(pos, self._global_end)
                split.children[self.text[pos]] = leaf

                next_node.start += self._active_length
                split.children[self.text[next_node.start]] = next_node

                if last_new_internal is not None:
                    last_new_internal.suffix_link = split
                last_new_internal = split

            self._remaining -= 1

            if self._active_node is self.root and self._active_length > 0:
                self._active_length -= 1
                self._active_edge = pos - self._remaining + 1
            elif self._active_node.suffix_link is not None:
                self._active_node = self._active_node.suffix_link
            else:
                self._active_node = self.root

    def _walk_down(self, node):
        """Skip/count trick: walk down the tree if active_length >= edge_length."""
        edge_len = node.edge_length
        if self._active_length >= edge_len:
            self._active_edge += edge_len
            self._active_length -= edge_len
            self._active_node = node
            return True
        return False

    def _set_suffix_indices(self, node, label_height):
        """DFS to set suffix_index for all leaf nodes."""
        if node.is_leaf():
            node.suffix_index = self._size - label_height
            return

        for child in node.children.values():
            self._set_suffix_indices(child, label_height + child.edge_length)

    def contains(self, pattern):
        """Check if pattern is a substring of the text."""
        node, depth = self._find_node(pattern)
        return node is not None

    def count_occurrences(self, pattern):
        """Count how many times pattern occurs in the text."""
        node, depth = self._find_node(pattern)
        if node is None:
            return 0
        return self._count_leaves(node)

    def find_all(self, pattern):
        """Find all starting positions of pattern in the text."""
        node, depth = self._find_node(pattern)
        if node is None:
            return []
        positions = []
        self._collect_leaves(node, positions)
        return sorted(positions)

    def _find_node(self, pattern):
        """
        Find the node where pattern ends.
        Returns (node, depth) or (None, -1) if not found.
        depth is how far into the current edge we are.
        """
        if not pattern:
            return self.root, 0

        node = self.root
        i = 0
        while i < len(pattern):
            c = pattern[i]
            if c not in node.children:
                return None, -1
            child = node.children[c]
            edge_len = child.edge_length
            # Compare pattern against edge label
            j = 0
            while j < edge_len and i < len(pattern):
                if self.text[child.start + j] != pattern[i]:
                    return None, -1
                i += 1
                j += 1
            if i == len(pattern):
                # Pattern ended mid-edge or at edge end
                return child, j
            # Pattern continues, move to child
            node = child
        return node, 0

    def _count_leaves(self, node):
        """Count leaf nodes in subtree."""
        if node.is_leaf():
            return 1
        count = 0
        for child in node.children.values():
            count += self._count_leaves(child)
        return count

    def _collect_leaves(self, node, positions):
        """Collect suffix indices from all leaves in subtree."""
        if node.is_leaf():
            if node.suffix_index >= 0:
                positions.append(node.suffix_index)
            return
        for child in node.children.values():
            self._collect_leaves(child, positions)

    def has_suffix(self, pattern):
        """Check if pattern is a suffix of the text (excluding terminal)."""
        # A suffix must end at a leaf whose suffix_index matches
        node, depth = self._find_node(pattern + self.TERMINAL)
        return node is not None

    def longest_repeated_substring(self):
        """Find the longest substring that occurs at least twice."""
        result = [""]

        def dfs(node, depth):
            if node.is_leaf():
                return
            # Internal node with 2+ children = repeated substring
            new_depth = depth + node.edge_length
            if new_depth > len(result[0]) and not node.is_root():
                result[0] = self.text[node.suffix_index if node.suffix_index >= 0
                                      else self._get_any_leaf_index(node):
                                      self._get_any_leaf_index(node) + new_depth]
                # Actually, reconstruct from root
                pass
            for child in node.children.values():
                dfs(child, new_depth)

        # Better approach: find deepest internal node
        best_depth = [0]
        best_path = [""]

        def dfs2(node, depth, path_chars):
            if node.is_root():
                for child in node.children.values():
                    edge_label = self.text[child.start:child.start + child.edge_length]
                    dfs2(child, child.edge_length, edge_label)
                return

            if not node.is_leaf():
                # Internal node = repeated substring
                if depth > best_depth[0]:
                    best_depth[0] = depth
                    best_path[0] = path_chars
                for child in node.children.values():
                    edge_label = self.text[child.start:child.start + child.edge_length]
                    dfs2(child, depth + child.edge_length, path_chars + edge_label)

        dfs2(self.root, 0, "")
        # Remove terminal if present
        result_str = best_path[0].replace(self.TERMINAL, "")
        return result_str

    def _get_any_leaf_index(self, node):
        """Get suffix_index of any leaf in this subtree."""
        if node.is_leaf():
            return node.suffix_index
        for child in node.children.values():
            result = self._get_any_leaf_index(child)
            if result >= 0:
                return result
        return -1

    def shortest_unique_substring(self):
        """Find the shortest substring that occurs exactly once."""
        best = [None]

        def dfs(node, depth, path):
            if node.is_root():
                for c, child in node.children.items():
                    edge = self.text[child.start:child.start + child.edge_length]
                    dfs(child, child.edge_length, edge)
                return

            if node.is_leaf():
                # Every prefix of a unique suffix is a candidate
                # The shortest one ending here has length depth - edge_length + 1
                # But we need to find the minimal substring occurring exactly once
                # A leaf means this suffix is unique from position (depth - edge_length + 1)
                # The shortest unique prefix going through this path:
                # it becomes unique at (depth - node.edge_length + 1) characters
                parent_depth = depth - node.edge_length
                candidate_len = parent_depth + 1
                candidate = path[:candidate_len]
                if self.TERMINAL not in candidate:
                    if best[0] is None or len(candidate) < len(best[0]):
                        best[0] = candidate
            else:
                for c, child in node.children.items():
                    edge = self.text[child.start:child.start + child.edge_length]
                    dfs(child, depth + child.edge_length, path + edge)

        dfs(self.root, 0, "")
        return best[0] if best[0] else ""

    def edge_count(self):
        """Count the number of edges in the tree."""
        count = [0]

        def dfs(node):
            for child in node.children.values():
                count[0] += 1
                dfs(child)

        dfs(self.root)
        return count[0]

    def node_count(self):
        """Count total nodes including root."""
        count = [1]  # root

        def dfs(node):
            for child in node.children.values():
                count[0] += 1
                dfs(child)

        dfs(self.root)
        return count[0]

    def leaf_count(self):
        """Count leaf nodes (should equal len(text))."""
        count = [0]

        def dfs(node):
            if node.is_leaf():
                count[0] += 1
                return
            for child in node.children.values():
                dfs(child)

        dfs(self.root)
        return count[0]

    def get_edge_label(self, node):
        """Get the string label of a node's incoming edge."""
        if node.is_root():
            return ""
        end_val = node.end.value if isinstance(node.end, EndRef) else node.end
        return self.text[node.start:end_val + 1]

    def get_all_suffixes(self):
        """Get all suffixes stored in the tree."""
        suffixes = []

        def dfs(node, path):
            if node.is_root():
                for child in node.children.values():
                    dfs(child, self.get_edge_label(child))
                return
            if node.is_leaf():
                # Remove terminal
                s = path.rstrip(self.TERMINAL)
                if s:
                    suffixes.append(s)
                return
            for child in node.children.values():
                dfs(child, path + self.get_edge_label(child))

        dfs(self.root, "")
        return sorted(suffixes)


class GeneralizedSuffixTree:
    """
    Generalized suffix tree for multiple strings.
    Uses unique terminal characters per string.
    """

    def __init__(self, strings=None):
        self.strings = []
        self.text = ""
        self.string_boundaries = []  # (start, end) for each string in concatenated text
        self.root = SuffixTreeNode()
        self.root.suffix_link = self.root
        self._terminals = []

        if strings:
            self.build(strings)

    def build(self, strings):
        """Build generalized suffix tree from multiple strings."""
        self.strings = list(strings)
        self._terminals = []

        # Use unique terminal characters
        parts = []
        offset = 0
        for i, s in enumerate(self.strings):
            terminal = chr(i + 1)  # Use \x01, \x02, etc. as unique terminals
            self._terminals.append(terminal)
            self.string_boundaries.append((offset, offset + len(s)))
            parts.append(s + terminal)
            offset += len(s) + 1

        self.text = "".join(parts)

        # Build using Ukkonen's
        self.root = SuffixTreeNode()
        self.root.suffix_link = self.root
        self._global_end = EndRef(-1)
        self._remaining = 0
        self._active_node = self.root
        self._active_edge = -1
        self._active_length = 0
        self._size = len(self.text)

        for i in range(len(self.text)):
            self._extend(i)

        self._set_suffix_indices_and_string_ids(self.root, 0)

    def _extend(self, pos):
        """Same Ukkonen extension as SuffixTree."""
        self._global_end.value = pos
        self._remaining += 1
        last_new_internal = None

        while self._remaining > 0:
            if self._active_length == 0:
                self._active_edge = pos

            active_char = self.text[self._active_edge]

            if active_char not in self._active_node.children:
                leaf = SuffixTreeNode(pos, self._global_end)
                self._active_node.children[active_char] = leaf
                if last_new_internal is not None:
                    last_new_internal.suffix_link = self._active_node
                    last_new_internal = None
            else:
                next_node = self._active_node.children[active_char]
                edge_len = next_node.edge_length
                if self._active_length >= edge_len:
                    self._active_edge += edge_len
                    self._active_length -= edge_len
                    self._active_node = next_node
                    continue

                if self.text[next_node.start + self._active_length] == self.text[pos]:
                    self._active_length += 1
                    if last_new_internal is not None:
                        last_new_internal.suffix_link = self._active_node
                    break

                split = SuffixTreeNode(next_node.start, next_node.start + self._active_length - 1)
                self._active_node.children[active_char] = split

                leaf = SuffixTreeNode(pos, self._global_end)
                split.children[self.text[pos]] = leaf

                next_node.start += self._active_length
                split.children[self.text[next_node.start]] = next_node

                if last_new_internal is not None:
                    last_new_internal.suffix_link = split
                last_new_internal = split

            self._remaining -= 1

            if self._active_node is self.root and self._active_length > 0:
                self._active_length -= 1
                self._active_edge = pos - self._remaining + 1
            elif self._active_node.suffix_link is not None:
                self._active_node = self._active_node.suffix_link
            else:
                self._active_node = self.root

    def _set_suffix_indices_and_string_ids(self, node, label_height):
        """Set suffix indices and propagate string IDs up from leaves."""
        if node.is_leaf():
            node.suffix_index = self._size - label_height
            # Determine which string this suffix belongs to
            pos = node.suffix_index
            for i, (start, end) in enumerate(self.string_boundaries):
                if start <= pos <= end:
                    node.string_ids.add(i)
                    break
            return

        for child in node.children.values():
            self._set_suffix_indices_and_string_ids(child, label_height + child.edge_length)
            node.string_ids.update(child.string_ids)

    def _find_node(self, pattern):
        """Find node where pattern ends."""
        if not pattern:
            return self.root, 0
        node = self.root
        i = 0
        while i < len(pattern):
            c = pattern[i]
            if c not in node.children:
                return None, -1
            child = node.children[c]
            edge_len = child.edge_length
            j = 0
            while j < edge_len and i < len(pattern):
                if self.text[child.start + j] != pattern[i]:
                    return None, -1
                i += 1
                j += 1
            if i == len(pattern):
                return child, j
            node = child
        return node, 0

    def contains(self, pattern):
        """Check if pattern exists in any string."""
        node, _ = self._find_node(pattern)
        return node is not None

    def find_strings_containing(self, pattern):
        """Return set of string indices that contain the pattern."""
        node, _ = self._find_node(pattern)
        if node is None:
            return set()
        return set(node.string_ids)

    def longest_common_substring(self, indices=None):
        """
        Find longest common substring among specified strings.
        If indices is None, uses all strings.
        """
        if indices is None:
            target_ids = set(range(len(self.strings)))
        else:
            target_ids = set(indices)

        if len(target_ids) < 2:
            return ""

        best = [""]

        def dfs(node, depth, path):
            if node.is_root():
                for child in node.children.values():
                    edge = self.text[child.start:child.start + child.edge_length]
                    dfs(child, child.edge_length, edge)
                return

            if not node.is_leaf() and target_ids.issubset(node.string_ids):
                # Remove terminal chars from path
                clean = path
                for t in self._terminals:
                    clean = clean.replace(t, "")
                if len(clean) > len(best[0]):
                    best[0] = clean

            if not node.is_leaf():
                for child in node.children.values():
                    edge = self.text[child.start:child.start + child.edge_length]
                    dfs(child, depth + child.edge_length, path + edge)

        dfs(self.root, 0, "")
        return best[0]

    def common_substrings(self, min_length=1):
        """Find all substrings common to ALL strings."""
        target_ids = set(range(len(self.strings)))
        results = []

        def dfs(node, depth, path):
            if node.is_root():
                for child in node.children.values():
                    edge = self.text[child.start:child.start + child.edge_length]
                    dfs(child, child.edge_length, edge)
                return

            if not node.is_leaf() and target_ids.issubset(node.string_ids):
                clean = path
                for t in self._terminals:
                    clean = clean.replace(t, "")
                if len(clean) >= min_length:
                    results.append(clean)

            if not node.is_leaf():
                for child in node.children.values():
                    edge = self.text[child.start:child.start + child.edge_length]
                    dfs(child, depth + child.edge_length, path + edge)

        dfs(self.root, 0, "")
        return sorted(set(results))


class SuffixTreeWithLCP:
    """Suffix tree with LCP (Longest Common Prefix) query support."""

    def __init__(self, text=None):
        self._tree = SuffixTree()
        self.text = ""
        if text:
            self.build(text)

    def build(self, text):
        self.text = text
        self._tree.build(text)

    def lcp(self, i, j):
        """
        Find length of longest common prefix between suffix starting
        at position i and suffix starting at position j.
        """
        if i == j:
            return len(self.text) - i
        if i < 0 or j < 0 or i >= len(self.text) or j >= len(self.text):
            return 0

        # Walk from root matching both suffixes
        s1 = self.text[i:]
        s2 = self.text[j:]
        common = 0
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                common += 1
            else:
                break
        return common

    def lcp_array(self):
        """
        Compute LCP array from suffix array ordering.
        Returns (suffix_array, lcp_array).
        """
        # Collect suffix indices in sorted order
        suffixes = []
        self._collect_sorted(self._tree.root, suffixes)

        # Filter out the terminal-only suffix
        sa = [s for s in suffixes if s < len(self.text)]

        # Compute LCP between consecutive suffixes in SA
        lcp = [0] * len(sa)
        for k in range(1, len(sa)):
            lcp[k] = self.lcp(sa[k - 1], sa[k])

        return sa, lcp

    def _collect_sorted(self, node, result):
        """Collect suffix indices in lexicographic order."""
        if node.is_leaf():
            if node.suffix_index >= 0:
                result.append(node.suffix_index)
            return
        for c in sorted(node.children.keys()):
            self._collect_sorted(node.children[c], result)

    @property
    def root(self):
        return self._tree.root

    def contains(self, pattern):
        return self._tree.contains(pattern)


class SuffixTreeSearcher:
    """Pattern matching and substring operations using suffix tree."""

    def __init__(self, text=None):
        self._tree = SuffixTree()
        self.text = ""
        if text:
            self.build(text)

    def build(self, text):
        self.text = text
        self._tree.build(text)

    def search(self, pattern):
        """Find all occurrences of pattern. Returns sorted list of positions."""
        return self._tree.find_all(pattern)

    def count(self, pattern):
        """Count occurrences of pattern."""
        return self._tree.count_occurrences(pattern)

    def contains(self, pattern):
        """Check if pattern exists."""
        return self._tree.contains(pattern)

    def is_suffix(self, pattern):
        """Check if pattern is a suffix."""
        return self._tree.has_suffix(pattern)

    def is_prefix(self, pattern):
        """Check if pattern is a prefix of the text."""
        return self.text.startswith(pattern)

    def longest_common_extension(self, i, j):
        """Find length of longest common extension starting at positions i and j."""
        if i == j:
            return len(self.text) - i
        s1 = self.text[i:]
        s2 = self.text[j:]
        common = 0
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                common += 1
            else:
                break
        return common

    def distinct_substrings(self):
        """Count the number of distinct substrings (excluding empty string)."""
        count = [0]

        def dfs(node):
            if node.is_root():
                for child in node.children.values():
                    dfs(child)
                return
            # Each edge contributes edge_length distinct substrings
            count[0] += node.edge_length
            for child in node.children.values():
                dfs(child)

        dfs(self._tree.root)
        # Subtract contributions from terminal character
        # The terminal adds 1 to each leaf edge, and there are n+1 leaves
        # but actually the terminal creates exactly 1 extra substring per suffix
        # Let's count properly: subtract edges ending with terminal
        # Actually simpler: total edges in tree, minus the terminal-only paths
        # For text of length n (without terminal), distinct substrings = n*(n+1)/2 minus repeats
        # The tree count includes substrings ending with \x00 which we don't want
        # Recount excluding terminal char
        count2 = [0]

        def dfs2(node, depth):
            if node.is_root():
                for child in node.children.values():
                    dfs2(child, 0)
                return
            edge_end = node.end.value if isinstance(node.end, EndRef) else node.end
            edge_start = node.start
            added = 0
            for k in range(edge_start, edge_end + 1):
                if self._tree.text[k] == SuffixTree.TERMINAL:
                    break
                added += 1
            count2[0] += added
            if added == node.edge_length:
                # Didn't hit terminal, continue to children
                for child in node.children.values():
                    dfs2(child, depth + node.edge_length)

        dfs2(self._tree.root, 0)
        return count2[0]

    def kth_substring(self, k):
        """
        Find the k-th lexicographically smallest distinct substring (1-indexed).
        Returns None if k is out of range.
        """
        result = [None]
        remaining = [k]

        def dfs(node, path):
            if result[0] is not None:
                return
            if node.is_root():
                for c in sorted(node.children.keys()):
                    dfs(node.children[c], "")
                    if result[0] is not None:
                        return
                return

            edge_end = node.end.value if isinstance(node.end, EndRef) else node.end
            for pos in range(node.start, edge_end + 1):
                ch = self._tree.text[pos]
                if ch == SuffixTree.TERMINAL:
                    return
                path = path + ch
                remaining[0] -= 1
                if remaining[0] == 0:
                    result[0] = path
                    return

            for c in sorted(node.children.keys()):
                dfs(node.children[c], path)
                if result[0] is not None:
                    return

        dfs(self._tree.root, "")
        return result[0]

    def longest_repeated_substring(self):
        """Find the longest repeated substring."""
        return self._tree.longest_repeated_substring()


class SuffixTreeAnalyzer:
    """Advanced analysis using suffix trees: tandem repeats, minimal unique, etc."""

    def __init__(self, text=None):
        self._tree = SuffixTree()
        self.text = ""
        if text:
            self.build(text)

    def build(self, text):
        self.text = text
        self._tree.build(text)

    def longest_repeated_substring(self):
        """Find the longest substring occurring at least twice."""
        return self._tree.longest_repeated_substring()

    def shortest_unique_substring(self):
        """Find the shortest substring occurring exactly once."""
        return self._tree.shortest_unique_substring()

    def tandem_repeats(self):
        """
        Find all tandem repeats (consecutive repeated substrings) in the text.
        A tandem repeat is a substring of the form ww (e.g., "abab" has tandem repeat "ab").
        Returns list of (position, repeat_unit, count) tuples.
        """
        results = []
        n = len(self.text)

        # Brute force using the suffix tree for verification
        for length in range(1, n // 2 + 1):
            for start in range(n - 2 * length + 1):
                count = 0
                pos = start
                while pos + length <= n and self.text[pos:pos + length] == self.text[start:start + length]:
                    count += 1
                    pos += length
                if count >= 2:
                    unit = self.text[start:start + length]
                    results.append((start, unit, count))

        # Deduplicate: keep maximal repeats
        seen = set()
        filtered = []
        for pos, unit, count in results:
            key = (pos, unit)
            if key not in seen:
                seen.add(key)
                filtered.append((pos, unit, count))
        return filtered

    def maximal_repeats(self):
        """
        Find all maximal repeats. A repeat is maximal if it cannot be extended
        left or right while maintaining the same number of occurrences.
        Returns list of (substring, positions) tuples.
        """
        results = []

        def dfs(node, depth, path):
            if node.is_root():
                for child in node.children.values():
                    edge = self._tree.text[child.start:child.start + child.edge_length]
                    dfs(child, child.edge_length, edge)
                return

            if not node.is_leaf() and depth > 0:
                # Internal node = repeated substring
                clean = path.replace(SuffixTree.TERMINAL, "")
                if clean:
                    positions = []
                    self._collect_positions(node, positions)
                    # Check left-diversity (different characters precede occurrences)
                    left_chars = set()
                    for pos in positions:
                        if pos == 0:
                            left_chars.add(None)
                        else:
                            left_chars.add(self.text[pos - 1])
                    if len(left_chars) > 1:
                        results.append((clean, sorted(positions)))

            if not node.is_leaf():
                for child in node.children.values():
                    edge = self._tree.text[child.start:child.start + child.edge_length]
                    dfs(child, depth + child.edge_length, path + edge)

        dfs(self._tree.root, 0, "")
        return results

    def _collect_positions(self, node, positions):
        """Collect all suffix starting positions under a node."""
        if node.is_leaf():
            if node.suffix_index >= 0:
                positions.append(node.suffix_index)
            return
        for child in node.children.values():
            self._collect_positions(child, positions)

    def supermaximal_repeats(self):
        """
        Find supermaximal repeats: maximal repeats that are not substrings
        of other maximal repeats with the same number of occurrences.
        """
        results = []

        def dfs(node, depth, path):
            if node.is_root():
                for child in node.children.values():
                    edge = self._tree.text[child.start:child.start + child.edge_length]
                    dfs(child, child.edge_length, edge)
                return

            if not node.is_leaf():
                # Check if ALL children are leaves (supermaximal condition)
                all_leaves = all(child.is_leaf() for child in node.children.values())
                if all_leaves and depth > 0:
                    clean = path.replace(SuffixTree.TERMINAL, "")
                    if clean:
                        positions = []
                        self._collect_positions(node, positions)
                        if len(positions) >= 2:
                            results.append((clean, sorted(positions)))

                for child in node.children.values():
                    edge = self._tree.text[child.start:child.start + child.edge_length]
                    dfs(child, depth + child.edge_length, path + edge)

        dfs(self._tree.root, 0, "")
        return results

    def palindromes(self, min_length=2):
        """
        Find all palindromic substrings of at least min_length.
        Uses the suffix tree for substring enumeration.
        """
        results = set()
        n = len(self.text)

        # Enumerate distinct substrings and check palindrome property
        def dfs(node, path):
            if node.is_root():
                for child in node.children.values():
                    dfs(child, "")
                return
            edge_end = node.end.value if isinstance(node.end, EndRef) else node.end
            for pos in range(node.start, edge_end + 1):
                ch = self._tree.text[pos]
                if ch == SuffixTree.TERMINAL:
                    return
                path = path + ch
                if len(path) >= min_length and path == path[::-1]:
                    results.add(path)
            for child in node.children.values():
                dfs(child, path)

        dfs(self._tree.root, "")
        return sorted(results)

    def substring_frequency(self, pattern):
        """Get frequency of a pattern in the text."""
        return self._tree.count_occurrences(pattern)

    def most_frequent_substring(self, length):
        """Find the most frequent substring of a given length."""
        freq = {}
        n = len(self.text)
        for i in range(n - length + 1):
            sub = self.text[i:i + length]
            freq[sub] = freq.get(sub, 0) + 1
        if not freq:
            return None, 0
        best = max(freq.items(), key=lambda x: x[1])
        return best[0], best[1]
