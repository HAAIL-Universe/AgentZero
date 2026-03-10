"""C085: Trie / Radix Tree -- Prefix tree data structures.

Four variants:
1. Trie -- standard trie (insert/search/delete/prefix/autocomplete/wildcard)
2. RadixTree -- compressed Patricia trie (edge-label compression)
3. PersistentTrie -- immutable trie with path-copying
4. TernarySearchTree -- space-efficient BST-of-characters
"""


# =============================================================================
# Variant 1: Standard Trie
# =============================================================================

class TrieNode:
    __slots__ = ('children', 'is_end', 'value', 'count')

    def __init__(self):
        self.children = {}     # char -> TrieNode
        self.is_end = False
        self.value = None      # optional associated value
        self.count = 0         # number of words through this node


class Trie:
    """Standard trie with string keys and optional values."""

    def __init__(self):
        self.root = TrieNode()
        self._size = 0

    def __len__(self):
        return self._size

    def __contains__(self, key):
        return self.search(key)

    def __bool__(self):
        return self._size > 0

    def insert(self, key, value=True):
        """Insert key with optional value. Returns previous value or None."""
        node = self.root
        for ch in key:
            node.count += 1
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        old = node.value if node.is_end else None
        if not node.is_end:
            self._size += 1
        node.is_end = True
        node.count += 1 if old is None else 0
        node.value = value
        return old

    def search(self, key):
        """Return True if key exists."""
        node = self._find_node(key)
        return node is not None and node.is_end

    def get(self, key, default=None):
        """Get value for key, or default."""
        node = self._find_node(key)
        if node is not None and node.is_end:
            return node.value
        return default

    def delete(self, key):
        """Delete key. Returns True if key existed."""
        # Find path, then clean up
        path = []
        node = self.root
        for ch in key:
            if ch not in node.children:
                return False
            path.append((node, ch))
            node = node.children[ch]
        if not node.is_end:
            return False
        node.is_end = False
        node.value = None
        self._size -= 1
        # Decrement counts and prune empty branches
        for parent, ch in path:
            parent.count -= 1
        node.count -= 1
        # Prune from bottom up
        for parent, ch in reversed(path):
            child = parent.children[ch]
            if not child.children and not child.is_end:
                del parent.children[ch]
            else:
                break
        return True

    def starts_with(self, prefix):
        """Return True if any key starts with prefix."""
        return self._find_node(prefix) is not None

    def keys_with_prefix(self, prefix):
        """Return all keys starting with prefix."""
        node = self._find_node(prefix)
        if node is None:
            return []
        results = []
        self._collect(node, list(prefix), results)
        return results

    def count_with_prefix(self, prefix):
        """Count keys starting with prefix."""
        node = self._find_node(prefix)
        if node is None:
            return 0
        return node.count

    def autocomplete(self, prefix, limit=10):
        """Return up to limit keys starting with prefix."""
        node = self._find_node(prefix)
        if node is None:
            return []
        results = []
        self._collect(node, list(prefix), results, limit)
        return results

    def wildcard_search(self, pattern):
        """Search with '.' as single-char wildcard. Returns matching keys."""
        results = []
        self._wildcard(self.root, pattern, 0, [], results)
        return results

    def longest_prefix_of(self, text):
        """Return the longest key that is a prefix of text."""
        node = self.root
        last_match = ""
        current = []
        for ch in text:
            if ch not in node.children:
                break
            current.append(ch)
            node = node.children[ch]
            if node.is_end:
                last_match = ''.join(current)
        return last_match

    def all_keys(self):
        """Return all keys in the trie."""
        results = []
        self._collect(self.root, [], results)
        return results

    def items(self):
        """Return all (key, value) pairs."""
        results = []
        self._collect_items(self.root, [], results)
        return results

    def _find_node(self, key):
        node = self.root
        for ch in key:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def _collect(self, node, path, results, limit=None):
        if limit is not None and len(results) >= limit:
            return
        if node.is_end:
            results.append(''.join(path))
        for ch in sorted(node.children):
            if limit is not None and len(results) >= limit:
                return
            path.append(ch)
            self._collect(node.children[ch], path, results, limit)
            path.pop()

    def _collect_items(self, node, path, results):
        if node.is_end:
            results.append((''.join(path), node.value))
        for ch in sorted(node.children):
            path.append(ch)
            self._collect_items(node.children[ch], path, results)
            path.pop()

    def _wildcard(self, node, pattern, idx, path, results):
        if idx == len(pattern):
            if node.is_end:
                results.append(''.join(path))
            return
        ch = pattern[idx]
        if ch == '.':
            for c in sorted(node.children):
                path.append(c)
                self._wildcard(node.children[c], pattern, idx + 1, path, results)
                path.pop()
        else:
            if ch in node.children:
                path.append(ch)
                self._wildcard(node.children[ch], pattern, idx + 1, path, results)
                path.pop()


# =============================================================================
# Variant 2: Radix Tree (Patricia Trie / Compressed Trie)
# =============================================================================

class RadixNode:
    __slots__ = ('children', 'is_end', 'value', 'prefix')

    def __init__(self, prefix=""):
        self.prefix = prefix       # edge label (compressed)
        self.children = {}         # first-char -> RadixNode
        self.is_end = False
        self.value = None


class RadixTree:
    """Compressed trie -- edges carry multi-char labels."""

    def __init__(self):
        self.root = RadixNode()
        self._size = 0

    def __len__(self):
        return self._size

    def __contains__(self, key):
        return self.search(key)

    def insert(self, key, value=True):
        """Insert key with optional value."""
        node = self.root
        remaining = key

        while remaining:
            first = remaining[0]
            if first not in node.children:
                # No matching child -- create new leaf
                child = RadixNode(remaining)
                child.is_end = True
                child.value = value
                node.children[first] = child
                self._size += 1
                return None

            child = node.children[first]
            # Find common prefix length
            cp = _common_prefix_len(remaining, child.prefix)

            if cp == len(child.prefix):
                # Child prefix is fully consumed
                remaining = remaining[cp:]
                if not remaining:
                    # Exact match
                    old = child.value if child.is_end else None
                    if not child.is_end:
                        self._size += 1
                    child.is_end = True
                    child.value = value
                    return old
                node = child
            else:
                # Split the child
                # child.prefix = common + child_rest
                common = child.prefix[:cp]
                child_rest = child.prefix[cp:]
                key_rest = remaining[cp:]

                # New intermediate node
                split = RadixNode(common)
                node.children[first] = split

                # Old child becomes child of split
                child.prefix = child_rest
                split.children[child_rest[0]] = child

                if key_rest:
                    # New leaf for remaining key
                    leaf = RadixNode(key_rest)
                    leaf.is_end = True
                    leaf.value = value
                    split.children[key_rest[0]] = leaf
                else:
                    # The split point is exactly the key
                    split.is_end = True
                    split.value = value

                self._size += 1
                return None

        # Empty remaining means key ended at current node
        old = node.value if node.is_end else None
        if not node.is_end:
            self._size += 1
        node.is_end = True
        node.value = value
        return old

    def search(self, key):
        """Return True if key exists."""
        node, remaining = self._find(key)
        return node is not None and remaining == "" and node.is_end

    def get(self, key, default=None):
        """Get value for key."""
        node, remaining = self._find(key)
        if node is not None and remaining == "" and node.is_end:
            return node.value
        return default

    def delete(self, key):
        """Delete key. Returns True if deleted."""
        # Use recursive approach for cleanup
        deleted = self._delete(self.root, key)
        if deleted:
            self._size -= 1
        return deleted

    def starts_with(self, prefix):
        """Return True if any key starts with prefix."""
        node, remaining = self._find(prefix)
        if node is not None and remaining == "":
            return True
        # Check if remaining is a prefix of a child's edge
        if node is not None and remaining:
            first = remaining[0]
            if first in node.children:
                child = node.children[first]
                if child.prefix.startswith(remaining):
                    return True
        return False

    def keys_with_prefix(self, prefix):
        """Return all keys starting with prefix."""
        results = []
        node, remaining = self._find(prefix)
        if node is not None and remaining == "":
            self._collect_radix(node, list(prefix), results)
        elif node is not None and remaining:
            first = remaining[0]
            if first in node.children:
                child = node.children[first]
                if child.prefix.startswith(remaining):
                    full_prefix = list(prefix) + list(child.prefix[len(remaining):])
                    self._collect_radix(child, full_prefix, results)
        return results

    def longest_prefix_of(self, text):
        """Return longest stored key that is a prefix of text."""
        node = self.root
        remaining = text
        last_match = ""
        consumed = []

        if node.is_end:
            last_match = ""

        while remaining:
            first = remaining[0]
            if first not in node.children:
                break
            child = node.children[first]
            cp = _common_prefix_len(remaining, child.prefix)
            if cp < len(child.prefix):
                break
            consumed.append(child.prefix)
            remaining = remaining[cp:]
            node = child
            if node.is_end:
                last_match = ''.join(consumed)

        return last_match

    def all_keys(self):
        """Return all keys."""
        results = []
        self._collect_radix(self.root, [], results)
        return results

    def items(self):
        """Return all (key, value) pairs."""
        results = []
        self._collect_radix_items(self.root, [], results)
        return results

    def _find(self, key):
        """Navigate to where key ends up. Returns (node, remaining_key)."""
        node = self.root
        remaining = key
        while remaining:
            first = remaining[0]
            if first not in node.children:
                return (node, remaining)
            child = node.children[first]
            cp = _common_prefix_len(remaining, child.prefix)
            if cp < len(child.prefix):
                return (node, remaining)
            remaining = remaining[cp:]
            node = child
        return (node, "")

    def _delete(self, node, key):
        """Recursive delete with merge-back."""
        if not key:
            if not node.is_end:
                return False
            node.is_end = False
            node.value = None
            return True

        first = key[0]
        if first not in node.children:
            return False

        child = node.children[first]
        cp = _common_prefix_len(key, child.prefix)
        if cp < len(child.prefix):
            return False

        rest = key[cp:]
        if not self._delete(child, rest):
            return False

        # Cleanup: remove childless non-end nodes, merge single-child non-end nodes
        if not child.is_end and not child.children:
            del node.children[first]
        elif not child.is_end and len(child.children) == 1:
            # Merge child with its only grandchild
            grandchild_key = next(iter(child.children))
            grandchild = child.children[grandchild_key]
            grandchild.prefix = child.prefix + grandchild.prefix
            node.children[first] = grandchild

        return True

    def _collect_radix(self, node, path, results, limit=None):
        if limit is not None and len(results) >= limit:
            return
        if node.is_end:
            results.append(''.join(path))
        for ch in sorted(node.children):
            if limit is not None and len(results) >= limit:
                return
            child = node.children[ch]
            path_len = len(path)
            path.extend(child.prefix)
            self._collect_radix(child, path, results, limit)
            del path[path_len:]

    def _collect_radix_items(self, node, path, results):
        if node.is_end:
            results.append((''.join(path), node.value))
        for ch in sorted(node.children):
            child = node.children[ch]
            path_len = len(path)
            path.extend(child.prefix)
            self._collect_radix_items(child, path, results)
            del path[path_len:]


# =============================================================================
# Variant 3: Persistent Trie (immutable, path-copying)
# =============================================================================

class PTrieNode:
    __slots__ = ('children', 'is_end', 'value')

    def __init__(self, children=None, is_end=False, value=None):
        self.children = children if children is not None else {}
        self.is_end = is_end
        self.value = value

    def _copy_with(self, **kwargs):
        children = kwargs.get('children', self.children)
        is_end = kwargs.get('is_end', self.is_end)
        value = kwargs.get('value', self.value)
        return PTrieNode(dict(children), is_end, value)


class PersistentTrie:
    """Immutable trie -- insert/delete return new tries, old versions preserved."""

    def __init__(self, root=None, size=0):
        self.root = root if root is not None else PTrieNode()
        self._size = size

    def __len__(self):
        return self._size

    def __contains__(self, key):
        return self.search(key)

    def insert(self, key, value=True):
        """Return a new PersistentTrie with key inserted."""
        new_root, was_new = self._insert(self.root, key, 0, value)
        new_size = self._size + (1 if was_new else 0)
        return PersistentTrie(new_root, new_size)

    def search(self, key):
        node = self._find_node(key)
        return node is not None and node.is_end

    def get(self, key, default=None):
        node = self._find_node(key)
        if node is not None and node.is_end:
            return node.value
        return default

    def delete(self, key):
        """Return a new PersistentTrie without key. Returns self if key not found."""
        if not self.search(key):
            return self
        new_root = self._delete(self.root, key, 0)
        return PersistentTrie(new_root, self._size - 1)

    def keys_with_prefix(self, prefix):
        node = self._find_node(prefix)
        if node is None:
            return []
        results = []
        self._collect(node, list(prefix), results)
        return results

    def all_keys(self):
        results = []
        self._collect(self.root, [], results)
        return results

    def _find_node(self, key):
        node = self.root
        for ch in key:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def _insert(self, node, key, idx, value):
        """Returns (new_node, was_new)."""
        new_node = node._copy_with()
        if idx == len(key):
            was_new = not new_node.is_end
            new_node.is_end = True
            new_node.value = value
            return new_node, was_new

        ch = key[idx]
        if ch in new_node.children:
            child, was_new = self._insert(new_node.children[ch], key, idx + 1, value)
        else:
            child, was_new = self._insert(PTrieNode(), key, idx + 1, value)
        new_node.children[ch] = child
        return new_node, was_new

    def _delete(self, node, key, idx):
        new_node = node._copy_with()
        if idx == len(key):
            new_node.is_end = False
            new_node.value = None
            return new_node

        ch = key[idx]
        child = self._delete(new_node.children[ch], key, idx + 1)
        if not child.is_end and not child.children:
            del new_node.children[ch]
        else:
            new_node.children[ch] = child
        return new_node

    def _collect(self, node, path, results):
        if node.is_end:
            results.append(''.join(path))
        for ch in sorted(node.children):
            path.append(ch)
            self._collect(node.children[ch], path, results)
            path.pop()


# =============================================================================
# Variant 4: Ternary Search Tree
# =============================================================================

class TSTNode:
    __slots__ = ('char', 'left', 'mid', 'right', 'is_end', 'value')

    def __init__(self, char):
        self.char = char
        self.left = None    # chars < self.char
        self.mid = None     # chars == self.char (next char)
        self.right = None   # chars > self.char
        self.is_end = False
        self.value = None


class TernarySearchTree:
    """Space-efficient trie using BST at each level.

    Better memory usage than standard trie for sparse key sets.
    Average O(log n + k) search where n = alphabet size, k = key length.
    """

    def __init__(self):
        self.root = None
        self._size = 0

    def __len__(self):
        return self._size

    def __contains__(self, key):
        return self.search(key)

    def insert(self, key, value=True):
        """Insert key with optional value."""
        if not key:
            return None
        self.root, old = self._insert(self.root, key, 0, value)
        if old is None:
            self._size += 1
        return old

    def search(self, key):
        """Return True if key exists."""
        if not key:
            return False
        node = self._find(self.root, key, 0)
        return node is not None and node.is_end

    def get(self, key, default=None):
        """Get value for key."""
        if not key:
            return default
        node = self._find(self.root, key, 0)
        if node is not None and node.is_end:
            return node.value
        return default

    def delete(self, key):
        """Delete key. Returns True if deleted."""
        if not key:
            return False
        found = self._find(self.root, key, 0)
        if found is None or not found.is_end:
            return False
        found.is_end = False
        found.value = None
        self._size -= 1
        return True

    def starts_with(self, prefix):
        """Return True if any key starts with prefix."""
        if not prefix:
            return self._size > 0
        node = self._find(self.root, prefix, 0)
        if node is None:
            return False
        if node.is_end:
            return True
        return self._has_any(node.mid)

    def keys_with_prefix(self, prefix):
        """Return all keys starting with prefix."""
        if not prefix:
            return self.all_keys()
        node = self._find(self.root, prefix, 0)
        if node is None:
            return []
        results = []
        if node.is_end:
            results.append(prefix)
        self._collect_tst(node.mid, list(prefix), results)
        return results

    def all_keys(self):
        """Return all keys."""
        results = []
        self._collect_tst(self.root, [], results)
        return results

    def longest_prefix_of(self, text):
        """Return longest stored key that is a prefix of text."""
        if not text:
            return ""
        node = self.root
        idx = 0
        last_match = ""
        while node is not None and idx < len(text):
            ch = text[idx]
            if ch < node.char:
                node = node.left
            elif ch > node.char:
                node = node.right
            else:
                idx += 1
                if node.is_end:
                    last_match = text[:idx]
                node = node.mid
        return last_match

    def near_search(self, pattern, max_dist):
        """Find all keys within Hamming distance max_dist of pattern."""
        results = []
        self._near(self.root, pattern, 0, max_dist, [], results)
        return results

    def _insert(self, node, key, idx, value):
        ch = key[idx]
        if node is None:
            node = TSTNode(ch)

        if ch < node.char:
            node.left, old = self._insert(node.left, key, idx, value)
            return node, old
        elif ch > node.char:
            node.right, old = self._insert(node.right, key, idx, value)
            return node, old
        else:
            if idx + 1 == len(key):
                old = node.value if node.is_end else None
                node.is_end = True
                node.value = value
                return node, old
            else:
                node.mid, old = self._insert(node.mid, key, idx + 1, value)
                return node, old

    def _find(self, node, key, idx):
        if node is None:
            return None
        ch = key[idx]
        if ch < node.char:
            return self._find(node.left, key, idx)
        elif ch > node.char:
            return self._find(node.right, key, idx)
        else:
            if idx + 1 == len(key):
                return node
            return self._find(node.mid, key, idx + 1)

    def _has_any(self, node):
        if node is None:
            return False
        if node.is_end:
            return True
        return self._has_any(node.left) or self._has_any(node.mid) or self._has_any(node.right)

    def _collect_tst(self, node, path, results):
        if node is None:
            return
        self._collect_tst(node.left, path, results)
        path.append(node.char)
        if node.is_end:
            results.append(''.join(path))
        self._collect_tst(node.mid, path, results)
        path.pop()
        self._collect_tst(node.right, path, results)

    def _near(self, node, pattern, idx, dist, path, results):
        """Hamming-distance search."""
        if node is None or dist < 0:
            return
        ch = pattern[idx] if idx < len(pattern) else None

        # Explore left (smaller chars)
        if ch is not None and ch < node.char or dist > 0:
            self._near(node.left, pattern, idx, dist, path, results)

        # Match/mismatch at this character
        if idx < len(pattern):
            cost = 0 if node.char == ch else 1
            path.append(node.char)
            if idx + 1 == len(pattern):
                if node.is_end and dist - cost >= 0:
                    results.append(''.join(path))
            else:
                self._near(node.mid, pattern, idx + 1, dist - cost, path, results)
            path.pop()

        # Explore right (larger chars)
        if ch is not None and ch > node.char or dist > 0:
            self._near(node.right, pattern, idx, dist, path, results)


# =============================================================================
# Utility
# =============================================================================

def _common_prefix_len(a, b):
    """Return length of common prefix between strings a and b."""
    i = 0
    limit = min(len(a), len(b))
    while i < limit and a[i] == b[i]:
        i += 1
    return i
