"""
C119: Trie / Patricia Trie / Radix Tree

Multiple trie variants for string/sequence operations:
1. Trie -- standard prefix tree with per-node children map
2. PatriciaTrie -- compressed/radix trie (edges are substrings)
3. TernarySearchTree -- balanced trie using BST at each level
4. GeneralizedSuffixTrie -- trie of all suffixes (for substring queries)
5. AutocompleteTrie -- trie with ranking and top-k prefix suggestions
6. TrieMap -- generic trie-based ordered map with prefix operations
"""


class TrieNode:
    """Node for standard Trie."""
    __slots__ = ('children', 'is_end', 'value', 'count')

    def __init__(self):
        self.children = {}
        self.is_end = False
        self.value = None
        self.count = 0  # number of words passing through this node


class Trie:
    """Standard prefix tree. O(m) insert/search/delete where m = key length."""

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
        node.count += 1
        old = node.value if node.is_end else None
        if not node.is_end:
            self._size += 1
        node.is_end = True
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
        """Delete key. Returns True if deleted, False if not found."""
        # Walk down, recording path
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
        node.count -= 1
        for parent, ch in reversed(path):
            parent.count -= 1
            child = parent.children[ch]
            if child.count == 0:
                del parent.children[ch]
        return True

    def starts_with(self, prefix):
        """Return True if any key starts with prefix."""
        return self._find_node(prefix) is not None

    def keys_with_prefix(self, prefix):
        """Return all keys with given prefix."""
        node = self._find_node(prefix)
        if node is None:
            return []
        result = []
        self._collect(node, list(prefix), result)
        return result

    def count_with_prefix(self, prefix):
        """Return count of keys with given prefix."""
        node = self._find_node(prefix)
        if node is None:
            return 0
        return self._count_ends(node)

    def longest_prefix_of(self, text):
        """Return longest key that is a prefix of text."""
        node = self.root
        longest = ""
        current = []
        for ch in text:
            if ch not in node.children:
                break
            current.append(ch)
            node = node.children[ch]
            if node.is_end:
                longest = "".join(current)
        return longest

    def all_keys(self):
        """Return all keys in sorted order."""
        result = []
        self._collect(self.root, [], result)
        return result

    def _find_node(self, key):
        """Find node for key prefix. Returns None if not found."""
        node = self.root
        for ch in key:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def _collect(self, node, prefix, result):
        """Collect all keys under node."""
        if node.is_end:
            result.append("".join(prefix))
        for ch in sorted(node.children):
            prefix.append(ch)
            self._collect(node.children[ch], prefix, result)
            prefix.pop()

    def _count_ends(self, node):
        """Count end nodes under node."""
        count = 1 if node.is_end else 0
        for child in node.children.values():
            count += self._count_ends(child)
        return count


# --- Patricia Trie (Compressed / Radix Tree) ---

class PatriciaNode:
    """Node for Patricia/Radix Trie. Edges carry substrings."""
    __slots__ = ('children', 'is_end', 'value')

    def __init__(self):
        self.children = {}  # first_char -> (edge_label, child_node)
        self.is_end = False
        self.value = None


class PatriciaTrie:
    """
    Compressed trie / radix tree. Edges are substrings, not single chars.
    Space-efficient for sparse key sets. O(m) operations.
    """

    def __init__(self):
        self.root = PatriciaNode()
        self._size = 0

    def __len__(self):
        return self._size

    def __contains__(self, key):
        return self.search(key)

    def insert(self, key, value=True):
        """Insert key with optional value."""
        if not key:
            old = self.root.value if self.root.is_end else None
            if not self.root.is_end:
                self._size += 1
            self.root.is_end = True
            self.root.value = value
            return old
        return self._insert(self.root, key, value)

    def _insert(self, node, key, value):
        first = key[0]
        if first not in node.children:
            # No edge with this first char -- create new leaf
            child = PatriciaNode()
            child.is_end = True
            child.value = value
            node.children[first] = (key, child)
            self._size += 1
            return None

        edge, child = node.children[first]
        # Find common prefix length
        cp = 0
        while cp < len(edge) and cp < len(key) and edge[cp] == key[cp]:
            cp += 1

        if cp == len(edge) and cp == len(key):
            # Exact match
            old = child.value if child.is_end else None
            if not child.is_end:
                self._size += 1
            child.is_end = True
            child.value = value
            return old
        elif cp == len(edge):
            # Key extends beyond edge -- recurse into child
            return self._insert(child, key[cp:], value)
        else:
            # Split edge at cp
            mid = PatriciaNode()
            mid.children[edge[cp]] = (edge[cp:], child)
            node.children[first] = (edge[:cp], mid)
            if cp == len(key):
                # Key ends at split point
                mid.is_end = True
                mid.value = value
                self._size += 1
                return None
            else:
                # Key continues past split
                new_child = PatriciaNode()
                new_child.is_end = True
                new_child.value = value
                mid.children[key[cp]] = (key[cp:], new_child)
                self._size += 1
                return None

    def search(self, key):
        """Return True if key exists."""
        node = self._find(key)
        return node is not None and node.is_end

    def get(self, key, default=None):
        """Get value for key."""
        node = self._find(key)
        if node is not None and node.is_end:
            return node.value
        return default

    def delete(self, key):
        """Delete key. Returns True if deleted."""
        if not key:
            if not self.root.is_end:
                return False
            self.root.is_end = False
            self.root.value = None
            self._size -= 1
            return True
        deleted = self._delete(self.root, key)
        if deleted:
            self._size -= 1
        return deleted

    def _delete(self, node, key):
        first = key[0]
        if first not in node.children:
            return False
        edge, child = node.children[first]
        cp = 0
        while cp < len(edge) and cp < len(key) and edge[cp] == key[cp]:
            cp += 1
        if cp < len(edge):
            return False
        if cp == len(key):
            if not child.is_end:
                return False
            child.is_end = False
            child.value = None
            # Merge child if it has exactly one child
            if len(child.children) == 1:
                ch2, (edge2, grandchild) = next(iter(child.children.items()))
                node.children[first] = (edge + edge2, grandchild)
            elif len(child.children) == 0:
                del node.children[first]
            return True
        # Recurse
        deleted = self._delete(child, key[cp:])
        if deleted:
            # Merge child if it has one child and is not an end
            if not child.is_end and len(child.children) == 1:
                ch2, (edge2, grandchild) = next(iter(child.children.items()))
                node.children[first] = (edge + edge2, grandchild)
        return deleted

    def starts_with(self, prefix):
        """Return True if any key starts with prefix."""
        node, remaining = self._find_prefix(prefix)
        return node is not None

    def keys_with_prefix(self, prefix):
        """Return all keys with given prefix."""
        node, remaining = self._find_prefix(prefix)
        if node is None:
            return []
        result = []
        self._collect(node, list(prefix), result)
        return result

    def all_keys(self):
        """Return all keys in sorted order."""
        result = []
        self._collect(self.root, [], result)
        return result

    def longest_prefix_of(self, text):
        """Return longest key that is a prefix of text."""
        node = self.root
        longest = ""
        matched = 0
        if node.is_end:
            longest = ""
        while matched < len(text):
            first = text[matched]
            if first not in node.children:
                break
            edge, child = node.children[first]
            # Check edge matches
            i = 0
            while i < len(edge) and matched + i < len(text) and edge[i] == text[matched + i]:
                i += 1
            if i < len(edge):
                # Partial edge match -- no further progress
                break
            matched += len(edge)
            node = child
            if node.is_end:
                longest = text[:matched]
        return longest

    def _find(self, key):
        """Find node for exact key."""
        if not key:
            return self.root
        node = self.root
        pos = 0
        while pos < len(key):
            first = key[pos]
            if first not in node.children:
                return None
            edge, child = node.children[first]
            if len(key) - pos < len(edge):
                return None
            for i in range(len(edge)):
                if key[pos + i] != edge[i]:
                    return None
            pos += len(edge)
            node = child
        return node

    def _find_prefix(self, prefix):
        """Find node where prefix ends. Returns (node, remaining_prefix)."""
        if not prefix:
            return self.root, ""
        node = self.root
        pos = 0
        while pos < len(prefix):
            first = prefix[pos]
            if first not in node.children:
                return None, ""
            edge, child = node.children[first]
            cp = 0
            while cp < len(edge) and pos + cp < len(prefix) and edge[cp] == prefix[pos + cp]:
                cp += 1
            if pos + cp == len(prefix):
                # Prefix ends within or at end of edge
                if cp == len(edge):
                    return child, ""
                else:
                    # Prefix ends within edge -- child subtree matches
                    return child, edge[cp:]
            if cp < len(edge):
                return None, ""
            pos += len(edge)
            node = child
        return node, ""

    def _collect(self, node, prefix, result):
        """Collect all keys under node."""
        if node.is_end:
            result.append("".join(prefix))
        for ch in sorted(node.children):
            edge, child = node.children[ch]
            prefix.extend(edge)
            self._collect(child, prefix, result)
            for _ in range(len(edge)):
                prefix.pop()


# --- Ternary Search Tree ---

class TSTNode:
    """Node for Ternary Search Tree."""
    __slots__ = ('char', 'left', 'mid', 'right', 'is_end', 'value')

    def __init__(self, char):
        self.char = char
        self.left = None
        self.mid = None
        self.right = None
        self.is_end = False
        self.value = None


class TernarySearchTree:
    """
    Ternary Search Tree -- combines trie-like prefix properties with
    BST-like space efficiency. O(m + log n) operations.
    """

    def __init__(self):
        self.root_node = None
        self._size = 0

    def __len__(self):
        return self._size

    def __contains__(self, key):
        return self.search(key)

    def insert(self, key, value=True):
        """Insert key with value."""
        if not key:
            return None
        self.root_node = self._insert(self.root_node, key, 0, value)

    def _insert(self, node, key, d, value):
        ch = key[d]
        if node is None:
            node = TSTNode(ch)
        if ch < node.char:
            node.left = self._insert(node.left, key, d, value)
        elif ch > node.char:
            node.right = self._insert(node.right, key, d, value)
        elif d < len(key) - 1:
            node.mid = self._insert(node.mid, key, d + 1, value)
        else:
            if not node.is_end:
                self._size += 1
            node.is_end = True
            node.value = value
        return node

    def search(self, key):
        """Return True if key exists."""
        if not key:
            return False
        node = self._get_node(self.root_node, key, 0)
        return node is not None and node.is_end

    def get(self, key, default=None):
        """Get value for key."""
        if not key:
            return default
        node = self._get_node(self.root_node, key, 0)
        if node is not None and node.is_end:
            return node.value
        return default

    def delete(self, key):
        """Delete key. Returns True if deleted."""
        if not key:
            return False
        node = self._get_node(self.root_node, key, 0)
        if node is None or not node.is_end:
            return False
        node.is_end = False
        node.value = None
        self._size -= 1
        return True

    def starts_with(self, prefix):
        """Return True if any key starts with prefix."""
        if not prefix:
            return self._size > 0
        node = self._get_node(self.root_node, prefix, 0)
        if node is None:
            return False
        if node.is_end:
            return True
        return node.mid is not None

    def keys_with_prefix(self, prefix):
        """Return all keys with given prefix."""
        result = []
        if not prefix:
            self._collect_all(self.root_node, [], result)
            return result
        node = self._get_node(self.root_node, prefix, 0)
        if node is None:
            return []
        if node.is_end:
            result.append(prefix)
        self._collect_all(node.mid, list(prefix), result)
        return result

    def all_keys(self):
        """Return all keys in sorted order."""
        result = []
        self._collect_all(self.root_node, [], result)
        return result

    def longest_prefix_of(self, text):
        """Return longest key that is a prefix of text."""
        if not text:
            return ""
        longest = ""
        node = self.root_node
        d = 0
        while node is not None and d < len(text):
            ch = text[d]
            if ch < node.char:
                node = node.left
            elif ch > node.char:
                node = node.right
            else:
                if node.is_end:
                    longest = text[:d + 1]
                d += 1
                node = node.mid
        return longest

    def _get_node(self, node, key, d):
        if node is None:
            return None
        ch = key[d]
        if ch < node.char:
            return self._get_node(node.left, key, d)
        elif ch > node.char:
            return self._get_node(node.right, key, d)
        elif d < len(key) - 1:
            return self._get_node(node.mid, key, d + 1)
        else:
            return node

    def _collect_all(self, node, prefix, result):
        if node is None:
            return
        self._collect_all(node.left, prefix, result)
        prefix.append(node.char)
        if node.is_end:
            result.append("".join(prefix))
        self._collect_all(node.mid, prefix, result)
        prefix.pop()
        self._collect_all(node.right, prefix, result)


# --- Autocomplete Trie ---

class AutocompleteTrie:
    """
    Trie with frequency tracking and top-k prefix suggestions.
    Supports weighted autocomplete with frequency-based ranking.
    """

    def __init__(self):
        self._trie = Trie()
        self._freq = {}  # key -> frequency

    def __len__(self):
        return len(self._trie)

    def record(self, key, count=1):
        """Record a key occurrence (or add count to frequency)."""
        self._freq[key] = self._freq.get(key, 0) + count
        self._trie.insert(key, self._freq[key])

    def search(self, key):
        """Return True if key was recorded."""
        return self._trie.search(key)

    def frequency(self, key):
        """Return frequency of key."""
        return self._freq.get(key, 0)

    def suggest(self, prefix, k=5):
        """Return top-k suggestions for prefix, sorted by frequency desc."""
        candidates = self._trie.keys_with_prefix(prefix)
        # Sort by frequency descending, then alphabetically
        candidates.sort(key=lambda x: (-self._freq.get(x, 0), x))
        return candidates[:k]

    def delete(self, key):
        """Remove a key entirely."""
        if self._trie.delete(key):
            self._freq.pop(key, None)
            return True
        return False

    def all_keys(self):
        """Return all keys sorted by frequency descending."""
        keys = self._trie.all_keys()
        keys.sort(key=lambda x: (-self._freq.get(x, 0), x))
        return keys


# --- Generalized Suffix Trie ---

class GeneralizedSuffixTrie:
    """
    Trie of all suffixes of one or more strings.
    Supports substring search, longest common substring, and suffix enumeration.
    Uses a sentinel to separate strings.
    """

    def __init__(self):
        self._trie = PatriciaTrie()
        self._strings = []
        self._suffix_owners = {}  # suffix -> set of string indices

    def add_string(self, s):
        """Add a string and index all its suffixes."""
        idx = len(self._strings)
        self._strings.append(s)
        for i in range(len(s)):
            suffix = s[i:]
            self._trie.insert(suffix, True)
            if suffix not in self._suffix_owners:
                self._suffix_owners[suffix] = set()
            self._suffix_owners[suffix].add(idx)
        return idx

    def contains_substring(self, pattern):
        """Return True if pattern is a substring of any added string."""
        return self._trie.starts_with(pattern)

    def find_occurrences(self, pattern):
        """Find all (string_idx, position) pairs where pattern occurs."""
        results = []
        for idx, s in enumerate(self._strings):
            start = 0
            while True:
                pos = s.find(pattern, start)
                if pos == -1:
                    break
                results.append((idx, pos))
                start = pos + 1
        return results

    def longest_common_substring(self):
        """Find longest substring common to ALL added strings."""
        if len(self._strings) <= 1:
            return self._strings[0] if self._strings else ""
        n = len(self._strings)
        best = ""
        # Check all substrings of shortest string
        shortest = min(self._strings, key=len)
        for length in range(len(shortest), 0, -1):
            for start in range(len(shortest) - length + 1):
                candidate = shortest[start:start + length]
                if all(candidate in s for s in self._strings):
                    return candidate
        return best

    def count_strings_with(self, pattern):
        """How many added strings contain pattern as substring."""
        return sum(1 for s in self._strings if pattern in s)


# --- TrieMap ---

class TrieMap:
    """
    Ordered map backed by a Trie. Supports prefix queries, range operations,
    and iteration in sorted key order.
    """

    def __init__(self):
        self._trie = Trie()

    def __len__(self):
        return len(self._trie)

    def __contains__(self, key):
        return self._trie.search(key)

    def __setitem__(self, key, value):
        self._trie.insert(key, value)

    def __getitem__(self, key):
        node = self._trie._find_node(key)
        if node is None or not node.is_end:
            raise KeyError(key)
        return node.value

    def __delitem__(self, key):
        if not self._trie.delete(key):
            raise KeyError(key)

    def get(self, key, default=None):
        return self._trie.get(key, default)

    def keys(self):
        """All keys in sorted order."""
        return self._trie.all_keys()

    def values(self):
        """All values in key-sorted order."""
        result = []
        self._collect_values(self._trie.root, result)
        return result

    def items(self):
        """All (key, value) pairs in sorted order."""
        result = []
        self._collect_items(self._trie.root, [], result)
        return result

    def prefix_keys(self, prefix):
        """Keys with given prefix."""
        return self._trie.keys_with_prefix(prefix)

    def prefix_items(self, prefix):
        """(key, value) pairs with given prefix."""
        node = self._trie._find_node(prefix)
        if node is None:
            return []
        result = []
        self._collect_items(node, list(prefix), result)
        return result

    def longest_prefix_of(self, text):
        return self._trie.longest_prefix_of(text)

    def _collect_values(self, node, result):
        if node.is_end:
            result.append(node.value)
        for ch in sorted(node.children):
            self._collect_values(node.children[ch], result)

    def _collect_items(self, node, prefix, result):
        if node.is_end:
            result.append(("".join(prefix), node.value))
        for ch in sorted(node.children):
            prefix.append(ch)
            self._collect_items(node.children[ch], prefix, result)
            prefix.pop()
