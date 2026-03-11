"""
C109: Trie / Patricia Tree / Ternary Search Tree

Variants:
  1. Trie -- standard trie with insert/search/delete/prefix/wildcard
  2. PatriciaTrie (Radix Tree) -- compressed trie with path compression
  3. TernarySearchTree -- space-efficient BST-trie hybrid
  4. AutocompleteTrie -- frequency-based autocomplete + fuzzy matching
  5. IPRoutingTrie -- longest prefix match for IP routing
"""

# ---------------------------------------------------------------------------
# 1. Standard Trie
# ---------------------------------------------------------------------------

class TrieNode:
    __slots__ = ('children', 'is_end', 'value', 'count')

    def __init__(self):
        self.children = {}
        self.is_end = False
        self.value = None
        self.count = 0  # number of words passing through this node


class Trie:
    """Standard trie supporting insert, search, delete, prefix queries, and wildcard matching."""

    def __init__(self):
        self.root = TrieNode()
        self._size = 0

    def __len__(self):
        return self._size

    def __contains__(self, key):
        return self.search(key)

    def insert(self, key, value=True):
        """Insert a key with optional value. Returns True if new key, False if updated."""
        node = self.root
        for ch in key:
            node.count += 1
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.count += 1
        is_new = not node.is_end
        node.is_end = True
        node.value = value
        if is_new:
            self._size += 1
        return is_new

    def search(self, key):
        """Return True if key exists in trie."""
        node = self._find_node(key)
        return node is not None and node.is_end

    def get(self, key, default=None):
        """Get value associated with key."""
        node = self._find_node(key)
        if node is not None and node.is_end:
            return node.value
        return default

    def delete(self, key):
        """Delete a key. Returns True if deleted, False if not found."""
        if not self.search(key):
            return False
        self._delete_recursive(self.root, key, 0)
        self._size -= 1
        return True

    def _delete_recursive(self, node, key, depth):
        if depth == len(key):
            node.is_end = False
            node.value = None
            node.count -= 1
            return len(node.children) == 0
        ch = key[depth]
        child = node.children[ch]
        should_delete = self._delete_recursive(child, key, depth + 1)
        node.count -= 1
        if should_delete:
            del node.children[ch]
            return not node.is_end and len(node.children) == 0
        return False

    def starts_with(self, prefix):
        """Return True if any key starts with prefix."""
        return self._find_node(prefix) is not None

    def keys_with_prefix(self, prefix):
        """Return all keys that start with prefix."""
        node = self._find_node(prefix)
        if node is None:
            return []
        result = []
        self._collect(node, list(prefix), result)
        return result

    def count_with_prefix(self, prefix):
        """Count keys starting with prefix."""
        node = self._find_node(prefix)
        if node is None:
            return 0
        return sum(1 for _ in self._iter_from(node, list(prefix)))

    def wildcard_match(self, pattern):
        """Match keys where '.' matches any single character."""
        result = []
        self._wildcard_search(self.root, pattern, 0, [], result)
        return result

    def longest_prefix_of(self, text):
        """Find the longest key that is a prefix of text."""
        node = self.root
        longest = ""
        current = []
        for ch in text:
            if ch not in node.children:
                break
            node = node.children[ch]
            current.append(ch)
            if node.is_end:
                longest = "".join(current)
        return longest

    def all_keys(self):
        """Return all keys in sorted order."""
        result = []
        self._collect(self.root, [], result)
        return result

    def _find_node(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def _collect(self, node, path, result):
        if node.is_end:
            result.append("".join(path))
        for ch in sorted(node.children):
            path.append(ch)
            self._collect(node.children[ch], path, result)
            path.pop()

    def _iter_from(self, node, path):
        if node.is_end:
            yield "".join(path)
        for ch in sorted(node.children):
            path.append(ch)
            yield from self._iter_from(node.children[ch], path)
            path.pop()

    def _wildcard_search(self, node, pattern, idx, path, result):
        if idx == len(pattern):
            if node.is_end:
                result.append("".join(path))
            return
        ch = pattern[idx]
        if ch == '.':
            for c in sorted(node.children):
                path.append(c)
                self._wildcard_search(node.children[c], pattern, idx + 1, path, result)
                path.pop()
        else:
            if ch in node.children:
                path.append(ch)
                self._wildcard_search(node.children[ch], pattern, idx + 1, path, result)
                path.pop()


# ---------------------------------------------------------------------------
# 2. Patricia Trie (Radix Tree) -- compressed trie
# ---------------------------------------------------------------------------

class RadixNode:
    __slots__ = ('label', 'children', 'is_end', 'value')

    def __init__(self, label=""):
        self.label = label
        self.children = {}
        self.is_end = False
        self.value = None


class PatriciaTrie:
    """Radix tree / Patricia trie with path compression."""

    def __init__(self):
        self.root = RadixNode()
        self._size = 0

    def __len__(self):
        return self._size

    def __contains__(self, key):
        return self.search(key)

    def insert(self, key, value=True):
        """Insert key with optional value. Returns True if new."""
        node = self.root
        remaining = key

        while remaining:
            # Find child with matching first character
            first_ch = remaining[0]
            if first_ch not in node.children:
                # No matching child -- create new leaf
                new_node = RadixNode(remaining)
                new_node.is_end = True
                new_node.value = value
                node.children[first_ch] = new_node
                self._size += 1
                return True

            child = node.children[first_ch]
            label = child.label
            # Find common prefix length
            common = 0
            while common < len(label) and common < len(remaining) and label[common] == remaining[common]:
                common += 1

            if common == len(label):
                # Label fully matched -- continue down
                remaining = remaining[common:]
                if not remaining:
                    # Exact match at this node
                    is_new = not child.is_end
                    child.is_end = True
                    child.value = value
                    if is_new:
                        self._size += 1
                    return is_new
                node = child
            else:
                # Split needed
                # Create split node with common prefix
                split = RadixNode(label[:common])
                node.children[first_ch] = split

                # Old child becomes child of split with remaining label
                child.label = label[common:]
                split.children[child.label[0]] = child

                rest = remaining[common:]
                if not rest:
                    # Key ends at split point
                    split.is_end = True
                    split.value = value
                else:
                    # Key continues past split
                    new_leaf = RadixNode(rest)
                    new_leaf.is_end = True
                    new_leaf.value = value
                    split.children[rest[0]] = new_leaf

                self._size += 1
                return True

        # Empty remaining -- key matched root path exactly
        # This only happens for empty string key
        is_new = not node.is_end
        node.is_end = True
        node.value = value
        if is_new:
            self._size += 1
        return is_new

    def search(self, key):
        """Return True if key exists."""
        node, remaining = self._traverse(key)
        return node is not None and remaining == "" and node.is_end

    def get(self, key, default=None):
        """Get value for key."""
        node, remaining = self._traverse(key)
        if node is not None and remaining == "" and node.is_end:
            return node.value
        return default

    def delete(self, key):
        """Delete key. Returns True if deleted."""
        if not self.search(key):
            return False
        self._delete(self.root, key)
        self._size -= 1
        return True

    def _delete(self, node, remaining):
        if not remaining:
            node.is_end = False
            node.value = None
            return

        first_ch = remaining[0]
        child = node.children[first_ch]
        label = child.label

        if remaining[:len(label)] == label:
            self._delete(child, remaining[len(label):])

            # Cleanup: if child is no longer end and has no children, remove it
            if not child.is_end and not child.children:
                del node.children[first_ch]
            # If child has exactly one child, merge them
            elif not child.is_end and len(child.children) == 1:
                only_key = next(iter(child.children))
                grandchild = child.children[only_key]
                grandchild.label = child.label + grandchild.label
                node.children[first_ch] = grandchild

    def starts_with(self, prefix):
        """Return True if any key starts with prefix."""
        node, remaining = self._traverse(prefix)
        if node is not None and remaining == "":
            return True
        # Check if remaining is a prefix of some child's label
        if node is not None and remaining:
            first_ch = remaining[0]
            if first_ch in node.children:
                child = node.children[first_ch]
                if child.label.startswith(remaining):
                    return True
        return False

    def keys_with_prefix(self, prefix):
        """Return all keys starting with prefix."""
        result = []
        node, remaining = self._traverse(prefix)

        if node is not None and remaining == "":
            self._collect_radix(node, list(prefix), result)
        elif node is not None and remaining:
            first_ch = remaining[0]
            if first_ch in node.children:
                child = node.children[first_ch]
                if child.label.startswith(remaining):
                    self._collect_radix(child, list(prefix + child.label[len(remaining):]), result)
        return result

    def all_keys(self):
        """Return all keys sorted."""
        result = []
        self._collect_radix(self.root, [], result)
        return result

    def longest_prefix_of(self, text):
        """Find longest key that is a prefix of text."""
        node = self.root
        remaining = text
        longest = ""
        matched = []

        if node.is_end:
            longest = ""

        while remaining:
            first_ch = remaining[0]
            if first_ch not in node.children:
                break
            child = node.children[first_ch]
            label = child.label

            # Check if label matches
            if len(remaining) < len(label):
                if remaining == label[:len(remaining)]:
                    # Partial match -- can't go further
                    break
                break
            if remaining[:len(label)] != label:
                break

            matched.extend(label)
            remaining = remaining[len(label):]
            node = child
            if node.is_end:
                longest = "".join(matched)

        return longest

    def _traverse(self, key):
        """Walk the trie following key. Returns (node, remaining_key)."""
        node = self.root
        remaining = key

        while remaining:
            first_ch = remaining[0]
            if first_ch not in node.children:
                return (node, remaining)
            child = node.children[first_ch]
            label = child.label

            if remaining[:len(label)] == label:
                remaining = remaining[len(label):]
                node = child
            elif label.startswith(remaining):
                # remaining is prefix of label
                return (node, remaining)
            else:
                return (None, remaining)

        return (node, "")

    def _collect_radix(self, node, path, result):
        if node.is_end:
            result.append("".join(path))
        for ch in sorted(node.children):
            child = node.children[ch]
            path.extend(child.label)
            self._collect_radix(child, path, result)
            for _ in range(len(child.label)):
                path.pop()


# ---------------------------------------------------------------------------
# 3. Ternary Search Tree
# ---------------------------------------------------------------------------

class TSTNode:
    __slots__ = ('char', 'left', 'mid', 'right', 'is_end', 'value')

    def __init__(self, char):
        self.char = char
        self.left = None
        self.mid = None
        self.right = None
        self.is_end = False
        self.value = None


class TernarySearchTree:
    """Ternary search tree -- space-efficient trie alternative."""

    def __init__(self):
        self.root = None
        self._size = 0

    def __len__(self):
        return self._size

    def __contains__(self, key):
        return self.search(key)

    def insert(self, key, value=True):
        """Insert key with optional value. Returns True if new."""
        if not key:
            return False
        is_new = [True]
        self.root = self._insert(self.root, key, 0, value, is_new)
        if is_new[0]:
            self._size += 1
        return is_new[0]

    def _insert(self, node, key, idx, value, is_new):
        ch = key[idx]
        if node is None:
            node = TSTNode(ch)

        if ch < node.char:
            node.left = self._insert(node.left, key, idx, value, is_new)
        elif ch > node.char:
            node.right = self._insert(node.right, key, idx, value, is_new)
        else:
            if idx + 1 < len(key):
                node.mid = self._insert(node.mid, key, idx + 1, value, is_new)
            else:
                if node.is_end:
                    is_new[0] = False
                node.is_end = True
                node.value = value
        return node

    def search(self, key):
        """Return True if key exists."""
        if not key:
            return False
        node = self._search(self.root, key, 0)
        return node is not None and node.is_end

    def get(self, key, default=None):
        """Get value for key."""
        if not key:
            return default
        node = self._search(self.root, key, 0)
        if node is not None and node.is_end:
            return node.value
        return default

    def _search(self, node, key, idx):
        if node is None:
            return None
        ch = key[idx]
        if ch < node.char:
            return self._search(node.left, key, idx)
        elif ch > node.char:
            return self._search(node.right, key, idx)
        else:
            if idx + 1 == len(key):
                return node
            return self._search(node.mid, key, idx + 1)

    def delete(self, key):
        """Delete key. Returns True if deleted."""
        if not key or not self.search(key):
            return False
        self.root = self._delete(self.root, key, 0)
        self._size -= 1
        return True

    def _delete(self, node, key, idx):
        if node is None:
            return None
        ch = key[idx]
        if ch < node.char:
            node.left = self._delete(node.left, key, idx)
        elif ch > node.char:
            node.right = self._delete(node.right, key, idx)
        else:
            if idx + 1 == len(key):
                node.is_end = False
                node.value = None
            else:
                node.mid = self._delete(node.mid, key, idx + 1)

        # Clean up node if it's not needed
        if not node.is_end and node.left is None and node.mid is None and node.right is None:
            return None
        return node

    def keys_with_prefix(self, prefix):
        """Return all keys starting with prefix."""
        if not prefix:
            return self.all_keys()
        node = self._search(self.root, prefix, 0)
        if node is None:
            return []
        result = []
        if node.is_end:
            result.append(prefix)
        self._collect_tst(node.mid, list(prefix), result)
        return result

    def starts_with(self, prefix):
        """Return True if any key starts with prefix."""
        return len(self.keys_with_prefix(prefix)) > 0

    def all_keys(self):
        """Return all keys sorted."""
        result = []
        self._collect_tst(self.root, [], result)
        return result

    def near_search(self, key, max_distance):
        """Find all keys within edit distance max_distance of key."""
        result = set()
        self._near_search(self.root, key, 0, max_distance, [], result, max_distance)
        return sorted(result)

    def _near_search(self, node, key, idx, dist, path, result, max_dist):
        if node is None or dist < 0:
            return

        ch = key[idx] if idx < len(key) else None

        # Explore left subtree (if current char might be less)
        if ch is None or ch < node.char or dist > 0:
            self._near_search(node.left, key, idx, dist, path, result, max_dist)

        # Middle subtree
        path.append(node.char)
        match = ch == node.char
        new_dist = dist if match else dist - 1

        if node.is_end:
            word = "".join(path)
            if self._edit_distance(word, key) <= max_dist:
                result.add(word)

        if idx < len(key):
            self._near_search(node.mid, key, idx + 1, new_dist, path, result, max_dist)
            # Insertion: advance in tree but not in key
            if dist > 0:
                self._near_search(node.mid, key, idx, dist - 1, path, result, max_dist)
        else:
            # Past end of key -- deletion case
            if dist > 0:
                self._near_search(node.mid, key, idx, dist - 1, path, result, max_dist)

        path.pop()

        # Explore right subtree
        if ch is None or ch > node.char or dist > 0:
            self._near_search(node.right, key, idx, dist, path, result, max_dist)

    def _edit_distance(self, a, b):
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if a[i-1] == b[j-1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return dp[n]

    def _collect_tst(self, node, path, result):
        if node is None:
            return
        self._collect_tst(node.left, path, result)
        path.append(node.char)
        if node.is_end:
            result.append("".join(path))
        self._collect_tst(node.mid, path, result)
        path.pop()
        self._collect_tst(node.right, path, result)


# ---------------------------------------------------------------------------
# 4. AutocompleteTrie -- frequency-based autocomplete + fuzzy matching
# ---------------------------------------------------------------------------

class AutocompleteNode:
    __slots__ = ('children', 'is_end', 'frequency', 'max_child_freq')

    def __init__(self):
        self.children = {}
        self.is_end = False
        self.frequency = 0
        self.max_child_freq = 0  # max frequency in subtree


class AutocompleteTrie:
    """Trie with frequency-based autocomplete and fuzzy matching."""

    def __init__(self):
        self.root = AutocompleteNode()
        self._size = 0

    def __len__(self):
        return self._size

    def insert(self, word, frequency=1):
        """Insert word with frequency. Frequencies accumulate."""
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = AutocompleteNode()
            node = node.children[ch]
        is_new = not node.is_end
        node.is_end = True
        node.frequency += frequency
        if is_new:
            self._size += 1
        # Update max_child_freq up the path
        self._update_max_freq(word)
        return is_new

    def _update_max_freq(self, word):
        node = self.root
        for ch in word:
            child = node.children[ch]
            node.max_child_freq = max(node.max_child_freq, self._subtree_max(child))
            node = child

    def _subtree_max(self, node):
        best = node.frequency if node.is_end else 0
        best = max(best, node.max_child_freq)
        return best

    def autocomplete(self, prefix, k=10):
        """Return top-k completions by frequency."""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]

        # Collect all words under this prefix with frequencies
        results = []
        self._collect_with_freq(node, list(prefix), results)
        # Sort by frequency descending, then alphabetically
        results.sort(key=lambda x: (-x[1], x[0]))
        return [(w, f) for w, f in results[:k]]

    def fuzzy_autocomplete(self, prefix, k=10, max_edits=1):
        """Autocomplete allowing up to max_edits typos in prefix."""
        candidates = {}
        self._fuzzy_prefix_search(self.root, prefix, 0, max_edits, [], candidates)

        # For each matching prefix node, collect completions
        results = []
        for word, freq in candidates.items():
            results.append((word, freq))
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:k]

    def _fuzzy_prefix_search(self, node, prefix, idx, edits_left, path, candidates):
        """Find all nodes reachable from prefix with up to edits_left edits."""
        if edits_left < 0:
            return

        if idx == len(prefix):
            # Reached end of prefix -- collect all completions from here
            self._collect_words(node, path, candidates)
            # Also allow insertions at end (extra chars in trie not in prefix)
            if edits_left > 0:
                for ch in node.children:
                    path.append(ch)
                    self._fuzzy_prefix_search(node.children[ch], prefix, idx, edits_left - 1, path, candidates)
                    path.pop()
            return

        target_ch = prefix[idx]

        for ch in node.children:
            child = node.children[ch]
            path.append(ch)
            if ch == target_ch:
                # Match -- advance both
                self._fuzzy_prefix_search(child, prefix, idx + 1, edits_left, path, candidates)
            else:
                # Substitution -- advance both
                self._fuzzy_prefix_search(child, prefix, idx + 1, edits_left - 1, path, candidates)
            # Insertion -- advance trie only (extra char in trie)
            if edits_left > 0:
                self._fuzzy_prefix_search(child, prefix, idx, edits_left - 1, path, candidates)
            path.pop()

        # Deletion -- skip character in prefix, stay at same trie node
        self._fuzzy_prefix_search(node, prefix, idx + 1, edits_left - 1, path, candidates)

    def _collect_words(self, node, path, candidates):
        if node.is_end:
            word = "".join(path)
            if word not in candidates or candidates[word] < node.frequency:
                candidates[word] = node.frequency
        for ch in sorted(node.children):
            path.append(ch)
            self._collect_words(node.children[ch], path, candidates)
            path.pop()

    def _collect_with_freq(self, node, path, results):
        if node.is_end:
            results.append(("".join(path), node.frequency))
        for ch in sorted(node.children):
            path.append(ch)
            self._collect_with_freq(node.children[ch], path, results)
            path.pop()

    def get_frequency(self, word):
        """Get frequency of a word."""
        node = self.root
        for ch in word:
            if ch not in node.children:
                return 0
            node = node.children[ch]
        return node.frequency if node.is_end else 0


# ---------------------------------------------------------------------------
# 5. IP Routing Trie -- longest prefix match on binary keys
# ---------------------------------------------------------------------------

class IPTrieNode:
    __slots__ = ('children', 'value', 'prefix_len')

    def __init__(self):
        self.children = [None, None]  # 0 and 1
        self.value = None  # route/next-hop info
        self.prefix_len = None


class IPRoutingTrie:
    """Binary trie for IP routing with longest prefix match.

    Stores CIDR prefixes (e.g., "192.168.1.0/24") and finds the
    most specific matching route for a given IP address.
    """

    def __init__(self):
        self.root = IPTrieNode()
        self._size = 0

    def __len__(self):
        return self._size

    def insert(self, cidr, value):
        """Insert a CIDR prefix (e.g., '192.168.1.0/24') with a route value."""
        bits, prefix_len = self._cidr_to_bits(cidr)
        node = self.root
        for i in range(prefix_len):
            bit = bits[i]
            if node.children[bit] is None:
                node.children[bit] = IPTrieNode()
            node = node.children[bit]
        is_new = node.value is None
        node.value = value
        node.prefix_len = prefix_len
        if is_new:
            self._size += 1
        return is_new

    def lookup(self, ip):
        """Longest prefix match for an IP address. Returns (value, prefix_len) or None."""
        bits = self._ip_to_bits(ip)
        node = self.root
        best_match = None

        if node.value is not None:
            best_match = (node.value, 0)

        for i in range(32):
            bit = bits[i]
            if node.children[bit] is None:
                break
            node = node.children[bit]
            if node.value is not None:
                best_match = (node.value, node.prefix_len)

        return best_match

    def delete(self, cidr):
        """Delete a CIDR route. Returns True if deleted."""
        bits, prefix_len = self._cidr_to_bits(cidr)
        node = self.root
        path = [(self.root, -1)]  # (node, bit_taken)

        for i in range(prefix_len):
            bit = bits[i]
            if node.children[bit] is None:
                return False
            node = node.children[bit]
            path.append((node, bit))

        if node.value is None:
            return False

        node.value = None
        node.prefix_len = None
        self._size -= 1

        # Cleanup empty nodes bottom-up
        for i in range(len(path) - 1, 0, -1):
            n, bit = path[i]
            if n.value is None and n.children[0] is None and n.children[1] is None:
                parent = path[i-1][0]
                parent.children[bit] = None
            else:
                break

        return True

    def all_routes(self):
        """Return all (cidr, value) pairs."""
        results = []
        self._collect_routes(self.root, [], results)
        return results

    def _collect_routes(self, node, bits, results):
        if node.value is not None:
            cidr = self._bits_to_cidr(bits, node.prefix_len)
            results.append((cidr, node.value))
        for bit in [0, 1]:
            if node.children[bit] is not None:
                bits.append(bit)
                self._collect_routes(node.children[bit], bits, results)
                bits.pop()

    def _cidr_to_bits(self, cidr):
        if '/' in cidr:
            ip_str, prefix_len = cidr.split('/')
            prefix_len = int(prefix_len)
        else:
            ip_str = cidr
            prefix_len = 32
        bits = self._ip_to_bits(ip_str)
        return bits, prefix_len

    def _ip_to_bits(self, ip):
        parts = ip.split('.')
        num = 0
        for p in parts:
            num = (num << 8) | int(p)
        bits = []
        for i in range(31, -1, -1):
            bits.append((num >> i) & 1)
        return bits

    def _bits_to_cidr(self, bits, prefix_len):
        # Pad to 32 bits
        full_bits = bits[:prefix_len] + [0] * (32 - prefix_len)
        num = 0
        for b in full_bits:
            num = (num << 1) | b
        octets = []
        for i in range(4):
            octets.append(str((num >> (24 - 8*i)) & 0xFF))
        return ".".join(octets) + "/" + str(prefix_len)
