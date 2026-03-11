"""
C101: Suffix Automaton (DAWG -- Directed Acyclic Word Graph)

A suffix automaton for a string s is the minimal DFA that accepts exactly
all suffixes of s. It has at most 2n-1 states and 3n-4 transitions for
a string of length n, and can be built in O(n) time online.

Applications:
- Count distinct substrings
- Find longest common substring of two strings
- Check if a pattern is a substring
- Find shortest non-occurring string
- Count occurrences of each substring
- Find the k-th lexicographically smallest substring
- Compute the longest repeated substring
"""

from collections import defaultdict


class State:
    """A state in the suffix automaton."""
    __slots__ = ('len', 'link', 'transitions', 'is_clone', 'end_pos',
                 'cnt', 'sorted_order')

    def __init__(self, length=0, link=-1):
        self.len = length       # longest string in this equivalence class
        self.link = link        # suffix link (parent in suffix link tree)
        self.transitions = {}   # char -> state_id
        self.is_clone = False   # whether this state was created by cloning
        self.end_pos = -1       # end position of first occurrence
        self.cnt = 0            # occurrence count (set during count_occurrences)
        self.sorted_order = 0   # topological order


class SuffixAutomaton:
    """
    Online suffix automaton with O(n) construction.

    The automaton accepts exactly the set of all substrings of the input string.
    Each state represents an equivalence class of substrings that always occur
    together (same set of ending positions).
    """

    def __init__(self, s=""):
        # State 0 is the initial state
        self.states = [State(0, -1)]
        self.last = 0  # last state after extending
        self.size = 1  # number of states
        self._occurrences_computed = False
        self._text = []  # store original characters for reconstruction

        for ch in s:
            self.extend(ch)

    def _new_state(self, length=0, link=-1):
        """Create a new state and return its id."""
        sid = self.size
        self.states.append(State(length, link))
        self.size += 1
        return sid

    def extend(self, ch):
        """Extend the automaton by one character. O(1) amortized."""
        self._occurrences_computed = False
        self._text.append(ch)

        cur = self._new_state(self.states[self.last].len + 1)
        self.states[cur].end_pos = self.states[cur].len - 1
        self.states[cur].cnt = 1  # this is a new terminal suffix

        p = self.last
        while p != -1 and ch not in self.states[p].transitions:
            self.states[p].transitions[ch] = cur
            p = self.states[p].link

        if p == -1:
            # No existing state has this transition -- link to initial
            self.states[cur].link = 0
        else:
            q = self.states[p].transitions[ch]
            if self.states[p].len + 1 == self.states[q].len:
                # No need to split -- q is already the correct suffix link target
                self.states[cur].link = q
            else:
                # Clone q into clone, then redirect
                clone = self._new_state(self.states[p].len + 1, self.states[q].link)
                self.states[clone].transitions = dict(self.states[q].transitions)
                self.states[clone].is_clone = True
                self.states[clone].end_pos = self.states[q].end_pos

                # Redirect p and its suffix-link ancestors from q to clone
                while p != -1 and self.states[p].transitions.get(ch) == q:
                    self.states[p].transitions[ch] = clone
                    p = self.states[p].link

                self.states[q].link = clone
                self.states[cur].link = clone

        self.last = cur

    def contains(self, pattern):
        """Check if pattern is a substring of the original string. O(|pattern|)."""
        cur = 0
        for ch in pattern:
            if ch not in self.states[cur].transitions:
                return False
            cur = self.states[cur].transitions[ch]
        return True

    def count_distinct_substrings(self):
        """
        Count the number of distinct non-empty substrings.
        Each state represents substrings of length (link.len+1) to (self.len),
        so it represents (self.len - link.len) distinct substrings.
        """
        total = 0
        for i in range(1, self.size):
            st = self.states[i]
            link_len = self.states[st.link].len if st.link >= 0 else 0
            total += st.len - link_len
        return total

    def _topo_sort(self):
        """Sort states in reverse topological order by length (longest first)."""
        # Counting sort by length
        max_len = max(st.len for st in self.states)
        buckets = [0] * (max_len + 1)
        for i in range(self.size):
            buckets[self.states[i].len] += 1

        # Cumulative
        for i in range(1, max_len + 1):
            buckets[i] += buckets[i - 1]

        order = [0] * self.size
        for i in range(self.size - 1, -1, -1):
            l = self.states[i].len
            buckets[l] -= 1
            order[buckets[l]] = i

        return order  # sorted by length ascending

    def compute_occurrences(self):
        """
        Compute occurrence count for each state.
        Non-clone states start with cnt=1 (they represent a new suffix endpoint).
        Propagate counts up through suffix links.
        """
        if self._occurrences_computed:
            return

        order = self._topo_sort()

        # Process in reverse order (longest first) to propagate up suffix links
        for i in range(len(order) - 1, -1, -1):
            sid = order[i]
            st = self.states[sid]
            if st.link >= 0:
                self.states[st.link].cnt += st.cnt

        self._occurrences_computed = True

    def count_occurrences(self, pattern):
        """Count how many times pattern occurs as a substring. O(|pattern|)."""
        self.compute_occurrences()
        cur = 0
        for ch in pattern:
            if ch not in self.states[cur].transitions:
                return 0
            cur = self.states[cur].transitions[ch]
        return self.states[cur].cnt

    def first_occurrence(self, pattern):
        """
        Find the index of the first occurrence of pattern, or -1 if not found.
        Returns the starting index.
        """
        cur = 0
        for ch in pattern:
            if ch not in self.states[cur].transitions:
                return -1
            cur = self.states[cur].transitions[ch]
        return self.states[cur].end_pos - len(pattern) + 1

    def all_occurrences(self, pattern):
        """
        Find all starting positions where pattern occurs.
        Uses the suffix link tree to collect all end positions.
        """
        cur = 0
        for ch in pattern:
            if ch not in self.states[cur].transitions:
                return []
            cur = self.states[cur].transitions[ch]

        # BFS/DFS through the inverse suffix link tree from cur
        # Collect end_pos from all non-clone states in subtree
        # Build inverse suffix link tree
        children = defaultdict(list)
        for i in range(1, self.size):
            children[self.states[i].link].append(i)

        positions = []
        stack = [cur]
        while stack:
            sid = stack.pop()
            st = self.states[sid]
            if not st.is_clone:
                positions.append(st.end_pos - len(pattern) + 1)
            for child in children[sid]:
                stack.append(child)

        positions.sort()
        return positions

    def longest_common_substring(self, t):
        """
        Find the longest common substring between the automaton's string and t.
        Returns (length, start_in_t).
        O(|t|) time.
        """
        cur = 0
        cur_len = 0
        best_len = 0
        best_pos = -1

        for i, ch in enumerate(t):
            while cur != 0 and ch not in self.states[cur].transitions:
                cur = self.states[cur].link
                cur_len = self.states[cur].len

            if ch in self.states[cur].transitions:
                cur = self.states[cur].transitions[ch]
                cur_len += 1
            else:
                # Still at state 0 and no transition
                cur_len = 0

            if cur_len > best_len:
                best_len = cur_len
                best_pos = i - cur_len + 1

        return (best_len, best_pos)

    def shortest_non_occurring(self):
        """
        Find the lexicographically smallest shortest string that does NOT
        occur as a substring. BFS from initial state.
        """
        from collections import deque

        # BFS: find shortest path to a state missing some character
        # We need to find the shortest string not accepted
        # Strategy: BFS level by level, at each state try all chars a-z (or
        # all chars in the alphabet). First missing transition = answer.

        # Determine alphabet from all transitions
        alphabet = set()
        for i in range(self.size):
            alphabet.update(self.states[i].transitions.keys())
        if not alphabet:
            # Empty string automaton -- any single char works
            return 'a'
        alphabet = sorted(alphabet)

        queue = deque()
        queue.append((0, ""))

        visited = {0}
        while queue:
            sid, path = queue.popleft()
            for ch in alphabet:
                if ch not in self.states[sid].transitions:
                    return path + ch
                nxt = self.states[sid].transitions[ch]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + ch))

        # All strings of all lengths occur (impossible for finite string)
        # Return something longer than any substring
        return alphabet[0] * (self.states[self.last].len + 1)

    def longest_repeated_substring(self):
        """
        Find the longest substring that occurs at least twice.
        This is the longest string in a non-leaf state of the suffix link tree.
        """
        self.compute_occurrences()
        best_len = 0
        best_state = -1
        for i in range(1, self.size):
            if self.states[i].cnt >= 2 and self.states[i].len > best_len:
                best_len = self.states[i].len
                best_state = i

        if best_state == -1:
            return ""

        # Reconstruct the string by walking from initial state
        return self._reconstruct(best_state, best_len)

    def _reconstruct(self, target_state, length):
        """Reconstruct a string of given length that leads to target_state."""
        text = ''.join(self._text)
        # Find a non-clone descendant to get an end_pos
        if not self.states[target_state].is_clone:
            end_pos = self.states[target_state].end_pos
        else:
            # Walk down suffix link tree to find a non-clone state
            children = defaultdict(list)
            for i in range(1, self.size):
                children[self.states[i].link].append(i)
            stack = [target_state]
            end_pos = -1
            while stack:
                sid = stack.pop()
                if not self.states[sid].is_clone:
                    end_pos = self.states[sid].end_pos
                    break
                for child in children[sid]:
                    stack.append(child)

        if end_pos == -1:
            return ""
        start = end_pos - length + 1
        return text[start:end_pos + 1]

    def kth_smallest_substring(self, k):
        """
        Find the k-th lexicographically smallest distinct substring (1-indexed).
        Returns "" if k exceeds the number of distinct substrings.
        """
        # Count distinct substrings reachable from each state
        # paths[v] = number of distinct non-empty paths from v
        order = self._topo_sort()
        paths = [0] * self.size

        # Process in reverse topological order (longest first)
        for i in range(len(order) - 1, -1, -1):
            sid = order[i]
            st = self.states[sid]
            cnt = 0
            for ch in st.transitions:
                cnt += 1 + paths[st.transitions[ch]]
            paths[sid] = cnt

        if k < 1 or k > paths[0]:
            return ""

        # Walk the automaton, consuming k
        result = []
        cur = 0
        remaining = k
        while remaining > 0:
            for ch in sorted(self.states[cur].transitions.keys()):
                nxt = self.states[cur].transitions[ch]
                # Taking this transition accounts for 1 (the substring ending here)
                # plus all substrings extending further
                subtree = 1 + paths[nxt]
                if remaining <= subtree:
                    result.append(ch)
                    remaining -= 1  # count this substring
                    cur = nxt
                    break
                remaining -= subtree

        return ''.join(result)

    def suffix_links_tree(self):
        """
        Return the suffix link tree as adjacency list.
        Root is state 0. Each edge goes from parent to child.
        """
        children = defaultdict(list)
        for i in range(1, self.size):
            children[self.states[i].link].append(i)
        return dict(children)

    def num_states(self):
        """Return the number of states."""
        return self.size

    def num_transitions(self):
        """Return the total number of transitions."""
        return sum(len(st.transitions) for st in self.states)

    def to_dot(self, max_states=50):
        """Export to Graphviz DOT format for visualization."""
        lines = ['digraph SuffixAutomaton {', '  rankdir=LR;']
        n = min(self.size, max_states)
        for i in range(n):
            st = self.states[i]
            label = f"s{i}\\nlen={st.len}"
            shape = "doublecircle" if i == 0 else "circle"
            lines.append(f'  s{i} [label="{label}", shape={shape}];')

        for i in range(n):
            st = self.states[i]
            for ch, nxt in sorted(st.transitions.items()):
                if nxt < n:
                    lines.append(f'  s{i} -> s{nxt} [label="{ch}"];')
            if st.link >= 0 and st.link < n:
                lines.append(f'  s{i} -> s{st.link} [style=dashed, color=red];')

        lines.append('}')
        return '\n'.join(lines)


class GeneralizedSuffixAutomaton:
    """
    Generalized suffix automaton for multiple strings.
    Accepts all substrings of all input strings.
    Supports per-string occurrence tracking.
    """

    def __init__(self, strings=None):
        self.sa = SuffixAutomaton()
        self.num_strings = 0
        self._strings = []  # store all added strings
        # Track which strings each state belongs to
        self._string_sets = defaultdict(set)

        if strings:
            for s in strings:
                self.add_string(s)

    def add_string(self, s):
        """Add a string to the generalized automaton."""
        string_id = self.num_strings
        self.num_strings += 1
        self._strings.append(s)

        # Reset to initial state for each new string
        self.sa.last = 0
        self.sa._occurrences_computed = False

        for ch in s:
            # Check if transition already exists from last
            if ch in self.sa.states[self.sa.last].transitions:
                nxt = self.sa.states[self.sa.last].transitions[ch]
                if self.sa.states[self.sa.last].len + 1 == self.sa.states[nxt].len:
                    # Can just follow the existing transition
                    self.sa.last = nxt
                    self._string_sets[nxt].add(string_id)
                    continue
                else:
                    # Need to clone
                    clone = self.sa._new_state(
                        self.sa.states[self.sa.last].len + 1,
                        self.sa.states[nxt].link
                    )
                    self.sa.states[clone].transitions = dict(self.sa.states[nxt].transitions)
                    self.sa.states[clone].is_clone = True
                    self.sa.states[clone].end_pos = self.sa.states[nxt].end_pos

                    p = self.sa.last
                    while p != -1 and self.sa.states[p].transitions.get(ch) == nxt:
                        self.sa.states[p].transitions[ch] = clone
                        p = self.sa.states[p].link

                    self.sa.states[nxt].link = clone
                    self.sa.last = clone
                    self._string_sets[clone].add(string_id)
                    self._string_sets[clone].update(self._string_sets[nxt])
                    continue

            self.sa.extend(ch)
            self._string_sets[self.sa.last].add(string_id)

        # Propagate string sets up suffix links
        cur = self.sa.last
        while cur > 0:
            if string_id in self._string_sets[cur]:
                # Already propagated from a previous char
                if cur != self.sa.last:
                    break
            self._string_sets[cur].add(string_id)
            cur = self.sa.states[cur].link

    def contains(self, pattern):
        """Check if pattern is a substring of any added string."""
        return self.sa.contains(pattern)

    def strings_containing(self, pattern):
        """Return set of string IDs that contain pattern as a substring."""
        cur = 0
        for ch in pattern:
            if ch not in self.sa.states[cur].transitions:
                return set()
            cur = self.sa.states[cur].transitions[ch]

        # Collect all string IDs from this state and its subtree
        children = defaultdict(list)
        for i in range(1, self.sa.size):
            children[self.sa.states[i].link].append(i)

        result = set()
        stack = [cur]
        while stack:
            sid = stack.pop()
            result.update(self._string_sets[sid])
            for child in children[sid]:
                stack.append(child)

        return result

    def longest_common_substring_all(self):
        """
        Find the longest common substring of all added strings.
        Returns the substring (or "" if none).
        """
        if self.num_strings == 0:
            return ""
        if self.num_strings == 1:
            # Only one string, return the whole thing
            # (longest substring of itself)
            best = 0
            for i in range(1, self.sa.size):
                if self.sa.states[i].len > self.sa.states[best].len:
                    best = i
            return self._reconstruct_from_state(best)

        # Propagate string sets through suffix links
        children = defaultdict(list)
        for i in range(1, self.sa.size):
            children[self.sa.states[i].link].append(i)

        # DFS to propagate string sets up
        order = self.sa._topo_sort()
        full_sets = defaultdict(set)
        for i in range(self.sa.size):
            full_sets[i] = set(self._string_sets[i])

        for i in range(len(order) - 1, -1, -1):
            sid = order[i]
            link = self.sa.states[sid].link
            if link >= 0:
                full_sets[link].update(full_sets[sid])

        # Find state with all strings and max length
        best_len = 0
        best_state = -1
        target = set(range(self.num_strings))
        for i in range(1, self.sa.size):
            if full_sets[i] == target and self.sa.states[i].len > best_len:
                best_len = self.sa.states[i].len
                best_state = i

        if best_state == -1:
            return ""

        return self._reconstruct_from_state(best_state)

    def _reconstruct_from_state(self, state):
        """Reconstruct the longest string represented by a state."""
        length = self.sa.states[state].len

        # DFS from state 0 to find a path of exactly `length` to target state
        def dfs(sid, depth):
            if depth == length and sid == state:
                return []
            if depth >= length:
                return None
            for ch in sorted(self.sa.states[sid].transitions.keys()):
                nxt = self.sa.states[sid].transitions[ch]
                result = dfs(nxt, depth + 1)
                if result is not None:
                    result.insert(0, ch)
                    return result
            return None

        result = dfs(0, 0)
        return ''.join(result) if result else ""
