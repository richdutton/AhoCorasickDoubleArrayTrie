# todo: does this thing have to be pickled to work on workers? surely yes. and surely that's not gonna work if it's compiled

# todo: test performance of our a/c against the python compiled package. if it's faster try on spark
# todo: implement state (not a lot of work)
# todo: there are issues with the collections used
#       for simple collections it seems ok to use np.array
#       it seems v can be done as an [] - it's just sized once, based on the map that's passed to build
#       it seems siblings can also be an [] (although an slist might be better)
#       so we're probably good!
# todo: implement tests (not a lot of work)
# todo: make it work
# todo: test performance against faster of our aho_corasick and the python compiled  aho_corasick package

from collections import deque
import numpy as np


# todo: use attr library
class Hit:
    def __init__(self, begin, end, value):  # int, int, V
        self._begin = begin
        self._end = end
        self._value = value

    # todo: define repr and str if need to before moving to attr - String.format("[%d:%d]=%s", begin, end, value);


class Builder:
    def __init__(self, aho_corasick_double_array_trie):
        # todo: may not need to define all these yet
        self._root_state = State()
        self._used = np.array((), dtype=bool)
        self._alloc_size = 0  # int
        self._progress = 0  # int
        self._next_check_pos = 0  # int
        self._key_size = 0  # int
        # todo: is this going to actually work?
        self._aho_corasick_double_array_trie = aho_corasick_double_array_trie

    def build(self, map):  # Map<String, V>
        # todo: if we're going to mess with it these shouldn't be hidden (but this whole thing is far from ideal anyway)
        self._aho_corasick_double_array_trie._v = map.values()
        # todo: might be better served by array or np.array. tbh this could be true of a lot of other []s
        l = np.array((), dtype=int)
        l.resize(len(self._aho_corasick_double_array_trie._v))
        self._aho_corasick_double_array_trie._l = l

        key_set = set(map.keys())
        self._add_all_keywords(key_set)
        self.build_double_array_trie(len(key_set))
        self._used = None
        self._construct_failure_states()
        # todo: seems weird that we'd create it in init and kill it here
        self._root_state = None
        self.lose_weight()

    def _fetch(parent, siblings):  # State, List<Map.Entry<Integer, State>> -> int
        if parent.is_acceptable():
            fake_node = State(-(parent.get_depth() + 1))
            fake_node.add_emit(parent.get_largest_value_id())
            # todo: for now assuming that the best analogue of siblings' type is array of pair but linked list might prove better. at least namedtuple
            siblings.append((0, fake_node))

        # todo: entry_set should be a property
        for entry in parent.get_success().entry_set():
            siblings.add((entry.get_key() + 1, entry.get_value()))

        return len(siblings)

    def _add_keyword(self, keyword, index):  # string, int -> void
        current_state = self._root_state
        for character in keyword:
            current_state = current_state.add_state(character)

        current_state.add_emit(index)

    # todo: should be plural
    def _add_all_keyword(self, keyword_set):  # Collection<String> -> void
        i = 0
        for keyword in keyword_set:
            self._add_keyword(keyword, i)
            # todo: is this whole index thing really necessary?
            i += 1

    def _construct_failure_states(self):  # -> void
        fail = np.array((), dtype=int)
        fail.resize(self._aho_corasick_double_array_trie._size + 1)
        fail[1] = self._aho_corasick_double_array_trie._base[0]
        self._aho_corasick_double_array_trie.output = np.array((), dtype=int)
        fail.resize(self._aho_corasick_double_array_trie._size + 1)
        # todo: is this type sufficient?
        queue = deque()  # new LinkedBlockingDeque<State>()

        # todo: states could be property
        for depth_one_state in self._root_state.get_states():
            depth_one_state.set_failure(self._root_state, fail)
            queue.add(depth_one_state)
            self._construct_output(depth_one_state)

        while len(queue) != 0:
            current_state = queue.pop()

            # todo: could also be a property
            for transition in current_state.get_transitions():
                target_state = current_state.next_state(transition)
                queue.append(target_state)

                trace_failure_state = current_state.failure()
                while trace_failure_state.next_state(transition is None):
                    trace_failure_state = trace_failure_state.failure()

                new_failure_state = trace_failure_state.next_state(transition)
                target_state.set_failure(new_failure_state, fail)
                target_state.add_emit(new_failure_state.emit())
                self._construct_output(target_state)

    def _construct_output(self, target_state):  # State -> void
        emit = target_state.emit()  # Collection<Integer>
        if emit is None or len(emit) == 0:
            return

        output = np.array((), dtype=int)
        output.resize(emit.size())
        it = emit.iterator()
        for i in it:
            output.append(i)

        self._aho_corasick_double_array_trie._output[target_state.get_index()] = output

    def _build_double_array_trie(self, key_size):  # int -> void
        self._progress = 0
        self._key_size = key_size
        self._resize(65536 * 32)

        # todo: is this right? (from base[0] = 1)
        self._base.append(1)
        self._next_check_pos = 0

        # todo: this seems pointless
        root_node = self._root_state
        siblings = []  # new ArrayList<Map.Entry<Integer, State>>(root_node.getSuccess().entrySet().size());
        self._fetch(root_node, siblings)
        self._insert(siblings)

    def _resize(self, new_size):  # int -> int
        assert self._alloc_size == len(self._aho_corasick_double_array_trie._base)
        assert self._alloc_size == len(self._aho_corasick_double_array_trie._check)
        assert self._alloc_size == len(self._used)

        self._aho_corasick_double_array_trie._base.resize(new_size)
        self._aho_corasick_double_array_trie._check.resize(new_size)
        self._used.resize.resize(new_size)

        # todo: is alloc size even necessary?
        self._alloc_size = new_size

        return self._alloc_size

    def insert(self, siblings):  # List<Map.Entry<Integer, State>>  -> int
        begin = 0  # int
        pos = max(siblings.get(0).get_key() + 1, self._next_check_pos) - 1
        nonzero_num = 0  # int
        first = 0  # int

        if self._alloc_size <= pos:
            self._resize(pos + 1)

        while True:
            pos += 1

            if self._alloc_size <= pos:
                self._resize(pos + 1)

            if self._aho_corasick_double_array_trie._check[pos] != 0:
                nonzero_num += 1
                continue
            elif first == 0:
                next_check_pos = pos
                first = 1

            # todo: get is horrible java syntax and will not work on an array or anything like it
            # also: i don't understand how this would ever work as the array is seemingly empty by the time we're called (by insert itself, below)
            begin = pos - siblings.get(0).get_key()
            # todo: more like a [-1]
            if self._alloc_size <= begin + siblings.get(siblings.size() - 1).get_key():
                l = 1.05 if 1.05 > 1.0 * self._key_size / (self._progress + 1) else 1.0 * self._key_size / (self._progress + 1)
                self.resize(self._alloc_size * l)

            if self._used[begin]:
                continue

            break_flag = True
            # todo: better as for sibling in siblings
            for i in range(1, siblings.size() - 1):
                if self._aho_corasick_double_array_trie._check[begin + siblings.get(i).get_key()] != 0:
                    break_flag = False
                    break

            if break_flag:
                break

        if 1.0 * nonzero_num / (pos - next_check_pos + 1) >= 0.95:
            next_check_pos = pos

        self._used[begin] = True
        self._size = self._size if self._size > begin + siblings.get(siblings.size() - 1).get_key() + 1 else begin + siblings.get(siblings.size() - 1).get_key() + 1

        for integer, state in siblings:
            self._aho_corasick_double_array_trie._check[begin + integer] = begin

        # todo: for sibling in siblings?
        for integer, state in siblings:
            # todo: this will never work. maybe an array (not np) of pairs
            new_siblings = []

            if self._fetch(state, new_siblings) == 0:
                self._aho_corasick_double_array_trie._base[begin + integer] = -state.get_largest_value_id() - 1
                self._progress += 1
            else:
                h = self._insert(new_siblings)
                self._aho_corasick_double_array_trie._base[begin + integer] = h

            state.set_index(begin + integer)

        return begin

    def _lose_weight(self):
        self._aho_corasick_double_array_trie._base.resize(self._aho_corasick_double_array_trie._base._size + 65535)
        self._aho_corasick_double_array_trie._check.resize(self._aho_corasick_double_array_trie._check._size + 65535)


class AhoCorasickDoubleArrayTrie:
    # todo: should we be initializing arrays to None?
    def __init__(self):
        # todo: may not need these definitions but they're informative for now
        self._check = np.array((), dtype=int)
        self._base = np.array((), dtype=int)
        self._fail = np.array((), dtype=int)
        self._output = np.array((), dtype=int)
        # todo: this would have to be an array
        self._v = []  # of V[]
        self._l = np.array((), dtype=int)
        self._size = None  # int

    def parse_text(self, text):  # String -> List<Hit<V>>
        position = 1
        current_state = 0
        # todo: would this be better as a list?
        collected_emits = []  # LinkedList<Hit<V>>

        for character in text:  # int
            current_state = self._get_state(current_state, character)
            self._store_emits(position, current_state, collected_emits)
            position += 1

        return collected_emits

    def parse_text_with_processor(self, text, processor):  # String or char[], IHit<V> or IHitFull<V> -> void
        position = 1
        current_state = 0

        for character in text:  # int
            current_state = self._get_state(current_state, character)
            hit_array = self._output[current_state]  # int[]
            if hit_array is not None:
                for hit in hit_array:  # int
                    processor.hit(position - self._l[hit], position, self._v[hit])
            position += 1

    # note: other variants of parse_text will be accommodated through dynamic typing by the above
    # note: not implementing serialization methods save and load

    def get_by_str(self, key):
        index = self.exact_match_search(key)
        return self._v[index] if index >= 0 else None

    def get_by_int(self, index):
        return self._v[index]

    # note: not defining interfaces
    def _get_state(self, current_state, character):  # int, character -> int
        new_current_state = self._transition_with_root(current_state, character)
        while new_current_state == -1:
            current_state = self._fail[current_state]
            new_current_state = self._transition_with_root(current_state, character)

        return new_current_state

    def _store_emits(self, position, current_state, collected_emits):  # int, int, List<Hit><V> -> void
        hit_array = self._output[current_state]  # int[]
        if hit_array is not None:
            for hit in hit_array:
                collected_emits.append(Hit(position - self._l[hit], position, self._v[hit]))

    def _transition(self, current, c):  # int, char -> int
        b = current
        p = b + c + 1
        if b == self._check[p]:
            b = self._base[p]
        else:
            return -1

        # todo: may as well return b
        p = b
        return p

    def _transition_with_root(self, node_pos, c):  # int, char -> int
        b = self._base[node_pos]
        p = b + c + 1
        if b != self._check[p]:
            return 0 if node_pos == 0 else -1

        return p

    def build(self, map):  # Map<String, V> -> void
        Builder(self).build(map)

    def exact_match_search(self, key):
        # todo: make these default arguments
        return self._exact_match_search_with_indices(key, 0, 0, 0)

    def _exact_match_search_with_indices(self, key, pos, length, node_pos):  # String, int, int, int -> int
        if length <= 0:
            length = len(key)
        if node_pos <= 0:
            node_pos = 0

        result = -1

        b = self._base[node_pos]

        for i in range(pos, length):
            p = b + self.key_chars[i] + 1
            if b == self._check[p]:
                b = self._base[p]
            else:
                return result

        p = b
        n = self._base[p]
        if b == self._check[p] and n < 0:
            result = -n - 1

        return result

        # note: not implementing char[] variant as is handled by above

        # todo: property
        def size(self):
            return len(self._v)
