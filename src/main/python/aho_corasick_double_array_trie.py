# todo: use attr library
class Hit:
    def __init__(self, begin, end, value):  # int, int, V
        self._begin = begin
        self._end = end
        self._value = value

    # todo: define repr and str if need to before moving to attr - String.format("[%d:%d]=%s", begin, end, value);


class Builder:
    pass  # todo:


class AhoCorasickDoubleArrayTrie:
    def __init__(self):
        # todo: may not need these definitions but they're informative for now
        self._check = []  # int[]
        self._base = []  # int[]
        self._fail = []  # int[]
        self._output = []  # int[][]
        self._v = []  # of V[]
        self._l = []  # int[]
        self._size = None  # int

    def parse_text(self, text):  # String -> List<Hit<V>>
        position = 1
        current_state = 0
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

    def build(map):  # Map<String, V> -> void
        Builder().build(map)

    def exact_match_search(self, key):
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
