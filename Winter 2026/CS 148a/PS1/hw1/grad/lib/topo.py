from ..autograd import Value


def topo_sort(result):
    _ordered: list[Value] = list()
    _visited: set[Value] = set()

    def add_list(curr):
        if curr not in _visited:
            _visited.add(curr)
            for p in curr.prev:
                add_list(p)
            _ordered.append(curr)

    add_list(result)
    return _ordered
