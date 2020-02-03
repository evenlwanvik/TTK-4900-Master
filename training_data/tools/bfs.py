from collections import deque
from operator import eq

def bfs(array, neighbors, start, similar):
    """Run a BFS algorithm based on criteria from arguments."""
    match = get_item(array, start)
    block = {start}
    visit = deque(block)
    child = deque.popleft
    while visit:
        node = child(visit)
        for offset in neighbors:
            index = get_next(node, offset)
            if index not in block:
                block.add(index)
                if is_valid(array, index):
                    value = get_item(array, index)
                    if similar(value, match):
                        visit.append(index)
        yield node


def get_item(array, index):
    """Access the data structure based on the given position information."""
    row, column = index
    return array[row][column]


def get_next(node, offset):
    """Find the next location based on an offset from the current location."""
    row, column = node
    row_offset, column_offset = offset
    return row + row_offset, column + column_offset


def is_valid(array, index):
    """Verify that the index is in range of the data structure's contents."""
    row, column = index
    return 0 <= row < len(array) and 0 <= column < len(array[row])