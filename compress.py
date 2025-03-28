"""
Assignment 2 starter code
CSC148, Winter 2025
Instructors: Bogdan Simion, Rutwa Engineer, Marc De Benedetti, Romina Piunno

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2025 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

import time
from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    d = {}
    for b in text:
        d.setdefault(b, 0)
        d[b] += 1
    return d


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    if len(freq_dict) == 1:
        symb = next(iter(freq_dict))
        return HuffmanTree(None, HuffmanTree(symb), HuffmanTree((symb + 1) % 256))
    srtd_keys = sorted(freq_dict, key=lambda x: freq_dict[x])
    l2 = []
    for k in srtd_keys:
        l2.append((HuffmanTree(k), freq_dict[k], 1))
    while len(l2) > 1:
        val1, val2 = l2[0], l2[1]
        l2 = l2[2:]
        l2.append((HuffmanTree(None, val1[0], val2[0]), val1[1] + val2[1], val1[2] + val2[2]))
        l2.sort(key=lambda x: (x[1], x[2]))
    return l2[0][0]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(None, HuffmanTree(3), HuffmanTree(69)), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "00", 69: "01",2: "1"}
    True
    """
    codes = {}

    def _traverse(huffman: HuffmanTree, code: str) -> None:
        """#TODO put in the docstring"""
        if huffman.symbol is not None:
            codes[huffman.symbol] = code
            return
        if huffman.left is not None:
            _traverse(huffman.left, code + '0')
        if huffman.right is not None:
            _traverse(huffman.right, code + '1')

    _traverse(tree, '')
    return codes


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(None, HuffmanTree(3), HuffmanTree(2)), HuffmanTree(5))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    1
    >>> tree.right.number
    2
    >>> tree.number
    3
    """

    def _number_nodes_helper(huffman: HuffmanTree, counter: list) -> None:
        #TODO put in the docstring
        """
        """
        if huffman is None:
            return
        _number_nodes_helper(huffman.left, counter)
        _number_nodes_helper(huffman.right, counter)
        if huffman.symbol is None:
            huffman.number = counter[0]
            counter[0] += 1

    _number_nodes_helper(tree, [0])


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    d = get_codes(tree)
    denominator = 0
    numerator = 0
    for key in d:
        freq = freq_dict[key]
        numerator += (len(d[key]) * freq)
        denominator += freq
    return numerator / denominator


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    temp = ''
    ret = []
    for b in text:
        temp += codes[b]
    if temp:
        for i in range(0, len(temp), 8):
            ret.append(bits_to_byte(temp[i: i + 8]))
    return bytes(ret)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    result = []

    def _post_order(huffman: HuffmanTree) -> None:
        if not huffman:
            return
        if huffman.left and not huffman.left.is_leaf():
            _post_order(huffman.left)
        if huffman.right and not huffman.right.is_leaf():
            _post_order(huffman.right)
        if huffman.symbol is None:
            if huffman.left.is_leaf():
                result.append(0)
                result.append(huffman.left.symbol)
            else:
                result.append(1)
                result.append(huffman.left.number)
            if huffman.right.is_leaf():
                result.append(0)
                result.append(huffman.right.symbol)
            else:
                result.append(1)
                result.append(huffman.right.number)

    _post_order(tree)
    return bytes(result)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    if node_lst and root_index < len(node_lst):
        root_node = node_lst[root_index]

        if root_node.l_type == 0 and root_node.r_type == 0:
            return HuffmanTree(None,
                               HuffmanTree(root_node.l_data),
                               HuffmanTree(root_node.r_data))

        elif root_node.l_type == 0 and root_node.r_type == 1:
            l_subtree = HuffmanTree(root_node.l_data, None, None)
            r_subtree = generate_tree_general(node_lst, root_node.r_data)
            return HuffmanTree(None, l_subtree, r_subtree)

        elif root_node.l_type == 1 and root_node.r_type == 0:
            l_subtree = generate_tree_general(node_lst, root_node.l_data)
            r_subtree = HuffmanTree(root_node.r_data, None, None)
            return HuffmanTree(None, l_subtree, r_subtree)

        elif root_node.l_type == 1 and root_node.r_type == 1:
            l_subtree = generate_tree_general(node_lst, root_node.l_data)
            r_subtree = generate_tree_general(node_lst, root_node.r_data)
            return HuffmanTree(None, l_subtree, r_subtree)
    return HuffmanTree()

def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    # TODO adjust for empty inputs???
    stack = []
    for i in range(root_index + 1):
        node = node_lst[i]

        if node.l_type == 0 and node.r_type == 0:
            left_child = HuffmanTree(node.l_data, None, None)
            right_child = HuffmanTree(node.r_data, None, None)
            stack.append(HuffmanTree(None, left_child, right_child))
        elif node.l_type == 0 and node.r_type == 1:
            left_child = HuffmanTree(node.l_data, None, None)
            right_child = stack.pop()
            stack.append(HuffmanTree(None, left_child, right_child))
        elif node.l_type == 1 and node.r_type == 0:
            left_child = stack.pop()
            right_child = HuffmanTree(node.r_data, None, None)
            stack.append(HuffmanTree(None, left_child, right_child))
        elif node.l_type == 1 and node.r_type == 1:
            right_child = stack.pop()
            left_child = stack.pop()
            stack.append(HuffmanTree(None, left_child, right_child))
    if stack:
        return stack[-1]
    return HuffmanTree()


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    # TODO adjust spacing
    bits = ''.join(byte_to_bits(b) for b in text)
    chars = []
    bit_index = 0

    while len(chars) < size and bit_index < len(bits):
        current_node = tree

        while current_node.symbol is None and bit_index < len(bits):
            bit = bits[bit_index]
            if bit == '0':
                current_node = current_node.left
            else:
                current_node = current_node.right
            bit_index += 1

        if current_node.symbol is not None:
            chars.append(current_node.symbol)

    return bytes(chars)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    nodes = []
    freqs = sorted(freq_dict, key=lambda x: freq_dict[x])

    def _recurse(huffman: HuffmanTree, code: str) -> None:
        if not huffman:
            return
        if huffman.symbol is not None:
            nodes.append((huffman, code))
            return
        if huffman.right is not None:
            _recurse(huffman.left, code + '1')
        if huffman.left is not None:
            _recurse(huffman.right, code + '1')

    _recurse(tree, '')

    if nodes:
        nodes.sort(key=lambda x: x[1], reverse=True)

    for i in range(len(nodes)):
        nodes[i][0].symbol = freqs[i]


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
