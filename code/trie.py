# 树状词典搜索
"""
    Trie：前缀字典树
    insert：添加字符串
    search：查找字符串
    get_freq: 获取词频
"""


class Trie:
    def __init__(self):
        self.root = {}
        self.max_word_len = 0
        self.total_word_freq = 0
        self.end_token = '[END]'
        self.freq_token = '[FREQ]'

    def insert(self, word: str, freq: int = 1, tag: str = None):
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        self.max_word_len = max(self.max_word_len, len(word))
        node[self.end_token] = self.end_token
        node[self.freq_token] = freq
        self.total_word_freq += freq

    def search(self, word: str):
        node = self.root
        for char in word:
            if char not in node:
                return None
            node = node[char]
        return node if self.end_token in node else None

    def get_freq(self, word: str):
        node = self.search(word)
        if node:
            return node.get(self.freq_token, 1)
        else:
            return 0

