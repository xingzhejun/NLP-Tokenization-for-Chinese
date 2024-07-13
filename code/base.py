# 基类
import os
from trie import Trie


class BaseSegment:
    def __init__(self):
        self.trie = Trie()

    def cut(self, sentence: str):
        pass

    def load_dict(self, file: str):
        if not os.path.exists(file):
            print('%s 不存在！' % file)
            return

        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                if len(line) == 3:
                    try:
                        self.trie.insert(line[0], int(line[1]), line[2])
                    except ValueError:
                        print(line[1])
                        print(line[2])
                elif len(line) == 2:
                    try:
                        self.trie.insert(line[0], int(line[1]))
                    except ValueError:
                        print(line[1])
                else:
                    try:
                        self.trie.insert(line[0])
                    except:
                        print(' ')
        f.close()
        print(file + '词典加载完成！')
