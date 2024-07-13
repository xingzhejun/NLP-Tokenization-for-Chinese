class HMMDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.vocab_dict = {}
        self.corpus = []

    # 将原始分词数据集处理成BMES标记的数据集
    def generate_corpus_status(self, encoding="utf-8"):
        with open(self.data_path, 'r', encoding=encoding) as f:
            for line in f:
                l = []
                for word in line.strip().split():
                    if len(word) == 1:
                        l.append((word[0], 'S'))
                        continue
                    for i in range(len(word)):
                        if i == 0:
                            l.append((word[i], 'B'))
                        elif i == len(word) - 1:
                            l.append((word[i], 'E'))
                        else:
                            l.append((word[i], 'M'))
                self.corpus.append(l)
        return self.corpus

    #  对数据集进行编码
    def index_corpus(self):
        obsv2idx, idx2obsv = {'unk': 0}, {0: 'unk'}
        hide2idx, idx2hide = {}, {}
        obsv_idx, hide_idx = 1, 0

        idxed_corpus = []
        for seq in self.generate_corpus_status():
            idxed_seq = []
            for obsv, hide in seq:
                if obsv not in obsv2idx.keys():
                    obsv2idx[obsv] = obsv_idx
                    idx2obsv[obsv_idx] = obsv
                    obsv_idx += 1
                if hide not in hide2idx.keys():
                    hide2idx[hide] = hide_idx
                    idx2hide[hide_idx] = hide
                    hide_idx += 1
                # indexing
                idxed_seq.append((obsv2idx[obsv], hide2idx[hide]))
            idxed_corpus.append(idxed_seq)
        return idxed_corpus, (obsv2idx, idx2obsv), (hide2idx, idx2hide)

    def generate_vocab_dict(self, encoding='utf-8'):
        """
        统计语料库，获取词表
        :param encoding: 编码方式
        :return: 词表字典
        """
        with open(self.data_path, 'r', encoding=encoding) as f:
            print("Start generate vocab dict...")

            for line in f.readlines():
                for word in line.strip().split():
                    self.vocab_dict[word] = self.vocab_dict.get(word, 0) + 1
            count = len(self.vocab_dict)
            self.vocab_dict['_total_'] = count
            print("Finished. Total number of words: {0}".format(count))

        return self.vocab_dict

