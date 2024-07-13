# 隐马尔可夫模型HMM算法
import numpy as np
from data_process import HMMDataLoader


class HMM:
    def __init__(self, data_path='icwb2-data/training/pku_training.utf8'):
        self.A = None   # 状态转移矩阵
        self.B = None   # 状态与观察序列转移矩阵
        self.Pi = None  # 初始状态矩阵
        self.data_loader = HMMDataLoader(data_path)
        self._get_index()

    def _get_index(self):
        self.idxed_corpus, (self.obsv2idx, self.idx2obsv), (self.hide2idx, self.idx2hide) = self.data_loader.index_corpus()
        self.num_obsv = len(self.obsv2idx.keys())
        self.num_hide = len(self.hide2idx.keys())
        print("Status dict:", self.hide2idx)

    def build_supervised_model(self, smooth="add1"):
        if self.num_hide and self.num_obsv:
            self.Pi = np.zeros(self.num_hide)
            self.A = np.zeros([self.num_hide, self.num_hide])
            self.B = np.zeros([self.num_obsv, self.num_hide])
        else:
            self.Pi = None
            self.A = None
            self.B = None

        # 统计频率，计算A，B，Pi参数
        for seq in self.idxed_corpus:
            for i in range(len(seq)):
                obsv_cur, hide_cur = seq[i]

                if (i == 0):
                    self.Pi[hide_cur] += 1
                else:
                    obsv_pre, hide_pre = seq[i - 1]
                    self.A[hide_cur, hide_pre] += 1

                self.B[obsv_cur, hide_cur] += 1

        # 平滑
        if smooth == 'add1':
            self.A += 1
            self.B += 1
            self.Pi += 1

            self.Pi /= self.Pi.sum()
            self.A /= self.A.sum(axis=1)[:, None]
            self.B /= self.B.sum(axis=1)[:, None]

        return self.A, self.B, self.Pi

    def get_status_seq(self, obsv_seq):
        return self._veterbi(obsv_seq)

    def _veterbi(self, obsv_seq):
        # 初始化
        len_seq = len(obsv_seq)
        f = np.zeros([len_seq, self.num_hide])
        f_arg = np.zeros([len_seq, self.num_hide], dtype=int)
        for i in range(0, self.num_hide):
            f[0, i] = self.Pi[i] * self.B[obsv_seq[0], i]
            f_arg[0, i] = 0
        # 动态规划求解
        for i in range(1, len_seq):
            for j in range(self.num_hide):
                fs = [f[i-1, k] * self.A[j, k] * self.B[obsv_seq[i], j] for k in range(self.num_hide)]
                f[i, j] = max(fs)
                f_arg[i, j] = np.argmax(fs)
        # 反向求解概率最大的隐藏序列
        hidden_seq = [0] * len_seq
        z = np.argmax(f[len_seq-1, self.num_hide-1])
        hidden_seq[len_seq-1] = z
        for i in reversed(range(1, len_seq)):
            z = f_arg[i, z]
            hidden_seq[i-1] = z
        return hidden_seq

    def cut(self, sentence):
        sentence = sentence.strip()
        idxed_seq = [self.obsv2idx[obsv] if obsv in self.obsv2idx.keys() else 0 for obsv in sentence]
        idxed_hide = self.get_status_seq(idxed_seq)
        hide = [self.idx2hide[idx] for idx in idxed_hide]
        assert len(sentence) == len(hide), "状态序列与观测序列长度不一致"

        words = []
        lo, hi = 0, 0


        for i in range(len(hide)):
            if hide[i] in ['B', 'S']:
                # print(1)
                words.append(sentence[i])
            else:
                words[-1] += sentence[i]

            # if hide[i] == 'B':
            #     lo = i
            # elif hide[i] == 'E':
            #     hi = i + 1
            #     words.append(sentence[lo:hi])
            # elif hide[i] == 'S':
            #     words.append(sentence[i:i + 1])

        # if hide[-1] == 'B':
        #     words.append(sentence[-1])  # 处理 SB,EB
        # elif hide[-1] == 'M':
        #     words.append(sentence[lo:-1])

        assert len(sentence) == len("".join(words)), "还原失败,长度不一致\n{0}\n{1}\n{2}".format(sentence, "".join(words),
                                                                                        "".join(hide))
        return words
