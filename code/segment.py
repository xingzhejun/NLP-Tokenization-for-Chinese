import os
from fmm import FMMSegment
from rmm import RMMSegment
from mmseg import MMSegment
from hmm import HMM
from unigram_model import UniGramSeg
from data_process import HMMDataLoader
import jieba


# 传统的匹配算法
def traditional_matching_seg(test_path, name, vocabulary_path, encoding="utf-8"):
    fmm = FMMSegment(vocabulary_path)
    rmm = RMMSegment(vocabulary_path)
    mmseg = MMSegment(vocabulary_path)
    pred_fmm = []
    pred_rmm = []
    pred_mmseg = []
    save_path_fmm = os.path.join('test_result/traditional/', name + '_fmm_seg.utf8')
    save_path_rmm = os.path.join('test_result/traditional/', name + '_rmm_seg.utf8')
    save_path_mmseg = os.path.join('test_result/traditional/', name + '_mmseg_seg.utf8')

    # fmm
    with open(test_path, "r", encoding=encoding) as f:
        lines = f.readlines()
        count = 0
        total_count = len(lines)
        for sent in lines:
            try:
                pred_fmm.append(fmm.cut(sent))
            except:
                pred_fmm.append(["Error"] * len(sent))
                continue
            count += 1
            if count % 500 == 0:
                print("Processed FMM {0}/{1} ---- {2}".format(count, total_count, count / total_count))
    f.close()

    #rmm
    with open(test_path, "r", encoding=encoding) as f:
        lines = f.readlines()
        count = 0
        total_count = len(lines)
        for sent in lines:
            try:
                pred_rmm.append(rmm.cut(sent))
            except:
                pred_rmm.append(["Error"] * len(sent))
                continue
            count += 1
            if count % 500 == 0:
                print("Processed RMM{0}/{1} ---- {2}".format(count, total_count, count / total_count))
    f.close()

    # mmseg
    with open(test_path, "r", encoding=encoding) as f:
        lines = f.readlines()
        count = 0
        total_count = len(lines)
        for sent in lines:
            try:
                pred_mmseg.append(mmseg.cut(sent))
            except:
                pred_mmseg.append(["Error"] * len(sent))
                continue
            count += 1
            if count % 500 == 0:
                print("Processed MMSeg{0}/{1} ---- {2}".format(count, total_count, count / total_count))
    f.close()

    # 保存结果
    with open(save_path_fmm, "w", encoding='utf-8') as f:
        for words in pred_fmm:
            s = " ".join(words)
            s = s.strip() + '\n'
            s.encode('utf-8')
            f.write(s)
    f.close()
    print("Segmentation result is saved in {0}.".format(save_path_fmm))
    with open(save_path_rmm, "w", encoding='utf-8') as f:
        for words in pred_rmm:
            s = " ".join(words)
            s = s.strip() + '\n'
            s.encode('utf-8')
            f.write(s)
    f.close()
    print("Segmentation result is saved in {0}.".format(save_path_rmm))
    with open(save_path_mmseg, "w", encoding='utf-8') as f:
        for words in pred_mmseg:
            s = " ".join(words)
            s = s.strip() + '\n'
            s.encode('utf-8')
            f.write(s)
    f.close()
    print("Segmentation result is saved in {0}.".format(save_path_mmseg))

# 不含神经网络的机器学习算法
def ML_without_nn_seg(test_path, name, data_path, encoding="utf-8"):
    # hmm
    hmm = HMM(data_path)
    pred_hmm = []
    save_path_hmm = os.path.join('test_result/ML_without_nn', name + '_hmm_seg.utf8')

    A, B, Pi = hmm.build_supervised_model()
    with open(test_path, "r", encoding=encoding) as f:
        lines = f.readlines()
        count = 0
        total_count = len(lines)
        for sent in lines:
            try:
                pred_hmm.append(hmm.cut(sent))
            except:
                pred_hmm.append(["Error"] * len(sent))
                continue
            count += 1
            if count % 500 == 0:
                print("Processed HMM {0}/{1} ---- {2}".format(count, total_count, count / total_count))
    f.close()

    # 保存结果
    with open(save_path_hmm, "w", encoding='utf-8') as f:
        for words in pred_hmm:
            s = " ".join(words)
            s = s.strip() + '\n'
            s.encode('utf-8')
            f.write(s)
    f.close()
    print("Segmentation result is saved in {0}.".format(save_path_hmm))

    # unigram
    unigram = UniGramSeg()
    data_loader = HMMDataLoader(data_path)
    vocab_dict = data_loader.generate_vocab_dict()
    unigram.set_dict(vocab_dict)
    pred_unigram = []
    save_path_unigram = os.path.join('test_result/ML_without_nn/', name + '_unigram_seg.utf8')

    with open(test_path, "r", encoding=encoding) as f:
        lines = f.readlines()
        count = 0
        total_count = len(lines)
        for sent in lines:
            try:
                pred_unigram.append(unigram.cut(sent))
            except:
                pred_unigram.append(["Error"] * len(sent))
                continue
            count += 1
            if count % 500 == 0:
                print("Processed HMM {0}/{1} ---- {2}".format(count, total_count, count / total_count))
    f.close()

    # 保存结果
    with open(save_path_unigram, "w", encoding='utf-8') as f:
        for words in pred_unigram:
            s = " ".join(words)
            s = s.strip() + '\n'
            s.encode('utf-8')
            f.write(s)
    f.close()
    print("Segmentation result is saved in {0}.".format(save_path_unigram))


# jieba分词
def Using_jieba_seg(test_path, name, encoding="utf-8"):
    save_path = os.path.join('test_result/jieba/', name + '_jieba_seg.utf8')
    pred_jieba = []
    with open(test_path, "r", encoding=encoding) as f:
        lines = f.readlines()
        count = 0
        total_count = len(lines)
        for sent in lines:
            try:
                pred_jieba.append(jieba.cut(sent))
            except:
                pred_jieba.append(["Error"] * len(sent))
                continue
            count += 1
            if count % 500 == 0:
                print("Processed FMM {0}/{1} ---- {2}".format(count, total_count, count / total_count))
    f.close()

    # 保存结果
    with open(save_path, "w", encoding='utf-8') as f:
        for words in pred_jieba:
            s = " ".join(words)
            s = s.strip() + '\n'
            s.encode('utf-8')
            f.write(s)
    f.close()
    print("Segmentation result is saved in {0}.".format(save_path))

if __name__ == "__main__":
    name = ('as')
    test_path = os.path.join('icwb2-data/testing/' + name + '_test.utf8')
    vocabulary_path = os.path.join('icwb2-data/gold/' + name + '_training_words.utf8')
    data_path = os.path.join('icwb2-data/training/' + name + '_training.utf8')


    traditional_matching_seg(test_path, name, vocabulary_path, 'utf-8')
    ML_without_nn_seg(test_path, name, data_path, 'utf-8')
    Using_jieba_seg(test_path,name,'utf-8')


