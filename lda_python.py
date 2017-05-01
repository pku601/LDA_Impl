# -*- encoding:utf-8 -*-

# LDA主题模型之算法实现
# http://skyhigh233.com/blog/2016/10/20/lda-realize/

import random
import time

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import numpy as np

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


# LDA模型
class LDAModel:
    def __init__(self, K, copora, alpha=None, beta=None, iteration=None):
        # K个主题
        self.K = K
        # alpha工程取值一般为0.1
        self.alpha = alpha if alpha else 0.1
        # beta工程取值一般为0.01
        self.beta = beta if beta else 0.01
        # 迭代次数一般取值为1000
        self.iteration = iteration if iteration else 1000

        self.nw = object     # K*V   每个主题下的每个词的出现频次
        self.nd = object     # D*K   每个文档下每个主题的词数
        self.nwsum = object  # K     每个主题的总词数
        self.ndsum = object  # D     每个文档的总词数
        self.theta = object  # doc->topic    D*K
        self.phi = object    # topic->word   K*V
        self.z = object      # D*V  (m,w)对应每篇文档的每个词的具体主题

        # 3个语料整合到一起
        self.corpora = list()
        for theme in copora:
            with open(theme) as fr:
                for line in fr:
                    self.corpora.append(line.strip())

        # 文档数
        self.D = len(self.corpora)
        cut_docs = self.cut(self.corpora)

        # 分词并且id化的文档
        self.word2id, self.id2word, self.id_cut_docs, self.wordnum = self.create_dict(cut_docs)
        self.V = len(self.id2word)

        # 初始化参数
        self.initial(self.id_cut_docs)

        # gibbs采样,进行文本训练
        self.gibbsSamppling()

        # 保存word2id,id_cut_docs,z,theta,phi,以便应用的时候使用
        with open('./result/word2id', 'w') as fw:
            for word, id in self.word2id.iteritems():
                fw.write(word + '\t' + str(id) + '\n')

        with open('./result/id_cut_docs', 'w') as fw:
            for doc in self.id_cut_docs:
                for vocab in doc:
                    fw.write(str(vocab) + '\t')
                fw.write('\n')

        with open('./result/z', 'w') as fw:
            for doc in self.z:
                for vocab in doc:
                    fw.write(str(vocab) + '\t')
                fw.write('\n')

        with open('./result/theta', 'w') as fw:
            for doc in self.theta:
                for topic in doc:
                    fw.write(str(topic) + '\t')
                fw.write('\n')

        with open('./result/phi', 'w') as fw:
            for topic in self.phi:
                for vocab in topic:
                    fw.write(str(vocab) + '\t')
                fw.write('\n')

    # gibbs采样
    def gibbsSamppling(self):
        for iter in range(self.iteration):
            for i, doc in enumerate(self.id_cut_docs):
                for j, word_id in enumerate(doc):
                    theme = self.z[i, j]
                    nd = self.nd[i, theme] - 1
                    nw = self.nw[theme, word_id] - 1
                    ndsum = self.ndsum[i] - 1
                    nwsum = self.nwsum[theme] - 1
                    # 重新给词选择新的主题
                    new_theme = self.reSamppling(nd, nw, ndsum, nwsum)

                    self.nd[i, theme] -= 1
                    self.nw[theme, word_id] -= 1
                    self.nwsum[theme] -= 1

                    self.nd[i, new_theme] += 1
                    self.nw[new_theme, word_id] += 1
                    self.nwsum[new_theme] += 1
                    self.z[i, j] = new_theme

            sys.stdout.write('\rIteration:{0} done!'.format(iter + 1))
            sys.stdout.flush()

            # 模型评估指标,只能作为参考.计算perplexity,比较耗时
            if (iter + 1) % 100 == 0:
                pp = 0.
                for m in range(self.D):
                    for w in range(self.V):
                        pdzzmulpzw = np.sum((self.nd[m, :] / float(np.sum(self.nd[m, :]))).flatten() * (self.nw[:, w] / map(float,np.sum(self.nw, 1))).flatten())
                        pdzzmulpzw = 1. if pdzzmulpzw == 0. else pdzzmulpzw
                        # print pdzzmulpzw
                        pp -= np.log2(pdzzmulpzw)
                        # print pp
                pp /= self.wordnum
                pp = np.exp(pp)

                sys.stdout.write('\rIteration:{0} done!\tPerplexity:{1}'.format(iter + 1, pp))
                sys.stdout.flush()

        # 更新theta和phi
        self.updatePara()

    # 更新theta和phi
    def updatePara(self):
        for d in range(self.D):
            for k in range(self.K):
                self.theta[d, k] = float(self.nd[d, k] + self.alpha) / (self.ndsum[d] + self.alpha*self.K)

        for k in range(self.K):
            for v in range(self.V):
                self.phi[k, v] = float(self.nw[k, v] + self.beta) / (self.nwsum[k] + self.beta*self.K)

    # 重新选择主题
    def reSamppling(self, nd, nw, ndsum, nwsum):
        pk = np.ndarray([self.K])
        for i in range(self.K):
            # gibbs采样公式
            pk[i] = float(nd + self.alpha) * (nw + self.beta) /\
                    ((ndsum + self.alpha*self.K)*(nwsum + self.beta*self.V))
            if i > 0:
                pk[i] += pk[i - 1]

        # 轮盘方式随机选择主题
        u = random.random() * pk[self.K - 1]
        for k in range(len(pk)):
            if pk[k] >= u:
                return k

    # 初始化参数
    def initial(self, id_cut_docs):
        self.nd = np.array(np.zeros([self.D, self.K]), dtype=np.int32)
        self.nw = np.array(np.zeros([self.K, self.V]), dtype=np.int32)
        self.ndsum = np.array(np.zeros([self.D]), dtype=np.int32)
        self.nwsum = np.array(np.zeros([self.K]), dtype=np.int32)
        self.z = np.array(np.zeros([self.D, self.V]), dtype=np.int32)
        self.theta = np.ndarray([self.D, self.K])
        self.phi = np.ndarray([self.K, self.V])
        # 给每篇文档的每个词随机分配主题
        for i, doc in enumerate(id_cut_docs):
            for j, word_id in enumerate(doc):
                theme = random.randint(0, self.K - 1)
                self.z[i, j] = theme
                self.nd[i, theme] += 1
                self.nw[theme, word_id] += 1
                self.ndsum[i] += 1
                self.nwsum[theme] += 1

    # 文档分词,去无用词
    # 可以考虑去除文本低频词
    def cut(self, docs):

        cut_docs = list()
        tokenizer = RegexpTokenizer(r'\w+')
        en_stop = set(stopwords.words('english'))
        p_stemmer = PorterStemmer()

        for doc in docs:
            # clean and tokenize document string
            raw = doc.lower()
            tokens = tokenizer.tokenize(raw)

            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if i not in en_stop]

            # stem token
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

            cut_docs.append(stemmed_tokens)

        return cut_docs

    # 创建word2id,id2word和document字典
    def create_dict(self, cut_docs):
        word2id = dict()
        word_num = 0
        for i, doc in enumerate(cut_docs):
            for j, word in enumerate(doc):
                word_num += 1
                if word not in word2id:
                    word2id[word] = len(word2id)
                cut_docs[i][j] = word2id[word]
        return word2id, dict(zip(word2id.values(), word2id.keys())), cut_docs, word_num

    # 返回各个主题的top词汇
    def getTopWords(self, top_num=20):
        with open('./result/topwords', 'w') as fw:
            for k in range(self.K):
                top_words = np.argsort(-self.phi[k, :])[:top_num]
                top_words = [self.id2word[word] for word in top_words]
                top_words = '\t'.join(top_words)
                res = 'topic{0}\t{1}'.format(k, top_words)
                fw.write(res + '\n')
                print res

    # 返回文档前几个topic中的前几个词
    def getTopTopics(self, top_topic=5, top_word=5):
        with open('./data/result/toptopics', 'w') as fw:
            for d in range(self.D):
                top_topics = np.argsort(-self.theta[d, :])[:top_topic]
                print 'document{0}:'.format(d)
                for topic in top_topics:
                    top_words_id = np.argsort(-self.phi[topic, :])[:top_word]
                    top_words = [self.id2word[word] for word in top_words_id]
                    top_words = '\t'.join(top_words)
                    res = 'topic{0}\t{1}'.format(topic, top_words)
                    fw.write(res + '\n')
                    print res
                fw.write('\n')

if __name__ == '__main__':
    print '====== begin ===='

    corpus = ['./data/theme_three.txt']
    time1 = time.time()
    lda_model = LDAModel(20, corpus, iteration=300)
    time2 = time.time()
    print ' Training time: {0}'.format(time2-time1)

    # 各个主题的top20词汇
    lda_model.getTopWords(20)

    # 各个文档的top5话题,每个话题的top5词汇
    print '每篇文档的top topics中的top words'
    lda_model.getTopTopics()
