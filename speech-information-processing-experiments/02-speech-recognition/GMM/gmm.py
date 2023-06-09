# HMM Algorithm
# Author: songtongmail@163.com (Tongtong Song (TJU))
# Date: 2021/09/04 16:00
# Last modified: 2021/09/27 17:25
import os
import numpy as np
import scipy.cluster.vq as vq

from utils import Dataloader, get_feats,get_all_feats,extract_feat

import logging
def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
class GMM:
    def __init__(self,K, feats):
        super().__init__()
        self.K = K
        self.D = feats.shape[1]
        self.pi, self.mu, self.sigma = self.kmeans_initial(feats)

    def kmeans_initial(self,feats):
        mu = []
        sigma = []
        (centroids, labels) = vq.kmeans2(feats, self.K, minit="points", iter=100)
        clusters = [[] for i in range(self.K)]
        for (l, d) in zip(labels, feats):
            clusters[l].append(d)

        for cluster in clusters:
            mu.append(np.mean(cluster, axis=0))
            sigma.append(np.cov(cluster, rowvar=False))
        pi = [len(c) * 1.0 / len(feats) for c in clusters]
        return  np.array(pi), np.array(mu), np.array(sigma)

    def gaussian(self, x, mu, sigma):
        """
        :param x: D x 1
        :param mu: D x 1
        :param sigma: D x D
        :return:
        """
        sigma = np.where(sigma == 0, np.finfo(float).eps, sigma)
        det_sigma = np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma)
        x_diff = x-mu # D x 1

        coeff = 1/((2 * np.pi) ** (self.D/2))
        prob = coeff * det_sigma**(-0.5) * np.exp(-0.5*(np.dot(x_diff.T,inv_sigma).dot(x_diff)))
        prob = np.nan_to_num(prob)
        if prob==0:
            prob = np.finfo(float).eps
        return prob

    def log_likelihood(self, feats):
        N = len(feats)
        # E step
        gamma = np.zeros((N, self.K))
        for i in range(N):
            for k in range(self.K):
                gamma[i, k] = self.pi[k] * self.gaussian(feats[i], self.mu[k], self.sigma[k])
        llh = np.mean(np.log10(np.sum(gamma,axis=1)))
        return llh

    def EM(self,feats):
        """
        :param feats: N x D
        :return:
        """
        N = len(feats)
        # E step
        gamma = np.zeros((N, self.K))
        for i in range(N):
            for k in range(self.K):
                gamma[i,k] = self.pi[k]*self.gaussian(feats[i],self.mu[k],self.sigma[k])
            gamma_k_sum = np.sum(gamma[i])
            gamma[i] = gamma[i]/gamma_k_sum

        # M step
        gamma_feats_sum = gamma.sum(0)
        for k in range(self.K):
            # if gamma_feats_sum[k]==0:
            #     continue
            self.pi[k] = gamma_feats_sum[k]/N
            self.mu[k] = np.sum([gamma[n,k] * feats[n] for n in range(N)], axis=0)/gamma_feats_sum[k]
            data_diff = feats - self.mu[k] # N x D
            self.sigma[k] = np.sum( [(gamma[n,k]*data_diff[n].reshape(self.D,1)).dot(data_diff[n].reshape(1,self.D))
                   for n in range(N)],axis=0)/gamma_feats_sum[k]
        return self.log_likelihood(feats)


def train(gmms, class2utt, utt2wav, epochs, class_items):
    for class_ in class_items:
        feats = get_feats(class_, class2utt, utt2wav)
        logger.info('Class:{} Initial llh:{:.6f}'.format(class_, gmms[class_].log_likelihood(feats)))
        for epoch in range(epochs):
            llh = gmms[class_].EM(feats)
            logger.info('Class:{} Train Epoch[{}/{}]:{:.6f}'.format(class_, epoch, epochs, llh))
    return gmms


def test(gmms, class2utt, utt2wav, class_items):
    utt2class = {}
    for key in class2utt.keys():
        values = class2utt[key]
        for value in values:
            utt2class[value] = key
    N = len(utt2class)
    correct = 0
    for idx, utt in enumerate(utt2class.keys()):

        true_class = utt2class[utt]
        feats = extract_feat(utt2wav[utt])
        scores = []
        for class_ in class_items:
            scores.append(gmms[class_].log_likelihood(feats))
        pred_class = class_items[np.argmax(np.array(scores))]
        logger.info('Test[{}/{}]-True:{}'.format(idx, N,true_class))
        logger.info('Test[{}/{}]-Pred:{}'.format(idx, N, pred_class))
        if pred_class == true_class:
            correct += 1
    return correct / N

if __name__ == '__main__':
    result_path = './result'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    logger = init_logger(log_file='result/gmm.log')
    class_items = ['Z', 'O', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    K = 3 # gaussian numbers of one class
    epochs = 3
    feats = get_all_feats('data/train/wav.scp')
    gmms = {}
    class2utt, utt2wav = Dataloader('data/train/wav.scp', 'data/train/text')
    for class_ in class_items:
        gmms[class_] = GMM(K, feats)
    gmms = train(gmms, class2utt, utt2wav, epochs, class_items)
    class2utt, utt2wav = Dataloader('data/test/wav.scp', 'data/test/text')
    acc = test(gmms, class2utt, utt2wav, class_items)
    logger.info('ACC:{:.3f}'.format(acc))


