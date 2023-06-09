# HMM Algorithm
# Author: songtongmail@163.com (Tongtong Song (TJU))
# Date: 2021/8/28 16:00
# Last modified: 2021/9/5 15:50
import os
import numpy as np
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
class HMM:
    def __init__(self, pi, A, B):
        super().__init__()
        self.pi = pi
        self.A = A
        self.B = B
        self.s_set_num = len(self.pi)
        self.o_set_num = len(self.B[0])

    def forward(self,O):
        """
        get P(O|M) by forward algorithm
            \alpha_{t+1}(j) =b_{j}(o_{t+1})\sum_{i=1}^M(\alpha_{t}(i)a_{ij})
        :param O:
        :return:
        """
        #  definition forward probability matrix
        self.o_len =len(O)
        self.forward_prob_matrix = np.zeros((self.o_len,self.s_set_num))

        # t=0
        self.forward_prob_matrix[0,:] = self.pi * self.B[:, O[0]]

        # T=1:T, compute forward probability matrix
        for t in range(1,self.o_len):
            for k in range(self.s_set_num):
                # compute the current total transition probability for every last state to the current state
                self.forward_prob_matrix[t,k] = np.dot(self.forward_prob_matrix[t-1,:], self.A[:,k]) * self.B[k,O[t]]

        # get the last time total forward probability
        forward_prob = np.sum(self.forward_prob_matrix[self.o_len-1,:])

        return forward_prob

    def backward(self,O):
        """
        get P(O|M) by backward algorithm
            \beta_{t}(i) = \sum_{j=1}^M(\beta_{t+1}(j)a_{ij}b_{j}(o_{t+1}))
        :param O:
        :return:

        """
        # definition backward probability matrix
        self.o_len = len(O)
        self.backward_prob_matrix = np.zeros((self.o_len, self.s_set_num))

        # t = T-1
        self.backward_prob_matrix[-1,:] = 1

        # t=T-2:-1, compute backward probability matrix
        for t in reversed(range(self.o_len-1)):
            for k in range(self.s_set_num):
                self.backward_prob_matrix[t,k] = np.sum(self.backward_prob_matrix[t+1,:]*self.A[k,:]*self.B[:,O[t+1]])

        # get the first time total backward probability
        backward_prob = np.sum(self.pi*self.B[:,O[0]]*self.backward_prob_matrix[0])

        return backward_prob

    def get_state_probability(self, t, q):
        """get the probability when time is t and state is q
            \alpha_{t}(q)*\beta_{t}(q)/\sum(\alpha_{t}\beta_{t})
        :param forward_prob_matrix: forward probability matrix
        :param backward_prob_matrix: backward probability matrix
        :param t: time
        :param q: state
        :return:
            prob: the probability
        """
        assert 0<=t<=self.o_len, 't must be in {}-{}'.format(0,self.o_len)
        assert 0<=q<=self.s_set_num, 'q must be in {}-{}'.format(0,self.s_set_num)
        q_prob = self.forward_prob_matrix[t][q]*self.backward_prob_matrix[t][q]
        total_prob = np.sum(self.forward_prob_matrix[t]*self.backward_prob_matrix[t])
        prob = q_prob/total_prob
        return prob

    def decoding(self,O):
        """ Viterbi Decoding
        get argmax P(Q|O,M)
            \delta_{t+1}(j) = max\limits_{1<=i<=M}(\delta_{t}(i)a_{ij}))b_{j}(o_{t+1}
        :return:
            best_path:
            best_prob:
        """

        #  definition the best probability matrix and last node maxtrix
        o_len = len(O)
        best_prob_matrix = np.zeros((o_len,self.s_set_num))
        last_node_matrix = np.zeros((o_len,self.s_set_num),dtype=int)

        # t=0
        best_prob_matrix[0,:] = self.pi * self.B[:,O[0]]
        last_node_matrix[0,:] = -1

        # T=1:T, compute best probability matrix and last node matrix
        for t in range(1, o_len):
            for j in range(self.s_set_num):
                current_prob = best_prob_matrix[t-1,:]*self.A[:,j]
                best_s = np.argmax(current_prob)
                best_prob_matrix[t,j] = current_prob[best_s]*self.B[j,O[t]]
                last_node_matrix[t,j] = best_s

        # t=T-1, get best_prob and the last node
        last_node =  np.argmax(best_prob_matrix[o_len-1,:])
        best_prob = best_prob_matrix[o_len - 1,last_node]
        best_path = [last_node]

        # t=T-1:0,backtrack the path
        for t in range(o_len-1,0,-1):
            last_node = last_node_matrix[t, last_node]
            best_path.append(last_node)

        # reverse to the noraml path
        best_path = best_path[::-1]

        return best_path, best_prob

if __name__ == '__main__':

    result_path = './result'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    logger = init_logger(log_file='result/hmm.log')
    pi = np.array([0.2, 0.4, 0.4])
    A = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])

    # observation sequence
    O = [0, 1, 0]

    hmm_model = HMM(pi, A, B)
    forward_prob = hmm_model.forward(O)
    logger.info('forward_prob:{:.6f}'.format(forward_prob))
    backward_prob = hmm_model.backward(O)
    logger.info('backward_prob:{:.6f}'.format(backward_prob))
    prob = hmm_model.get_state_probability(2, 2)
    logger.info('state(2,2) prob:{:.6f}'.format(prob))
    best_path, best_prob = hmm_model.decoding(O)
    logger.info('best_path:{},best_prob:{:.6f}'.format(best_path, best_prob))

