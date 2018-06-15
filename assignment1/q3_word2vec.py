# -*- coding: UTF-8 -*-
#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    x = x / np.sqrt(np.sum(x ** 2, 1)).reshape(-1, 1) # 对矩阵x的每一行正规化
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    # 计算cost
    softmaxVec = softmax(np.dot(outputVectors, predicted.reshape(-1, 1)).reshape(1, -1)[0]) # 得到一个与词表等长的数组，代表了预测为词表中每一个词的概率
    cost = -np.log(softmaxVec[target])

    # 计算gradPred，根据作业中对3（a）的解答
    gradPred = np.sum(outputVectors * softmaxVec.reshape(-1, 1), 0) - outputVectors[target]

    # 计算grad，根据作业中对3（b）的解答
    oneHotVec = np.zeros(outputVectors.shape[0])
    oneHotVec[target] = 1
    grad = np.dot((softmaxVec - oneHotVec).reshape(-1, 1), predicted.reshape(1, -1))
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    # 1、计算cost，根据作业3（c）里面的公式（6）
    u_target = outputVectors[target]
    U_samples = np.zeros((len(indices) - 1, outputVectors.shape[1]))
    for i in xrange(len(indices) - 1):
        U_samples[i] = outputVectors[indices[i + 1]]
    cost = -np.log(sigmoid(np.vdot(u_target, predicted))) - np.sum(np.log(sigmoid(np.dot(-U_samples, predicted.reshape(-1, 1)))))

    # 2、计算gradPred，根据对作业3（c）的解答
    c = (1 - sigmoid(np.dot(-U_samples, predicted.reshape(-1, 1)))) * U_samples
    gradPred = (sigmoid(np.vdot(u_target, predicted)) - 1) * u_target + np.sum((1 - sigmoid(np.dot(-U_samples, predicted.reshape(-1, 1)))) * U_samples, 0)

    # 3、计算grad，根据对作业3（c）的解答
    # 3.1、求负样本对应词向量的梯度
    gradWeight = 1 - sigmoid(np.dot(-U_samples, predicted.reshape(-1, 1)))
    grad = np.zeros(outputVectors.shape)
    for i in xrange(len(indices) - 1):
        index = indices[i + 1]
        grad[index] = grad[index] + gradWeight[i][0] * predicted
    # 3.2、求目标样本对应词向量的梯度
    grad[target] = grad[target] + (sigmoid(np.vdot(u_target, predicted)) - 1) * predicted
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    # 求predicted，这个比较容易。也不用先把输入词转为one-hot向量，然后再用矩阵乘法。直接在inputVectors里面查找就行了。
    currentWordIndex = tokens.get(currentWord)
    predicted = inputVectors[currentWordIndex]

    for i in xrange(2*C):
        if i < contextWords.__len__():
            contextWord = contextWords[i]
            target = tokens.get(contextWord)
            costTemp, gradPred, grad = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
            cost = cost + costTemp
            gradIn[currentWordIndex] = gradIn[currentWordIndex] + gradPred
            gradOut = gradOut + grad
        else:
            break
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    print "==== 梯度检查想运行通过，需要在word2vec_sgd_wrapper开头增加random.seed(31415)、np.random.seed(9265)两行代码 ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    # 先不实现CBOW
    # print "\n==== Gradient check for CBOW      ===="
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #     cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
    #     dummy_vectors)
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #     cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
    #     dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    # print cbow("a", 2, ["a", "b", "c", "a"],
    #     dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    # print cbow("a", 2, ["a", "b", "a", "c"],
    #     dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
    #     negSamplingCostAndGradient)


if __name__ == "__main__":
    # test_normalize_rows()
    test_word2vec()
