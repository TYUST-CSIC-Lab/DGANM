
###
import os
import sklearn
import sys
from sklearn.metrics import confusion_matrix,roc_curve,auc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, Embedding, Dropout
import random
from keras import backend as K
import copy as cy
import tensorflow as tf
import os


saved_fedset_cifar10 = os.path.join(os.getcwd(), 'cifar10_fedset_accuracy')
saved_fedset_mnist = os.path.join(os.getcwd(), 'mnist_fedset_accuracy')
saved_fedset_mlp = os.path.join(os.getcwd(), 'mlp_fedset_accuracy')

save_flag = True
CNN_data_flag = 'mnist'  # mnist or cifar10
optimize_flag = False


class Constraint(object):

    def __call__(self, w):
        return w

    def get_config(self):
        return {}


#################   参数权重变化    ##############
class MaskWeights(Constraint):

    def __init__(self, mask):
        self.mask = mask
        self.mask = K.cast(self.mask, K.floatx())

    def __call__(self, w):
        w *= self.mask
        return w

    def get_config(self):
        return {'mask': self.mask}


def createWeightsMask(epsilon, noRows, noCols):
    # np.random.seed(1)
    mask_weights = np.random.normal(noRows, noCols, None)
    prob = 1 -(epsilon*(noRows + noCols)) / (noRows * noCols)
    # mask_weights[mask_weights < prob] = 0
    # mask_weights[mask_weights >= prob] = 1
    if mask_weights < prob:
        mask_weights = 0
    else:
        mask_weights = 1
    # noParameters = sum(mask_weights)
    # print("Create Sparse Matrix: No parameters, NoRows, NoCols ", noParameters, noRows, noCols)
    return mask_weights


def rewireMask(weights, noWeights, zeta):
    # remove zeta largest negative and smallest positive weights
    values = np.sort(weights.ravel())
    pos_index = np.where(values > 0)[0]
    neg_index = np.where(values < 0)[0]
    rewiredWeights = cy.deepcopy(weights)

    if len(pos_index) != 0:
        smallestPositive = values[int(pos_index[0] + len(pos_index) * zeta)]
        rewiredWeights[rewiredWeights >= smallestPositive] = 1
    if len(neg_index) != 0:
        largestNegative = values[int(neg_index[0] + len(neg_index) * (1 - zeta))]
        rewiredWeights[rewiredWeights < largestNegative] = 1
    rewiredWeights[rewiredWeights != 1] = 0
    weightMaskCore = cy.deepcopy(rewiredWeights)

    # add zeta random weights
    noRewires = noWeights - np.sum(rewiredWeights)
    if noRewires > 0:
        Rewire_row, Rewire_column = np.where(rewiredWeights == 0)
        index_zeros = np.vstack((Rewire_row, Rewire_column)).T
        nrAdd = np.random.choice(len(index_zeros), int(noRewires), replace=False)
        for i in nrAdd:
            rewiredWeights[index_zeros[i][0], index_zeros[i][1]] = 1
    return rewiredWeights, weightMaskCore


def remove_small_weights(weights, zeta):
    # remove zeta largest negative and smallest positive weights

    values = np.sort(weights.ravel())
    pos_index = np.where(values > 0)[0]
    neg_index = np.where(values < 0)[0]
    rewiredWeights = cy.deepcopy(weights)
    if len(pos_index) != 0:
        smallestPositive = values[int(pos_index[0] + len(pos_index) * zeta)]
        #smallestPositive = np.insert(smallestPositive, 0, [random.uniform(0.01, 0.1)])
        rewiredWeights[rewiredWeights >= smallestPositive] = 1
    if len(neg_index) != 0:
        largestNegative = values[int(neg_index[0] + len(neg_index) * (1 - zeta))]
        #largestNegative = np.insert(largestNegative, -1, [random.uniform(-0.01, -0.1)])
        rewiredWeights[rewiredWeights < largestNegative] = 1
    rewiredWeights[rewiredWeights != 1] = 0
    No_paramters = np.sum(rewiredWeights)
    return rewiredWeights, No_paramters


def flattenCalculation_valid(image_dim, kernel_size, conv_layer):
    return ((image_dim - (kernel_size - 1) * len(conv_layer)) // 2) ** 2 * conv_layer[-1]


def flattenCalculation_same(image_dim, conv_layer):
    return (image_dim // 2) ** 2 * conv_layer[-1]


################  模型建立   ##################################################
class GlobalModel_SET:
    BATCH_SIZE =32

    steps_per_epoch = 1

    def __init__(self):
        self.build_model()  # ??
        # self.current_weights = self.model.get_weights()

    def initialize(self):
        raise NotImplementedError()

    def build_model(self):
        raise NotImplementedError()


################  卷积神经网络建立   #################################################
class GlobalModel_SET_CNN(GlobalModel_SET):

    def __init__(self, model_param):
        self.batch_size = 32
        self.epochs = 1
        self.data = data
        self.num_clients = int(Fed_learning_SET.C * Fed_learning_SET.TOTAL_CLIENTS)
        self.FC_LAYER = model_param['FC_LAYER']
        self.CONV_LAYER = model_param['CONV_LAYER']
        # SPARSE_PARAMETER = model_param['SPARSE_PARAM']
        self.kernel_size = model_param['KERNEL_SIZE']
        self.learning_rate = model_param['LEARNING_RATE']
        self.eps = model_param['sparsity']  # control the sparsity level as discussed in the paper
        self.zeta = model_param['fraction']  # the fraction of the weights removed

        self.nb_classes = 7
        self.input_shape = 300, 300 , 1  # (28 x 28 x 1) / (32 x 32 x 3) tuple
        self.padding = 'same'
        # generate an Erdos Renyi sparse weights mask for each layer
        self.initialize()
        super(GlobalModel_SET_CNN, self).__init__()

    def initialize(self):  # need seed mask??
        self.noParm = 0
        for i in range(len(self.CONV_LAYER)):
            if i == 0:
                self.noParm += (self.kernel_size ** 2 * self.input_shape[1] + 1) * self.CONV_LAYER[i]
            else:
                self.noParm += (self.kernel_size ** 2 * self.CONV_LAYER[i - 1] + 1) * self.CONV_LAYER[i]
        self.conv_noParm = self.noParm

        self.mask = {}

        for l in range(0, len(self.FC_LAYER)):
            if l == 0:
                if self.padding == 'same':
                    self.mask['FC_wm_1'] = createWeightsMask(
                        self.eps, flattenCalculation_same(self.input_shape[0], self.CONV_LAYER),
                        self.FC_LAYER[l]
                    )
                elif self.padding == 'valid':
                    self.mask['FC_wm_1'] = createWeightsMask(
                        self.eps,
                        flattenCalculation_valid(self.input_shape[0], self.kernel_size, self.CONV_LAYER),
                        self.FC_LAYER[l]
                    )
            else:
                self.mask['FC_wm_' + str(l + 1)] = createWeightsMask(self.eps, self.FC_LAYER[l - 1], self.FC_LAYER[l])
            # self.W['CONV_w_' + str(l + 1)] = None
        self.mask['FC_wm_' + str(len(self.FC_LAYER) + 1)] = \
            createWeightsMask(self.eps, self.FC_LAYER[-1], self.nb_classes)

    def build_model(self):
        sgd = keras.optimizers.SGD(lr=self.learning_rate, decay=0.)
        adam = keras.optimizers.Adam(lr=self.learning_rate)
        Rmsprop = keras.optimizers.rmsprop(lr=self.learning_rate, decay=0.)
        self.model = Sequential()
        for l in range(len(self.CONV_LAYER)):
            if l == 0:
                self.model.add(Conv2D(self.CONV_LAYER[l], name='Conv_1',
                                      kernel_size=(self.kernel_size, self.kernel_size),
                                      activation='relu',
                                      padding=self.padding,
                                      input_shape=self.input_shape
                                      ))
            else:
                self.model.add(Conv2D(self.CONV_LAYER[l], name='Conv_' + str(l + 1),
                                      kernel_size=(self.kernel_size, self.kernel_size),
                                      activation='relu',
                                      padding=self.padding))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        for i in range(len(self.FC_LAYER)):
            self.model.add(Dense(self.FC_LAYER[i], name='sparse_' + str(i + 1), activation='relu',
                                 kernel_constraint=MaskWeights(self.mask['FC_wm_' + str(i + 1)])))
            self.model.add(Dropout(0.2))
        self.model.add(Dense(self.nb_classes, name='sparse_' + str(len(self.FC_LAYER) + 1),
                             activation='softmax',
                             kernel_constraint=MaskWeights(self.mask['FC_wm_' + str(len(self.FC_LAYER) + 1)])))

        self.model.compile(optimizer=sgd, loss='binary_crossentropy',
                           metrics=['accuracy'])


    # def weightsEvolution(self):
    #     # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
    #     # sparse fully connected layer
    #     W_extract = self.model.get_weights()
    #     for l in range(1, len(self.FC_LAYER) + 2):
    #
    #
    #         self.mask['FC_wm_'+str(l)], self.Wmcore['wmCore_'+str(l)] = \
    #             rewireMask(W_extract[len(self.CONV_LAYER)*2 + (l-1)*2], self.mask['FC_noPar_'+str(l)], self.zeta)
    #         W_extract[len(self.CONV_LAYER)*2 + (l-1)*2] = \
    #             W_extract[len(self.CONV_LAYER)*2 + (l-1)*2] * self.Wmcore['wmCore_'+str(l)]
    #     # self.build_model()
    #     self.model.set_weights(W_extract)

    def train_and_score(self, x_train_set, y_train_set, local_epochs):
        self.model.fit(x_train_set, y_train_set,
                       batch_size=GlobalModel_SET.BATCH_SIZE,
                       epochs=local_epochs,
                       verbose=1)
        # remove small weights with boundary
        updated_w = self.model.get_weights()
        for i in range(len(self.CONV_LAYER) * 2, len(updated_w), 2):
            mask_removed, noParm_removed = remove_small_weights(updated_w[i], self.zeta)
            updated_w[i] *= mask_removed
            # noWeights, _ = np.where(updated_w[i] != 0)
            # assert len(noWeights) == noParm_removed
            self.noParm += noParm_removed
        self.noParm += sum(self.FC_LAYER) + self.nb_classes
        self.model.set_weights(updated_w)

        # return loss, accuracy

################  联邦学习建立   #################
class Fed_learning_SET(GlobalModel_SET_CNN):
    C = 3
    TOTAL_CLIENTS = 1
    # BATCH_SIZE = 50
    # EPOCHS_PER_ROUND = 5
    MAXIMUM_COMMUNICATION_ROUND = 1

    def __init__(self, model):
        self.num_clients = int(Fed_learning_SET.C * Fed_learning_SET.TOTAL_CLIENTS)
        self.round = Fed_learning_SET.MAXIMUM_COMMUNICATION_ROUND
        # self.round = 1
        # self.original_mask = model.mask  # to ensure 1st round every client has the same model mask !
        self.current_weights = model.model.get_weights()

    # model.model.summary()

    def prepare_client_data(self, client_data):
        raise NotImplementedError()

    def update_weights(self, client_weights, client_sizes, total_size):
        new_weights = [np.zeros(w.shape) for w in self.current_weights]
        for c in range(len(client_weights)):
            for i in range(len(new_weights)):
                new_weights[i] += client_weights[c][i] * client_sizes[c] / total_size
        self.current_weights = new_weights

    def Avg_model(self, client_weights, num_clients):
        new_weights = [np.zeros(w.shape) for w in self.current_weights]
        for c in range(len(client_weights)):
            for i in range(len(new_weights)):
                new_weights[i] += client_weights[c][i] / num_clients
        self.current_weights = new_weights

    def round_communication(self):
        raise NotImplementedError()


#############  卷积神经网络建立   #########
class Fed_learning_SET_CNN(Fed_learning_SET):
    IID = False

    def __init__(self,i, data, model_param):
        # self.client_model = client_model
        self.model_param = model_param
        self.data = data
        self.i = i
        self.nb_classes = self.data.nb_classes
        self.input_dim = self.data.input_shape
        # self.y_test = to_categorical(self.data.y_test, 10)
        self.Global = GlobalModel_SET_CNN(self.model_param)  # build_model??
        super(Fed_learning_SET_CNN, self).__init__(self.Global)  # sparse-connected

    def prepare_client_data(self, client_data):
        if optimize_flag == True:
            self.x_train_set, self.y_train_set = client_data
        else:
            if Fed_learning_SET_CNN.IID == True:
                if CNN_data_flag == 'mnist':
                    self.x_train_set, self.y_train_set = self.data.mnist_iid_shards(
                        self.num_clients)  # shards/labels swap
                elif CNN_data_flag == 'cifar10':
                    self.x_train_set, self.y_train_set = self.data.cifar10_iid_shards(self.num_clients)
            else:
                if CNN_data_flag == 'mnist':
                    self.x_train_set, self.y_train_set = self.data.mnist_iid_shards(self.num_clients)  # mnist_noniid
                elif CNN_data_flag == 'cifar10':
                    self.x_train_set, self.y_train_set = self.data.cifar10_noniid_shards(self.num_clients)

    def round_communication(self):
        # just used to save trained results per round and then plot
        F = []
        f1 = []
        f2 = []
        f3 = []
        f4 = []
        fff = open('C:/Users/HPC/Desktop/LY/morematlab/resultIS.txt', 'a+')
        for r in range(self.round):
            upload_Weights = []
            client_scores = []
            No_parameters = []
            i = self.i

            for client in range(0, self.num_clients):
                clients = client*3000
                self.Global.model.set_weights(self.current_weights)
                self.Global.train_and_score(
                    self.x_train_set[clients:clients*2 +3000],
                    self.y_train_set[clients:clients*2 +3000],

                    local_epochs=15


                )
                score_client = self.Global.model.evaluate(
                    self.x_train_set[0:2000],
                    self.y_train_set[0:2000],
                    verbose=0
                )
                No_parameters.append(self.Global.noParm)
                self.Global.noParm = self.Global.conv_noParm  # reset parameter numbers
                client_scores.append(np.array(score_client))
                # remove_small_weights(self.Global.model.get_weights(), self.Global.zeta)
                upload_Weights.append(self.Global.model.get_weights())
                # masks.append(self.Global.mask)
            # os.system("python Mutiobjective_NSGA2.py")
            clientAVG = sum(
                client_scores[i] for i in range(self.num_clients)) / self.num_clients  # average client scores
            paramAVG = int(
                sum(No_parameters[i] / self.num_clients for i in
                    range(self.num_clients)))  # average parameter numbers
            paramRate = paramAVG / self.Global.model.count_params()
            self.Avg_model(upload_Weights, self.num_clients)
            self.Global.model.set_weights(self.current_weights)
            global_score = self.Global.model.evaluate(
                self.data.x_test_set, self.data.y_test_set,
                verbose=0
            )

            y = self.Global.model.predict(self.data.x_test_set)
            y_pred = []
            for i in range(0, len(y)):
                a = np.argmax(y[i])
                y_pred = np.insert(y_pred, i, a)
            # F1 = f1_score(self.data.y_label, y_pred, average='macro')
            # recall = recall_score(self.data.y_label, y_pred, average='macro')

            fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.data.y_test_set.ravel(), y.ravel())
            auc = sklearn.metrics.auc(fpr, tpr)
            print(auc)
            print('Round {0}: global_loss={1}, global_accuracy={2}, paramAVG={3}, paramRate={4}'.format(r,
                                                                                                        global_score[0],
                                                                                                        global_score[1],
                                                                                                        paramAVG,
                                                                                                        paramRate))
            print('Round {0}:clientAVG_loss={1}, clientAVG_accuracy={2}'.format(r, clientAVG[0], clientAVG[1]))
            f1.append(global_score[1])
            f2.append(paramAVG)
            f3.append(auc)
            f4.append(global_score[0])
        F.append(f1[-1])
        F.append(f2[-1])
        F.append(f3[-1])
        F.append(f4[-1])


        # list2 = [pop[i], p1,p2,p3]
        for num in F:
            fff.write(str(num) + '\t')
        fff.write('\n')
        return F

        # save trained results # just for plot

        # del self.Global.model
        # K.clear_session()
        # tf.reset_default_graph()
        #  return 1 - global_score[1], paramAVG

    ############  主要方法   #########
import pandas as pd
class Mnist_CNN_shards():

    def __init__(self):
        self.data = self.mnist_iid_shards(7)
        self.nb_classes = 7
        self.input_shape = self.data[0][0].shape[0], 300, 300, 1

    def mnist_iid_shards(self, num_clients):
        nb_classes = 7
        data1 = pd.read_csv("C:/Users/HPC/Desktop/Fedlearning/Fed1/label.csv", header=None)
        list1 = data1.values.tolist()
        self.y_test_set = list1[0:2000]
        data2 = pd.read_csv("C:/Users/HPC/Desktop/Fedlearning/shiyan2/train-labelIS", header=None)
        list2 = data2.values.tolist()
        y_train_set = list2[0:32000]
        self.x_test_set = np.load('C:/Users/HPC/Desktop/Fedlearning/Fed1/test.npy')

        x_train_set = np.load('C:/Users/HPC/Desktop/Fedlearning/shiyan2/trainIS.npy')

        self.y_label = self.y_test_set
        y_train_set = keras.utils.np_utils.to_categorical(y_train_set, nb_classes)
        self.y_test_set = keras.utils.np_utils.to_categorical(self.y_test_set, nb_classes)
        return x_train_set, y_train_set


####主函数######
import re






if __name__ == '__main__':
    data = Mnist_CNN_shards()
    i = random.randint(0, 3000)
    model_param = {'CONV_LAYER': [12,12],
                   'FC_LAYER': [55],
                   'KERNEL_SIZE': 3,
                   'sparsity': 50,
                   'LEARNING_RATE': 0.0001,
                   'fraction': 0.3}  # 41, 0.0625
    GlobalModel_SET_CNN(model_param)
    fed = Fed_learning_SET_CNN(i, data, model_param)
    fed.prepare_client_data(None)
    fed.round_communication()

