import keras 
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from .utils import top_10_acc, generate_binary_matrix
from .generate_viable_ids import unique_ids
import os

class DLAnalyzer():
    '''
    class for analyzing with deep learning
    '''
    def __init__(self, model_path='RES-33L-15M-103C-5S-6-48E-53A', 
                 load_model=True):
        self.model_path = model_path
        if load_model:
            keras.metrics.top_10_acc = top_10_acc
            self.model = keras.models.load_model(self.model_path)
            self.model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adam(lr=0.002, beta_1=0.9,
                                                        beta_2=0.999, 
                                                        epsilon=0.01, 
                                                        amsgrad=False),
                        metrics=['accuracy', 
                                keras.metrics.top_k_categorical_accuracy, 
                                top_10_acc]
            )

    def predict(self, X):
        '''
        returns 1-hot vector with class id 
        '''
        P = self.model.predict(X)
        return P[0]

    def get_prediction_binary_matrix(self, prediction, min_confidence=None):
        '''
        returns the prediction as a binary kinetic matrix
        if min_confidence is given a value, it will add up models until
        the sum is greater than min_confidence
        '''
        if min_confidence == None:
            c_id = np.argmax(prediction)
            K_pred = generate_binary_matrix(unique_ids[c_id], 5)
            n = 1

        else:
            K_pred = np.zeros([5,5])
            for i in range(len(prediction)):
                if np.sum(np.sort(prediction)[-i-1:]) > min_confidence:
                    n = i+1
                    break

            for i in range(n):
                i_pred = np.argsort(prediction)[-i-1]
                K = generate_binary_matrix(unique_ids[i_pred], 5)
                for i in range(5):
                    K[i,i] = 0
                K_pred += K*prediction[i_pred]
            
            K_pred /= np.sum(np.sort(prediction)[-n:])

        return K_pred, n
    
    def draw_prediction(self, prediction, path, min_confidence=None, 
                        size=(3,3), show=False):
        '''
        draws the graph for the predicted model
        if min_confidence is given a value, it will add up models until
        the sum is greater than min_confidence
        '''

        K_pred, n = self.get_prediction_binary_matrix(prediction,
                                                min_confidence=min_confidence)

        K_true = np.zeros([5,5])
        K_false = np.zeros([5,5])
        positive_labels = {}
        negative_labels = {}
        for i in range(5):
            for j in range(5):
                if K_pred[i, j] > 0.5:
                    positive_labels[(i,j)] = int(K_pred[i,j]*100)/100
                    K_true[i, j] = 1
                    #negative_labels[(i,j)] = 0
                else:
                    if i>j:
                        negative_labels[(i,j)] = int(K_pred[i,j]*100)/100
                        K_false[i, j] = 1
                        #positive_labels[(i,j)] = 0

        plt.figure(figsize=size)

        G2 = nx.from_numpy_matrix(K_false.transpose(), 
                                  create_using=nx.DiGraph())
        nx.draw_circular(G2, labels={0:'A',1:'B',2:'C',3:'D',4:'E'}, 
                         node_color='blue', edge_color='lightgray', 
                         font_color='white')
        pos = nx.circular_layout(G2)
        nx.draw_networkx_edge_labels(G2,pos,edge_labels=negative_labels, 
                                     font_color='gray')

        G = nx.from_numpy_matrix(K_true.transpose(), create_using=nx.DiGraph())
        nx.draw_circular(G, labels={0:'A',1:'B',2:'C',3:'D',4:'E'}, 
                         node_color='blue', edge_color='black', 
                         font_color='white')
        pos = nx.circular_layout(G)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=positive_labels, 
                                     font_color='black')

        plt.axis('off')
        plt.savefig(path)
        if show:
            plt.show()
        else:
            plt.close()
        return n

    def get_num_decays(self, prediction):
        '''
        returns the number of decays of the predicted model with highest 
        confidence and wether the global analysis needs a constant offset.
        '''
        K, n = self.get_prediction_binary_matrix(prediction)
        num_k = 0
        offset = False
        for i in range(5):
            num_k += K[i,i] < 0
            if np.sum(np.clip(K[i], 0, None)) > 0 and K[i,i] == 0:
                offset = True

        return num_k, offset



