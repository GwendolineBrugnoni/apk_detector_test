import numpy as np
import pandas as pd
from keras import models
from keras import layers
from keras import losses
from keras import optimizers
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score
import logging
import sys
from pprint import pprint
import optparse

tmp = sys.path
sys.path.append("../")

from common import scores, load_dataset_properties,load_dataset_properties_original, Timer
sys.path.append(tmp)


parser = optparse.OptionParser()

parser.add_option('-d', '--dataset-dir',
    action="store", dest="dataset_dir",
    help="Directory of the text dataset created with count_words", default="../androdetPraGuard.csv")
parser.add_option('-t', '--train',
    action="store", dest="train",
    help="true: force training and overwrite the model. false: the trained model will be used", default="false")
parser.add_option('-f', '--fusion',
                  help="true : create a array list to fusion with the others tests",default ="false" )

options, args = parser.parse_args()


np.set_printoptions(threshold=sys.maxsize)
logging.basicConfig(level=logging.DEBUG)


network = {'n_layers': 3, 'n_neurons': 50, 'activation': 'tanh', 'learning_rate': 0.001, 'dropout_rate': 0.2, 'optimizer': 'adam', 'epochs': 50}


dataset = options.dataset_dir
model_name = 'model_trained.k'


def create_model(input_size, output_size, n_layers, n_neurons, activation_function, learning_rate, dropout_rate, optimizer):
    model = models.Sequential()
    model.add(layers.Dense(n_neurons, input_shape=(input_size, ), name='new_androdet_dense_1'))
    for _ in range(n_layers):
        if dropout_rate != 0.0:
            model.add(layers.Dropout(dropout_rate, noise_shape=None, seed=None))
        model.add(layers.Dense(n_neurons, activation = activation_function))
    model.add(layers.Dense(output_size, activation = "sigmoid"))
    #model.summary()
    if optimizer == 'rmsprop':
        opt = optimizers.rmsprop(lr=learning_rate)
    elif optimizer == 'adam':
        opt = optimizers.adam(lr=learning_rate)
    elif optimizer == 'sgd':
        opt = optimizers.sgd(lr=learning_rate)
    elif optimizer == 'adagrad':
        opt = optimizers.adagrad(lr=learning_rate)
    elif optimizer == 'adadelta':
        opt = optimizers.adadelta(lr=learning_rate)
    elif optimizer == 'adamax':
        opt = optimizers.adamax(lr=learning_rate)
    elif optimizer == 'nadam':
        opt = optimizers.nadam(lr=learning_rate)
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics = ["mean_squared_error"]
    )
    return model


def predict_and_score(model, test_X, test_Y):
    preds = model.predict(test_X)

    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    print(len(preds), len(preds[preds == 0]), len(test_Y), len(test_Y[test_Y == 0]))

    precision, recall, f1 = scores(preds, test_Y)
    print("Test-precision: ", precision)
    print("Test-recall: ", recall)
    print("Test-f1score: ", f1)

    test_Y = test_Y.reshape(-1)
    preds = preds.reshape(-1)


def main():
    t = Timer()

    logging.info("PREPARE DATASET")

    # Create train and test set
    train_X, train_Y, test_X, test_Y = load_dataset_properties_original(dataset, target=0)
#[ 0.23172642  7.         22.          3.          4.91891892  0.17340134
#  7.         11.          0.         12.54545455  0.43449413  0.
 # 0.          0.        ]

#[1.8666666666666667 0.36666666666666653 4.0 26.0 0.0 3.8666666666666663
# 0.10666666666666667 4.0 16.0 0.0 6.0 0.6 0.0 0.0 0.0]


    # train_names = train_X[:,0]
    train_X = train_X[:,1:]
    print(train_X[0])

    # test_names = test_X[:,0]
    test_X = test_X[:,1:]

    # train_size = train_X.shape[0]
    # test_size = test_X.shape[0]
    input_size = train_X.shape[1]
    output_size = train_Y.shape[1]
    try:
        if options.fusion == 'true':
            model = models.load_model(model_name)
            X, Y, Z, A = load_dataset_properties(dataset, target=0,training_set_part=1)
            print(X[0])
            X = X[:, 1:]
            print(X[0])
            data = model.predict(X)
            data[data >= 0.5] = 1
            data[data < 0.5] = 0
            score = pd.DataFrame(data=data, columns=['Androdet'])
            score.to_csv("../androdet.csv", index=False)
            return data
    except:
        raise Exception('Echec dans la génération du csv')

    try:
        if options.train == 'true':
            raise Exception('Force train model')
        model = models.load_model(model_name)
    except:
        # create the model
        logging.info("CREATE MODEL")
        model = create_model(
            input_size=input_size,
            output_size=output_size,
            n_layers=network['n_layers'],
            n_neurons=network['n_neurons'],
            activation_function=network['activation'],
            learning_rate=network['learning_rate'],
            dropout_rate=network['dropout_rate'],
            optimizer=network['optimizer']
        )

        t.reset_cpu_time()

        # train the model
        results = model.fit(
         x=train_X,
         y=train_Y,
         epochs= network['epochs'],
        )

        t.get_cpu_time("TRAIN")

        model.save(model_name)

    logging.info("TEST on TRAIN")
    t.reset_cpu_time()
    predict_and_score(model, np.asarray(train_X).astype(np.float32), train_Y)
    t.get_cpu_time("TEST on TRAIN")
    logging.info("TEST")
    predict_and_score(model, np.asarray(test_X).astype(np.float32), test_Y)
    t.get_cpu_time("TEST")


if __name__ == '__main__':
    main()
