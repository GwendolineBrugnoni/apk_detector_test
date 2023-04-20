import optparse
import pandas as pd
from PIL import Image # used for loading images
import tensorflow as tf
import numpy as np
import os # used for navigating to image path
import logging
import glob
import re
import pickle
import string
import math
import numpy as np
from tqdm import tqdm
from keras import models
from keras import layers
from keras import optimizers
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score
import sys
tmp = sys.path
sys.path.append("../")
from common import scores, load_dataset, load_light_dataset,load_new_light_dataset, Timer, get_target, load_dataset_properties
sys.path.append(tmp)


parser = optparse.OptionParser()

parser.add_option('-d', '--dataset-apk-dir',
    action="store", dest="dataset_apk_dir",
    help="Directory of the original apk dataset", default="../dataset/")
parser.add_option('-o', '--dataset-txt-dir',
    action="store", dest="dataset_txt_dir",
    help="Directory of the text (opcodes) dataset created with apk-parser", default="../dataset_txt/")
parser.add_option('-i', '--dataset-img-dir',
    action="store", dest="dataset_img_dir",
    help="Directory of the img dataset created with create_images", default="../dataset_img/")
parser.add_option('-b', '--dataset-bow',
    action="store", dest="dataset_bow",
    help="BOW dataset created with count_words", default="../dataset_tfidf.pv")
parser.add_option('-p', '--dataset-androdet',
    action="store", dest="dataset_androdet",
    help="Androdet IR dataset created with apk-parser", default="../androdetPraGuard.csv")
parser.add_option('-e', '--dataset-entropy',
    action="store", dest="dataset_entropy",
    help="Entropy dataset created with create_entropy_dataset", default="../dataset_entropy.csv")
parser.add_option('-t', '--train',
    action="store", dest="train",
    help="true: force training and overwrite the model. false: the trained model will be used", default="false")
parser.add_option('-f', '--fusion',
                  help="true : create a array list to fusion with the others tests", default="false")
options, args = parser.parse_args()


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)

IMG_SIZE = 2640
NLP_SIZE = 25
ANDRODET_SIZE = 15
images_root_dir = options.dataset_img_dir
levels = 3  # 1=bw 3=rgb
model_name = 'model_trained.k'

#target = 2
network = {'activation': 'tanh', 'learning_rate': 0.001, 'dropout_rate': 0.1, 'optimizer': 'sgd', 'epochs': 5, 'merged_layers': [40]}


def get_original_old_file(img_file):
    return img_file.replace(images_root_dir, options.dataset_txt_dir).replace('.jpeg','')

def get_original_file(img_file):
    return img_file.replace(images_root_dir, '').replace('.jpeg','')

def get_apk_file(img_file):
    return img_file.replace(images_root_dir, options.dataset_apk_dir).replace('.jpeg','.apk')

def create_model(activation, optimizer, learning_rate, output_size, merged_layers):

    original_new_androdet_model = models.load_model("../new_androdet/model_trained.k")
    original_cnn_model = models.load_model("../cnn/model_trained.k")
    original_dnn_model = models.load_model("../bow/model_trained.k")

    new_androdet_model = models.Sequential()
    cnn_model = models.Sequential()
    dnn_model = models.Sequential()

    for layer in original_new_androdet_model.layers[:-1]:
        layer.name = 'new_androdet_' + layer.name
        layer.trainable = False
        new_androdet_model.add(layer)

    for layer in original_cnn_model.layers[:-1]:
        layer.name = 'cnn_' + layer.name
        layer.trainable = False
        cnn_model.add(layer)

    for layer in original_dnn_model.layers[:-1]:
        layer.name = 'dnn_' + layer.name
        layer.trainable = False
        dnn_model.add(layer)

    entropy_input_layer = layers.Input(shape=(1,), name='entropy_input')

    merge_layer = layers.concatenate([cnn_model.layers[-1].get_output_at(-1), dnn_model.layers[-1].get_output_at(-1), entropy_input_layer])

    for (i, n_neurons) in enumerate(merged_layers):
        merge_layer = layers.Dense(n_neurons, activation=activation, name='dense{}'.format(i))(merge_layer)

    output_trivial = layers.concatenate([merge_layer, new_androdet_model.layers[-1].get_output_at(-1)])
    output_trivial = layers.Dense(1, activation='sigmoid')(output_trivial)

    output_rest = layers.Dense(output_size-1, activation='sigmoid')(merge_layer)

    output_all = layers.concatenate([output_trivial, output_rest])

    model = models.Model(inputs=[new_androdet_model.layers[0].get_input_at(-1), cnn_model.layers[0].get_input_at(-1), dnn_model.layers[0].get_input_at(-1), entropy_input_layer], outputs=output_all)

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
    model.summary()

    return model


def fit_one_at_time(model, files, targets, nlp_x, entropy_x, androdet_x, epochs=1):
    tot = 0
    with tqdm(total=len(files) * epochs) as pbar:
        for _ in range(epochs):
            for i, img_file in enumerate(files):
                try:
                    image = Image.open(img_file, 'r')
                    image_X = np.asarray(image).reshape(1, IMG_SIZE, IMG_SIZE, levels).copy()
                    androdet_X = androdet_x[get_apk_file(img_file)].reshape(1, ANDRODET_SIZE)
                    nlp_X = nlp_x[get_original_old_file(img_file) + '.txt'].reshape(1, NLP_SIZE)
                    entropy_X = entropy_x[get_original_file(img_file)].reshape(1, 1)
                    Y = np.array([targets[i]])
                    model.fit(x=[androdet_X, image_X, nlp_X, entropy_X], y=Y, epochs=1, verbose=0)
                except Exception as e:
                    import traceback
                    track = traceback.format_exc()
                    print(track)
                    logging.debug("error: " + img_file)
                    exit()
                pbar.update(1)


def score_one_at_time(model, files, test_Y, nlp_x, entropy_x, androdet_x):
    preds = np.empty((0,test_Y.shape[1]))
    right_test_Y = np.empty((0,test_Y.shape[1]))
    tot = 0

    with tqdm(total=len(files)) as pbar:
        for i, img_file in enumerate(files):
            try:
                image = Image.open(img_file, 'r')
                image_X = np.asarray(image).reshape(1, IMG_SIZE, IMG_SIZE, levels)
                androdet_X = androdet_x[get_apk_file(img_file)].reshape(1, ANDRODET_SIZE)
                nlp_X = nlp_x[get_original_old_file(img_file) + '.txt'].reshape(1, NLP_SIZE)
                entropy_X = entropy_x[get_original_file(img_file)].reshape(1, 1)
                Y = model.predict([androdet_X, image_X, nlp_X, entropy_X], verbose=0)
                preds = np.append(preds, Y, axis=0)
                right_test_Y = np.append(right_test_Y, test_Y[i:i+1], axis=0)
            except:
                logging.debug("error: " + img_file)
            pbar.update(1)

    test_Y = right_test_Y

    print(preds, test_Y)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    print("Other F1 score: ", f1_score(test_Y, preds, average='micro'))

    precision, recall, f1 = scores(preds, test_Y)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("f1score: ", f1)


def process_trainset(train_X, test_X, data_type):
    X = {}
    for x in train_X:
        X[x[0]] = np.array(x[1:],dtype=data_type)
    for x in test_X:
        X[x[0]] = np.array(x[1:],dtype=data_type)
    return X

def process_new_trainset(train_X, data_type):
    X = {}
    for x in train_X:
        X[x[0]] = np.array(x[1:],dtype=data_type)
    return X

def fusion():
    logging.info("PREPARE DATASET")
    df = pd.read_csv(options.dataset_entropy, sep=" ", header=None)
    zero = []
    del df[df.columns[-1]]
    for i in range(np.shape(df)[0]):
        zero.append([0, 0, 0])
    df = pd.concat([df, pd.DataFrame(zero)], axis=1)
    df.to_csv(options.dataset_entropy, sep=" ", header=None, index=False)
    train_X1= load_new_light_dataset(images_root_dir, target=0,training_set_part=1, extension='jpeg')
    train_X2, _, _, _ = load_dataset(options.dataset_bow,target=0,training_set_part=1)
    train_X3, _, _, _ = load_dataset(options.dataset_entropy,target=0,training_set_part=1)
    train_X4, _, _, _ = load_dataset_properties(options.dataset_androdet,target=0,training_set_part=1)
    nlp_X = process_new_trainset(train_X2, 'int')
    entropy_X = process_new_trainset(train_X3, 'float')
    androdet_X = process_new_trainset(train_X4, 'float')
    logging.info("CREATE MODEL")
    model = create_model(
        activation=network['activation'],
        optimizer=network['optimizer'],
        learning_rate=network['learning_rate'],
        output_size=4,
        merged_layers=network['merged_layers']
    )

    model = models.load_model(model_name)
    preds = np.empty((0, 4))

    with tqdm(total=len(train_X1)) as pbar:

        for i, img_file in enumerate(train_X1):
            image = Image.open(img_file, 'r')
            image_X = np.asarray(image).reshape(1, IMG_SIZE, IMG_SIZE, levels)
            print(androdet_X)
            print(get_apk_file(img_file))
            androdet_x = androdet_X[get_apk_file(img_file)].reshape(1, ANDRODET_SIZE)
            entropy_x = entropy_X[get_original_file(img_file)].reshape(1, 1)
            nlp_x = nlp_X[get_original_old_file(img_file) + '.txt'].reshape(1, NLP_SIZE)
            Y = model.predict([androdet_x, image_X, nlp_x, entropy_x], verbose=0)
            preds = np.append(preds, Y, axis=0)
            pbar.update(1)

    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    score = pd.DataFrame(data=preds, columns=['HybridT', 'HybridSE', 'HybridR', 'HybridCE'])
    score.to_csv("../Hybrid.csv", index=False)
    return preds

def main():
    try:
        if options.fusion == 'true':
            data = fusion()
            return data
    except:
        raise Exception('Echec dans la génération du csv')

    t = Timer()
    t.reset_cpu_time()
    logging.info("PREPARE DATASET")
    train_X1, train_Y1, test_X1, test_Y1 = load_light_dataset(images_root_dir, training_set_part=0.8, extension='jpeg')
    train_X2, _, test_X2, _ = load_dataset(options.dataset_bow)
    train_X3, _, test_X3, _ = load_dataset(options.dataset_entropy)
    train_X4, _, test_X4, _ = load_dataset_properties(options.dataset_androdet)
    nlp_X = process_trainset(train_X2, test_X2, 'int')
    entropy_X = process_trainset(train_X3, test_X3, 'float')
    androdet_X = process_trainset(train_X4, test_X4, 'float')
    logging.info("CREATE MODEL")
    model = create_model(
        activation=network['activation'],
        optimizer=network['optimizer'],
        learning_rate=network['learning_rate'],
        output_size=train_Y1.shape[1],
        merged_layers=network['merged_layers']
    )
    t.get_cpu_time("PREPARATION")
    logging.info("TRAIN")
    try:
        if options.train == 'true':
            raise Exception('Force train model')
        model = models.load_model(model_name)
    except:
        fit_one_at_time(model, train_X1, train_Y1, nlp_X, entropy_X, androdet_X, epochs=network['epochs'])
        model.save(model_name)
    t.get_cpu_time("TRAIN")
    logging.info("TEST on TRAIN")
    score_one_at_time(model, train_X1, train_Y1, nlp_X, entropy_X, androdet_X)
    t.get_cpu_time("TEST on TRAIN")
    logging.info("TEST")
    score_one_at_time(model, test_X1, test_Y1, nlp_X, entropy_X, androdet_X)
    t.get_cpu_time("TEST")


if __name__ == '__main__':
    main()
