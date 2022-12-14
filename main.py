# cree un CSV, le laisser vite , appeller androdet, initialiser colonne androdet,
# ajouter bow cnn et hybrid
# rajouter colonne verité
# tester avec leur dataset
# tester avec mon dataset
import optparse
import os
import subprocess
import sys

from subprocess import check_output

import pandas as pd
import numpy as np
from tqdm.contrib import logging
from common import get_target, get_new_target


# reading the csv file
# TODO : Regler ces problème de shell en modifiant les autres fichier pour pouvoir les appeller autrement (là ça donne
#  envie de pleurer !) et en profiter pour regler le pb d'import de common.
def preprocessing(dataset, destination):
    os.system(
        "cd preprocessing && python apk-parser.py -s ../" + dataset + "/ -d ../" + destination + "-parser.csv -t androdet_IR")
    os.system("cd preprocessing && python apk-parser.py -s ../" + dataset + "/ -d ../opcode/")
    os.system(
        "cd preprocessing && python count_words.py -s ../" + destination + "-opcode/ -d " + destination + "-tfidf")
    os.system(
        "cd preprocessing && python create_entropy_dataset.py -s ../" + dataset + "/ -d ../" + destination + "-entropy")
    os.system("cd preprocessing && python create_images.py -s ../" + dataset + "/ -d ../" + destination + "-img/")
# TODO verifier pour les colonne de 0 dans entropy

# TODO : si erreur mettre ligne de 0 et pas rien sinon ça décale tout
def get_true_csv(destination):
    df = pd.read_csv(destination + "-parser.csv")
    data = np.empty((0, 3))
    for index, row in df.iterrows():  # index c'est le num de la ligne , row c'est les info de la ligne
        a = get_new_target(row[1])
        data = np.append(data, [np.append([row[1]], [a])], axis=0)
    score = pd.DataFrame(data=data, columns=['Filename'] + ['Malwaes'] + ['True_score'])
    score.to_csv("score.csv", index=False)

#TODO : c'est quoi cette abomination ?
def get_androdet_score(destination):
    cmd = "cd new_androdet && python androdet.py -d ../" + destination + "-parser.csv -f true"
    os.system(cmd)


def get_bow_score(destination):
    cmd = "cd bow && python bow.py -d ../" + destination + "-tfidf.pv -f true"
    os.system(cmd)


def get_cnn_score(destination):
    cmd = "cd cnn && python cnn.py -d ../" + destination + "-img/ -f true"
    os.system(cmd)


def get_hybrid_score(destination):
    cmd = "cd hybrid && python hybrid.py --dataset-apk-dir  ../" + destination + "/ " \
                                        "--dataset-txt-dir ../" + destination + "-opcode/ " \
                                        "--dataset-img-dir ../" + destination + "-img/ " \
                                        "--dataset-bow ../" + destination + "-tfidf.pv " \
                                        "--dataset-androdet ../" + destination + "-parser.csv " \
                                        "--dataset-entropy ../" + destination + "-entropy.csv " \
                                                                                "-f true"
    os.system(cmd)


def main():
    # recuperation des vrai score et des noms en fusionnant les 4 cathégorie
    parser = optparse.OptionParser()

    parser.add_option('-s', '--source-dataset',
                      action="store", dest="dataset_text_dir",
                      help="Directory of the text dataset created with apk-parser, type opcodes",
                      default="apk")
    parser.add_option('-d', '--text-dataset-dest',
                      action="store", dest="dataset_dest",
                      help="Destination of the text dataset that will be created. Will be the suffix of all files name or preprocessing",
                      default="apk")
    parser.add_option('-p', '--type',
                      action="store", dest="preprocessing",
                      help="true : do the preprocessing", default="false")
    options, args = parser.parse_args()

    if options.preprocessing == 'true':
        preprocessing(options.dataset_text_dir, options.dataset_dest)

    # get_true_csv(options.dataset_dest)
    # get_androdet_score(options.dataset_dest)
    # get_bow_score(options.dataset_dest)
    # get_cnn_score(options.dataset_dest)
    get_hybrid_score(options.dataset_dest)

    # creation du fichier CSV qui fusionne tous

    # df = pd.read_csv("score.csv")
    # androdet = pd.read_csv("androdet.csv")
    # bow = pd.read_csv("bow.csv")
    # score = pd.concat([pd.concat([df, androdet],axis=1),bow],axis=1)
    # score.to_csv("final.csv",index=False)

    # test de score csv pour avoir la mesure F1


if __name__ == '__main__':
    main()
