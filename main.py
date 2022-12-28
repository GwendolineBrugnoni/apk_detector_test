# cree un CSV, le laisser vite , appeller androdet, initialiser colonne androdet,
# ajouter bow cnn et hybrid
# rajouter colonne verité
#tester avec leur dataset
# tester avec mon dataset
import os
import subprocess
import sys

from subprocess import check_output

import pandas as pd
import numpy as np
from tqdm.contrib import logging
# reading the csv file
df = pd.read_csv("androdetPraGuard.csv")
data = np.empty((0,6))

def get_true_csv():
    df = pd.read_csv("androdetPraGuard.csv")
    data = np.empty((0, 2))
    for index, row in df.iterrows():  # index c'est le num de la ligne , row c'est les info de la ligne
        a = get_true_score(row)
        data = np.append(data, [np.append([row[1]], [a])], axis=0)
    score = pd.DataFrame(data=data, columns=['filename'] + ['True_score'])
    score.to_csv("score.csv", index=False)
def get_true_score(p):
    result = [0]
    if (p["trivial"] == 1) | (p["string"] == 1) | (p["reflection"] == 1) | ( p["class"] == 1):
        result[0] = 1
    return result

def get_androdet_score():
    sys.path = "~/Documents/Recherche/ProjetGit/new_androdet"
    cmd = "python new_androdet/androdet.py -f true"
    os.system(cmd)


def get_bow_score():
    sys.path = "~/Documents/Recherche/ProjetGit/bow"
    cmd = "python bow/bow.py  -f true"
    os.system(cmd)

def get_cnn_score():
    pass

def get_hybrid_score():
    pass

def main():
    #recuperation des vrai score et des noms en fusionnant les 4 cathégorie

    get_true_csv()
    get_androdet_score()
    get_bow_score()


    #creation du fichier CSV qui fusionne tous


    df = pd.read_csv("score.csv")
    androdet = pd.read_csv("androdet.csv")
    bow = pd.read_csv("bow.csv")
    score = pd.concat([pd.concat([df, androdet],axis=1),bow],axis=1)
    score.to_csv("final.csv",index=False)

    #test de score csv pour avoir la mesure F1


if __name__ == '__main__':
    main()