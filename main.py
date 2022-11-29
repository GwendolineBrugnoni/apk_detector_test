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

def get_true_score(p):
    result = [0]
    if (p["trivial"] == 1) | (p["string"] == 1) | (p["reflection"] == 1) | ( p["class"] == 1):
        result[0] = 1
    return result

def get_androdet_score():
    sys.path = "~/Documents/Recherche/ProjetGit/new_androdet"
    cmd = "python new_androdet/androdet.py -f true"
    data = os.system(cmd)
    print(data)


def get_bow_score():
    pass

def get_cnn_score():
    pass

def get_hybrid_score():
    pass

def main():
    #recuperation des vrai score et des noms en fusionnant les 4 cathégorie
    df = pd.read_csv("androdetPraGuard.csv")
    data = np.empty((0,2))
    for index, row in df.iterrows():#index c'est le num de la ligne , row c'est les info de la ligne
        a = get_true_score(row)
        data = np.append(data,[np.append([row[1]],[a])],axis=0)
    score = pd.DataFrame(data=data, columns=['filename']+['True_score'])
    score.to_csv("score.csv", index=False)

    get_androdet_score()



    #creation du fichier CSV qui fusionne tous


    df1 = pd.read_csv("score.csv")
    df2 = pd.read_csv("androdet.csv")

    score = pd.concat([df1, df2],axis=1)
    score.to_csv("final.csv",index=False)
    #test de score csv pour avoir la mesure F1


if __name__ == '__main__':
    main()