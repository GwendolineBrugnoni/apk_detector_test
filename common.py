import pandas as pd
import numpy as np
import glob
import pickle
from tqdm import tqdm
import time
import os

def yo():
    print('yo')

def scores(preds, test_Y):
    if len(test_Y.shape) == 1:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for pred, test in zip(preds, test_Y):
            if pred == 1:
                if test == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if test == 0:
                    tn += 1
                else:
                    fn += 1
        precision = tp/(tp+fp) if tp+fp != 0 else 0
        recall = tp/(tp+fn) if tp+fn != 0 else 0
        f1_score = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0
        return precision, recall, f1_score
    else:
        print(test_Y.shape)
        num_classes = test_Y.shape[1]
        samples = [{'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for _ in range(num_classes)]
        for pred, test in zip(preds, test_Y):
            for value, true_value, sample in zip(pred, test, samples):
                if value == 1:
                    if true_value == 1:
                        sample['tp'] += 1
                    else:
                        sample['fp'] += 1
                else:
                    if true_value == 0:
                        sample['tn'] += 1
                    else:
                        sample['fn'] += 1
        precisions = list(map(lambda x: x['tp']/(x['tp']+x['fp']) if x['tp']+x['fp'] != 0 else 0, samples))
        recalls = list(map(lambda x: x['tp']/(x['tp']+x['fn']) if x['tp']+x['fn'] != 0 else 0, samples))
        f1_scores = list(map(lambda x: 2*x[0]*x[1]/(x[0]+x[1]) if x[0]+x[1] != 0 else 0, zip(precisions, recalls)))
        (list(map(lambda x: print( "tp : ",x['tp']," tn : ",x['tn']," fp : ",x['fp']," fn : ",x['fn']) ,samples)))

        # print("tp : ",sample['tp']," tn : ",sample['tn']," fp : ",sample['fp']," fn : ",sample['fn'])
        return precisions, recalls, f1_scores

def load_dataset_original(input_file, target = None, training_set_part = 0.8):
    df = pd.read_csv(input_file, sep=" ", header=None)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows
    X = df.iloc[:,0:-5].values
    if target == None:
        Y = df.iloc[:,-5:-1].values
    else:
        Y = df.iloc[:,target-5:target-4].values

    total_items = X.shape[0]
    input_size = X.shape[1]
    output_size = Y.shape[1]
    training_set_size = int(round(total_items * training_set_part))

    train_X = X[:training_set_size]
    train_Y = Y[:training_set_size]
    test_X = X[training_set_size:]
    test_Y = Y[training_set_size:]

    return train_X, train_Y, test_X,test_Y
# target: [0=TRIVIAL,1=STRING,2=REFLECTION,3=CLASS, None=ALL]
def load_dataset(input_file, target = None, training_set_part = 0.8):
    df = pd.read_csv(input_file, sep=" ", header=None)

    # df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows mis en commentaire pour pouvoir fusionner dans le bon ordre
    X = df.iloc[:,0:-3].values #on a modifié le -5 en -3
    # TODO : ce truc est debile ....
    print("targer = ", target)
    if target == None:
        Y = df.iloc[:,-3:-1].values
    else:
        Y = df.iloc[:,target-5:target-4].values #pas utiliser je crois

    total_items = X.shape[0]
    input_size = X.shape[1]
    output_size = Y.shape[1]
    training_set_size = int(round(total_items * training_set_part))

    train_X = X[:training_set_size]
    train_Y = Y[:training_set_size]
    test_X = X[training_set_size:]
    test_Y = Y[training_set_size:]

    return train_X, train_Y, test_X, test_Y



# target: [0=TRIVIAL,1=STRING,2=REFLECTION,3=CLASS]
def load_dataset_properties_original(input_file, target = 3, training_set_part = 0.8):
    df = pd.read_csv(input_file)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows
    X = df.iloc[:,1:-4].values
    Y = df.iloc[:,target-4:target-3].values

    total_items = X.shape[0]
    input_size = X.shape[1]
    output_size = Y.shape[1]
    training_set_size = int(round(total_items * training_set_part))

    train_X = X[:training_set_size]
    train_Y = Y[:training_set_size]
    test_X = X[training_set_size:]
    test_Y = Y[training_set_size:]

    return train_X, train_Y, test_X, test_Y
'''
A noter que le code comme modifier actuellement ne pourra plus générer de modele
'''
# target: [0=TRIVIAL,1=STRING,2=REFLECTION,3=CLASS]
def load_dataset_properties(input_file, target = 3, training_set_part = 0.8):
    df = pd.read_csv(input_file,sep=',')
    # df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows mis en commentaire pour pouvoir faire la fusion
    print(df.head(10))
    X = df.iloc[:,1:-2].values
    # print(X.head(10))
    Y = df.iloc[:,target-2:target-1].values # on enleve le nombre de colonnes de resultat ici deux avec malwares et obfusqués

    total_items = X.shape[0]
    input_size = X.shape[1]
    output_size = Y.shape[1]
    training_set_size = int(round(total_items * training_set_part))

    train_X = X[:training_set_size]
    train_Y = Y[:training_set_size]
    test_X = X[training_set_size:]
    test_Y = Y[training_set_size:]

    return train_X, train_Y, test_X, test_Y

def load_new_light_dataset(data_folder_name, target=None, training_set_part = 0.8, extension="txt"):
    '''
       get the dataset as [filename] [target]
       '''
    data = np.empty((0, 1))
    total = 0
    for _ in glob.iglob(data_folder_name + '**/*.' + extension, recursive=True):
        total += 1

    with tqdm(total=total) as pbar:
        for file in sorted(glob.iglob(data_folder_name + '**/*.' + extension, recursive=True)):
            data = np.append(data, np.array([file]))
            pbar.update(1)

    pickle.dump(data, open(os.path.join(data_folder_name, "filenames.p"), "wb"))
    X = data.reshape(-1)

    return X

def load_light_dataset(data_folder_name, target=None, training_set_part = 0.8, extension="txt"):
    '''
    get the dataset as [filename] [target]
    '''
    try:
        # data = pickle.load(open(os.path.join(data_folder_name, "filenames.p"), "rb" ) )
        raise Exception('test')
    except:
        data = np.empty((0,5))

        total = 0
        for _ in glob.iglob(data_folder_name + '**/*.' + extension, recursive=True):
            total += 1

        with tqdm(total=total) as pbar:
            for file in glob.iglob(data_folder_name + '**/*.' + extension, recursive=True):
                X = np.array([file])
                print(X)
                Y = get_target(file)
                print(Y)
                data = np.append(data, np.array([np.append(X,Y)]), axis=0)
                pbar.update(1)

        pickle.dump(data, open(os.path.join(data_folder_name, "filenames.p"), "wb" ) )

    np.random.shuffle(data)
    X = data[:,0]
    if target == None:
        Y = data[:,1:]
    else:
        Y = data[:,1 + target : 2 + target]
    Y = Y.astype(int)
    total_items = X.shape[0]
    training_set_size = int(round(total_items * training_set_part))
    X = X.reshape(-1)

    train_X = X[:training_set_size]
    train_Y = Y[:training_set_size]
    test_X = X[training_set_size:]
    test_Y = Y[training_set_size:]

    return train_X, train_Y, test_X, test_Y


def get_target(p):
    result = [0,0,0,0]
    if 'TRIVIAL' in p:
        result[0] = 1
    if 'STRING_ENCRY' in p:
        result[1] = 1
    if 'REFLECTION' in p:
        result[2] = 1
    if 'CLASS_ENCRYPTION' in p:
        result[3] = 1
    return result

def get_new_target(p):
    result = [0, 1]
    if 'malwares' in p:
        result[0] = 1
    if 'non_obfusqués' in p:
        result[1] = 0
    return result

class Timer:
    last_timestamp = None

    def reset_cpu_time(self):
        self.last_timestamp = time.clock()

    def get_cpu_time(self, task_name=''):
        interval = time.clock() - self.last_timestamp
        print('>'*30 + ' ' + task_name + ' time: ' + str(interval) + ' seconds')
        self.last_timestamp = time.clock()
