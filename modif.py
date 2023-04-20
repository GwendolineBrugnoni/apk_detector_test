import pandas as pd
import numpy as np
from androguard.core.bytecodes.apk import APK
from androguard.core.bytecodes.dvm import DalvikVMFormat
from common import scores, load_dataset

# pred, _, _, _ = load_dataset('apk-clean-entropy.csv',target=0,training_set_part=1)
# print(pred[0])

# predu = pd.read_csv('apk-clean-tfidf.pv', sep=" ", header=None)
# print(predu.iloc[1,].values)
# pred, _, _, _ = load_dataset('dataset_tfidf.pv',target=0,training_set_part=1)
# print(pred[0])
# def count_indentifier(identifier):
#     length = len(identifier)
#     if length == 1:
#         print(identifier)
#     elif length == 2:
#         print(identifier)
#     elif length == 3:
#         print(identifier)
# #
# data = pd.read_csv("androdetPraGuard.csv")
# i=0
# while i < data.shape[0]:
#     print(data['filename'])
#     if (data['Num_Cls_L1'][i] != 0 and data['Num_Cls_L2'][i] != 0 and data['Num_Cls_L3'][i] != 0):
#         print('oui')
#     if (data['Num_Mtds_L1'][i] != 0 and data['Num_Mtds_L2'][i] != 0 and data['Num_Mtds_L3'][i] != 0):
#         print('ah')
#     if (data['Num_Flds_L1'][i] != 0 and data['Num_Flds_L2'][i] != 0 and data['Num_Flds_L3'][i] != 0):
#         print('non')
#     i += 1

# a = APK('dataset/benines/non_obfusqués/01bc3e19315eadfaf8f2f78bf8c188523697241e58e1124c6c2cc301b1e7b05e.apk') #bow
# a = APK('dataset/benines/non_obfusqués/048db14d6e71d7b3a2a41689f0dc6f905950f1ac15a9bde8d549a63abb2f2804.apk')
#
# d = DalvikVMFormat(a) #echec de cette ligne sur certaine apk
# for c in d.get_classes():
#     for m in c.get_methods():
#         print("m.get_name : " +m.get_name())
#         count_indentifier(m.get_name())
#     for f in c.get_fields():
#         print("f.get_name : " + f.get_name())
#         count_indentifier(f.get_name())
#     print("c.get_name : "+ c.get_name())


'''
La;
Lb;
Lc;
Ld;
Le;
Lf;
Lg;
Lh;
Li;
Lj;


c.get_name : Ll/a/l/a/a;
c.get_name : Ll/a/o/h$a;
c.get_name : Ll/b/a/b/b$a;
c.get_name : Ll/b/a/b/b$b;
c.get_name : Ll/c/b/b$b;
c.get_name : Ll/f/b/d$b;
c.get_name : Ll/f/b/h;
c.get_name : Ll/f/b/k/a;
c.get_name : Ll/f/b/k/f;
c.get_name : Ll/f/b/k/k;
c.get_name : Ll/f/b/k/m/c;
'''
# def zero(fichier):
#     df = pd.read_csv(fichier, sep=" ", header=None)
#     zero = []
#     del df[df.columns[-1]]
#     for i in range(np.shape(df)[0]):
#         print(i)
#         zero.append([0,0,0,])
#     df = pd.concat([df,pd.DataFrame(zero)],axis=1)
#     df.to_csv('zero.csv', sep = " ",header=None,index=False)
#
# # zero('test-entropy.csv')
# df = pd.read_csv('reflexion-entropy.csv', sep=" ", header=None)
# del df[df.columns[-1]]
# del df[df.columns[-1]]
# df.to_csv('reflexion-entropy.csv', sep = " ",header=None,index=False)

df = pd.read_csv("tdif.pv",sep=' ',header=None)
print(df[1:2])
# print(df[26])
# print(df[27])
# print(df[28])
del(df[28])
del(df[29])
df.to_csv("test_dataset_tfidf.pv",index=False,sep=" ")