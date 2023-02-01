import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import f1_score, precision_score, recall_score,confusion_matrix, make_scorer, precision_score
from linear_r2 import HyperplaneR2
from sklearn.neural_network import MLPClassifier

#importing dataset
dataset = pd.read_excel('C:/Users/Carlo/Desktop/MIA/myenv/datasets/default of credit card clients.xls', skiprows = 1)
display(dataset.info())
display(dataset.head())

dataset_copy = dataset.copy()

# Suddivisione del dataset in Dati e Targets
X = dataset_copy.iloc[:, np.arange(1,24)].values             # tolgo la prima e l'ultima colonna!
y = dataset_copy['default payment next month'].values

test_p = 0.5
randomstate = 269072

# Split dei dati
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_p)

# PCA
pca_default_std = PCA()
scaler_default = StandardScaler()
scaler_default.fit(X_train)
X_default_scaled = scaler_default.transform(X_train)
pca_default_std.fit(X_default_scaled)

pca_default = PCA()
pca_default.fit(X_train)

plt.figure()
plt.plot(np.insert(np.cumsum(pca_default_std.explained_variance_ratio_), 0, 0), color = 'blue')
plt.title('DEFAULT PCA STANDARDIZED')
plt.xlabel('Principal components')
plt.ylabel('Cumulative explained variance (%)')
plt.grid()
plt.show()

plt.figure()
plt.plot(np.insert(np.cumsum(pca_default.explained_variance_ratio_), 0, 0), color = 'blue')
plt.title('DEFAULT COMPLETE PCA NO STANDARDIZATION')
plt.xlabel('Principal components')
plt.ylabel('Cumulative explained variance (%)')
plt.grid()
plt.show()

plt.figure()
plt.bar(np.arange(1,24,1), pca_default_std.explained_variance_ratio_, color = 'YELLOW')
plt.plot(np.arange(1,24,1), pca_default_std.explained_variance_ratio_, 'b--')
plt.title('SKREE PLOT')
plt.xlabel('Principal components')
plt.ylabel('Cumulative explained variance')
plt.grid()
plt.show()

# PCA with 95% ov variance
pca_default_std = PCA(0.95, svd_solver = 'full')
scaler_default = StandardScaler()
scaler_default.fit(X_train)
X_default_scaled = scaler_default.transform(X_train)
pca_default_std.fit(X_default_scaled)

# obtaining matrix
X_train_PCA = pca_default_std.transform(X_default_scaled)
Y_train_PCA = y_train

X_test_PCA = pca_default_std.transform(scaler_default.transform(X_test))
Y_test_PCA = y_test
C_hard = 5000
loss_hard = 'squared_hinge'
dual_hard = False
C_soft = 1
loss_soft = 'hinge'
dual_soft = True

# Inizializzazione SVM_PCA
lsvm_hard_pca = LinearSVC(C = C_hard, loss = loss_hard, dual = dual_hard, random_state = randomstate)
lsvm_soft_pca = LinearSVC(C = C_soft, loss = loss_soft, dual = dual_soft, random_state = randomstate)

# Addestramento SVM_PCA
lsvm_hard_pca.fit(X_train_PCA, Y_train_PCA)
lsvm_soft_pca.fit(X_train_PCA, Y_train_PCA)

LinearSVC(C=1, loss='hinge', random_state=269072)
df_lsvm = pd.DataFrame({'accuracy soft': [lsvm_soft_pca.score(X_train_PCA, Y_train_PCA), lsvm_soft_pca.score(X_test_PCA, Y_test_PCA)],
                        'accuracy hard': [lsvm_hard_pca.score(X_train_PCA, Y_train_PCA), lsvm_hard_pca.score(X_test_PCA, Y_test_PCA)]},
                            index=['training', 'test'])
display(df_lsvm)

CMSVMsoft = confusion_matrix(Y_test_PCA, lsvm_soft_pca.predict(X_test_PCA))
CMSVMhard = confusion_matrix(Y_test_PCA, lsvm_hard_pca.predict(X_test_PCA))
plt.figure(figsize=(4,4))
plt.title('Confusion Matrix SVM soft')
sn.heatmap(CMSVMsoft, annot=True, fmt = "d")
plt.show()

plt.figure(figsize=(4,4))
plt.title('Confusion Matrix SVM hard')
sn.heatmap(CMSVMhard, annot=True, fmt = "d")
plt.show()

# setting parametres for SVM Kenrle trick
ker_rbf = 'rbf'
gamma_rbf = 'auto'

C1 = 1
C2 = 5000
#  SVM
svm_rbf_1_pca = SVC(C = C1, kernel = ker_rbf, gamma = gamma_rbf, random_state = randomstate)
svm_rbf_2_pca = SVC(C = C2, kernel = ker_rbf, gamma = gamma_rbf, random_state = randomstate)

# training SVM for PCA matrix
svm_rbf_1_pca.fit(X_train_PCA, Y_train_PCA)
svm_rbf_2_pca.fit(X_train_PCA, Y_train_PCA)
SVC(C=5000, gamma='auto', random_state=269072)
df_svm_kt = pd.DataFrame({'Accuracy hard': [svm_rbf_2_pca.score(X_train_PCA, Y_train_PCA), svm_rbf_2_pca.score(X_test_PCA, Y_test_PCA)],
                            'Accuracy soft': [svm_rbf_1_pca.score(X_train_PCA, Y_train_PCA), svm_rbf_1_pca.score(X_test_PCA, Y_test_PCA)]},
                            index=['training', 'test'])
display(df_svm_kt)

CMSVMKTsoft = confusion_matrix(Y_test_PCA, svm_rbf_1_pca.predict(X_test_PCA))
CMSVMKThard = confusion_matrix(Y_test_PCA, svm_rbf_2_pca.predict(X_test_PCA))
plt.figure(figsize=(4,4))
plt.title('Confusion Matrix SVM KT soft')
sn.heatmap(CMSVMKTsoft, annot=True, fmt = "d")
plt.show()

plt.figure(figsize=(4,4))
plt.title('Confusion Matrix SVM KT hard')
sn.heatmap(CMSVMKThard, annot=True, fmt = "d")
plt.show()

# MLP
hidden_layer_sizes = [256]*3               #3 hidden layers
activation = 'relu'
patience = 75
max_epochs = 5000;
verbose = False
batch_sz = 100                             # mini batch
validation_perc = 0.5
mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, early_stopping=True,  solver='adam',
                    batch_size = batch_sz, learning_rate='constant', max_iter=max_epochs, shuffle=True,verbose=verbose,
                    warm_start=False,  validation_fraction = validation_perc, n_iter_no_change= patience, random_state = randomstate)
# Training MLP
mlp.fit(pca_default_std.fit_transform(X_default_scaled), y_train)

y_pred_trainval = mlp.predict(X_train_PCA)
y_pred_test = mlp.predict(X_test_PCA)

# values training and validation set
acc_trainval = mlp.score(X_train_PCA, y_train)
prec_trainval = precision_score(y_train, y_pred_trainval, average='weighted')
rec_trainval = recall_score(y_train, y_pred_trainval, average='weighted')
f1_trainval = f1_score(y_train, y_pred_trainval, average='weighted')

# values test set
acc = mlp.score(X_test_PCA, y_test)
prec = precision_score(y_test, y_pred_test, average='weighted')
rec = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

df_perf = pd.DataFrame({'Accuracy': [acc_trainval, acc],
                        'Precision': [prec_trainval, prec],
                        'Recall': [rec_trainval, rec],
                        'F1': [f1_trainval, f1]
                       },
                      index=['train. + val.', 'test'])

cmat = confusion_matrix(y_test, y_pred_test, labels=mlp.classes_)
cmat_norm_true = confusion_matrix(y_test, y_pred_test, labels=mlp.classes_, normalize='true')
cmat_norm_pred = confusion_matrix(y_test, y_pred_test, labels=mlp.classes_, normalize='pred')

indicatori = ['not default', 'default']
df_cmat = pd.DataFrame(cmat, columns=indicatori, index=indicatori)
df_cmat_norm_true = pd.DataFrame(cmat_norm_true, columns=indicatori, index=indicatori)
df_cmat_norm_pred = pd.DataFrame(cmat_norm_pred, columns=indicatori, index=indicatori)

print('USEFUL VALUES')
display(df_perf)

plt.figure(figsize=(4,4))
plt.title('Confusion Matrix MLP')
sn.heatmap(cmat, annot=True, fmt = "d")
plt.show()

print('CONFUSION MATRIX NORMALIZE OVER THE ROWS')
display(df_cmat_norm_true)
print('CONFUSION MATRIX NORMALIZE OVER THE COLUMNS')
display(df_cmat_norm_pred)

y_pred_test_lsvm = lsvm_soft_pca.predict(X_test_PCA)
acc_lsvm = mlp.score(X_test_PCA, y_test)
prec_lsvm = precision_score(y_test, y_pred_test_lsvm, average='weighted')
rec_lsvm = recall_score(y_test, y_pred_test_lsvm, average='weighted')
f1_lsvm = f1_score(y_test, y_pred_test_lsvm, average='weighted')

y_pred_test_svm = svm_rbf_1_pca.predict(X_test_PCA)
acc_svm = mlp.score(X_test_PCA, y_test)
prec_svm = precision_score(y_test, y_pred_test_svm, average='weighted')
rec_svm = recall_score(y_test, y_pred_test_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_test_svm, average='weighted')

val_mlp = [acc, rec, prec, f1]
val_svm_soft = [acc_lsvm, prec_lsvm, rec_lsvm, f1_lsvm ]
val_svm_kt_soft = [acc_svm, prec_svm, rec_svm, f1_svm]

colors = ['#E69F00', '#56B4E9', '#F0E442']
names = ['MLP', 'SVM', 'SVMKT']

fig, axs = plt.subplots(1,4, figsize = (10,6))
axs[0].bar(names, [val_mlp[0], val_svm_soft[0], val_svm_kt_soft[0]], color = colors)
axs[0].set_title('Accuracy')
axs[1].bar(names, [val_mlp[1], val_svm_soft[1], val_svm_kt_soft[1]], color = colors)
axs[1].set_title('Precision')
axs[2].bar(names, [val_mlp[2], val_svm_soft[2], val_svm_kt_soft[2]], color = colors)
axs[2].set_title('Recall')
axs[3].bar(names, [val_mlp[3], val_svm_soft[3], val_svm_kt_soft[3]], color = colors)
axs[3].set_title('F1')
