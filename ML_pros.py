import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import IsolationForest

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from statistics import mean
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN


def load_genomic_data(filename):
     # load the geneomic data
    genomic_df = pd.read_csv(filename)

    # Setting index to first column, else it will add its own indexing while doing transpose
    genomic_df.set_index('ID', inplace = True)
    
    # Need to take transpose since I want genes/features to be columns and each row should represent a patient information
    genomic_df = genomic_df.T
    
    # removing features with only zero values for all patients
    return genomic_df.loc[:, (genomic_df != 0).any(axis = 0)]    
    
def read_data(gfilename, cfilename):
    # Feature set, load geonomic data
    X = load_genomic_data(gfilename)
    
    # load the clinical data
    clinical_df = pd.read_csv(cfilename)
    print("Shape of genomic data: ", X.shape, " and Shape of clinical data: ", clinical_df.shape, "thus looks like we donot have genetic data for 5 patients, hence removing them")
    clinical_df = clinical_df.drop(labels=[213,227,297,371,469], axis=0)
    print("After droping 5 patients whose data were missing:\nShape of genomic data: ", X.shape, " and Shape of clinical data: ", clinical_df.shape, "\n")
    
    print("-- Checking if all patient ID's in genetic data set and clinical dataset matches\n")
    if(X.index.all() == clinical_df['PATIENT_ID'].all()):
        print("-- Yes, patient ID's in genetic data set and clinical dataset matches\n")
    else:
        print("Nope, patient ID's in genetic data set and clinical dataset do not match")
    
    y =  clinical_df['GLEASON_SCORE']                   
    
    return X, y


def visualize_data(X, y, title):
    # Visualizing dataset for outliers, using PCA prioir to LDA to prevent overfitting (https://stats.stackexchange.com/q/109810)
    pca = PCA(n_components=10)
    pca_reduced_data = pca.fit_transform(X,y)
    
    lda = LinearDiscriminantAnalysis(n_components = 2)
    pca_lda_reduced_data = lda.fit_transform(pca_reduced_data, y)
    
    # NOTE: Gleason score ranges from 6-10
    label = [6, 7, 8, 9, 10]
    colors = ['red','green','blue','purple','pink']
    
    fig = plt.figure(figsize=(6,6))
    plt.scatter(pca_lda_reduced_data[:,0], pca_lda_reduced_data[:,1], c=y, cmap=matplotlib.colors.ListedColormap(colors), alpha=0.7)
    plt.title(title)
    
    
def prepare_inputs(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)             
    return X_train_norm, X_test_norm

def remove_low_variance_feature(X):
    sel = VarianceThreshold()
    return sel.fit_transform(X)
    
    
def feature_selection(X_train_norm, y_train_enc, X_test_norm, score_function):
    best_k = SelectKBest(score_func=score_function, k=11)
    fit = best_k.fit(X_train_norm, y_train_enc)
    #print(fit.get_support(indices=True))
    X_train_fs = fit.transform(X_train_norm)
    X_test_fs = fit.transform(X_test_norm)
    
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_train.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Features','Score']
    print(featureScores.nlargest(9,'Score'))

    return X_train_fs, X_test_fs

def get_performace_measures(model, X_train, X_test, y_train, y_test, PPV_list, NPV_list, Specificity_list, Sensitivity_list, Accuracy_list):
    model = OneVsRestClassifier(model).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    Accuracy_list.append(accuracy_score(y_test, y_pred))
    

X, y = read_data('prad_tcga_genes.csv', 'prad_tcga_clinical_data.csv')

#Resampling dataset
sme = SMOTEENN(random_state=42,smote=SMOTE(random_state=42, k_neighbors=1))
X, y = sme.fit_resample(X, y)
print('Resampling of dataset using SMOTEENN %s' % Counter(y), '\n')

kf = KFold(n_splits = 10, shuffle=True, random_state=23)

PPV_list, NPV_list, Specificity_list, Sensitivity_list, Accuracy_list = [], [], [], [], []
for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    X_train_norm, X_test_norm = prepare_inputs(X_train, X_test)
    X_train_fs, X_test_fs = feature_selection(X_train_norm, y_train, X_test_norm, chi2)

    get_performace_measures(LinearDiscriminantAnalysis(), X_train_fs, X_test_fs, y_train, y_test, PPV_list, NPV_list, Specificity_list, Sensitivity_list, Accuracy_list)
    
print("-----------------------------------------------")
print("Accuracy: ", round(mean(Accuracy_list), 4))
print("-----------------------------------------------")
