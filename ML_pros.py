import pandas as pd
import numpy as np
from collections import Counter
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import IsolationForest
from sklearn import decomposition
import warnings
warnings.filterwarnings("ignore")

#Importing Dataset
Gene_Data = pd.read_excel('prad_tcga_genes.xlsx')
Patient_Data = pd.read_excel('prad_tcga_clinical_data.xlsx')
Gene_Data = Gene_Data.drop(columns=['ID'])

#Extracting Y Value
Y = Patient_Data['GLEASON_SCORE']
Y.drop(Y.tail(5).index,inplace = True)

#Transposing Data
X_old = Gene_Data.transpose()
X_old.drop(X_old.columns[60483], axis=1, inplace=True)

# Plot for dataset with outliers
pca = PCA(n_components=10)
pca_data = pca.fit_transform(X_old,Y)

lda = LinearDiscriminantAnalysis(n_components = 2)
pca_lda_data = lda.fit_transform(pca_data,Y)

label = [6,7,8,9,10]
colors = ['red','green','blue','purple','pink']

fig = plt.figure(figsize=(5,5))
plt.scatter(pca_lda_data[:,0],pca_lda_data[:,1], c=Y, cmap=matplotlib.colors.ListedColormap(colors),alpha=0.7)
plt.title('Dataset with Outliers')
