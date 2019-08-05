import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

dir = "C:\\Users\\sudhakarceemavaram_n\\Desktop\\Dataset\\Kaggle\\Breast_Cancer.csv"

def get_dataset(dir):
    cancer_dataset = pd.read_csv(dir)
    dataset = cancer_dataset.loc[:,"id":"fractal_dimension_worst"]
    return dataset

#HeatMap:
def Heat_map(dataset):
    cormat = dataset.corr(method='spearman')
    f,ax = plt.subplots(figsize = (12,7))
    sns.heatmap(cormat,ax=ax,cmap="YlGnBu", linewidths=0.05)
    plt.savefig("Breast_Cancer_Correlation")
    plt.show()

#Linkage Matrix:
def correlation_map(dataset):
    cormat = dataset.corr(method="spearman")
    cluster = sns.clustermap(cormat,cmap="YlGnBu",linewidths=0.1)
    plt.setp(cluster.ax_heatmap.yaxis.get_majorticklabels(),rotation = 0)
    plt.savefig("correlation_image.png")
    plt.show()

def hist(dataset):
    dataset.hist()
    plt.savefig("correlation.png")
    plt.show()


def normalization(x):
    scaler = MinMaxScaler()
    scaled_values_X = scaler.fit_transform(X)
    scaled_X = pd.DataFrame(data=scaled_values_X, columns=X.columns)
    return scaled_X


def K_means_visualization(X):
    tsne = TSNE(verbose=1, perplexity=40, n_iter=4000)
    Y_tsne = tsne.fit_transform(X)
    #print(Y_tsne)

    # Kmeans Clustering:
    Cluster1 = KMeans(n_clusters=2)
    Km = Cluster1.fit_predict(X)
    labels = Cluster1.labels_
    # print(labels)
    plt.scatter(Y_tsne[:, 0], Y_tsne[:, 1], c=Km, cmap="jet", edgecolors=None)
    plt.title("Kmeans Clustering")
    plt.show()
    return Km, labels

def io_split(x,y):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=4, test_size=0.3)
    return xtrain,xtest,ytrain,ytest


def NB(xtrain,ytrain,xtest,ytest):
    NB = GaussianNB()
    NB.fit(xtrain, ytrain)
    Prediction = NB.predict(xtest)
    #print("Prediction: ", Prediction)
    #print("Actual: ", ytest)
    accuracy = accuracy_score(ytest, Prediction)
    return Prediction,accuracy


def DT_Gini(xtrain,ytrain,xtest,ytest):
    dtg = DecisionTreeClassifier(criterion="gini",random_state=200,min_samples_leaf=5,max_depth=5)
    dtg.fit(xtrain,ytrain)
    prediction = dtg.predict(xtest)
    accuracy = accuracy_score(ytest,prediction)
    return prediction, accuracy


def DT_Entropy(xtrain,ytrain,xtest,ytest):
    dte = DecisionTreeClassifier(criterion="entropy", random_state=50, min_samples_leaf=5, max_depth=5)
    dte.fit(xtrain,ytrain)
    prediction = dte.predict(xtest)
    accuracy = accuracy_score(ytest,prediction)
    return prediction,accuracy


dataset = get_dataset(dir)
#Heat_map(dataset)
#correlation_map(dataset)
dataset_used = dataset[["diagnosis","area_mean", "area_worst", "texture_mean", "concavity_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "area_se", "texture_se", "smoothness_se",
                        "compactness_se","smoothness_mean", "symmetry_se", "fractal_dimension_se", "smoothness_worst", "compactness_worst", "concave points_mean", "symmetry_worst"]]

X = dataset_used.drop("diagnosis",axis = 1)
Y = dataset_used.diagnosis

hist(dataset_used)
#Heat_map(dataset_used)
correlation_map(dataset_used)
scaled_x = normalization(X)
km , labels = K_means_visualization(scaled_x)
print("K-means prediction:")
print(km)
print("K-means labels: ")
print(labels)
print("\n\n")
xtrain,xtest,ytrain,ytest = io_split(scaled_x,Y)
prediction,accuracy = NB(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest)
prediction1,accuracy1 = DT_Gini(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest)
prediction2,accuracy2 = DT_Entropy(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest)
print("Prediction of NB: ", prediction)
print("Prediction of DT Gini: ", prediction1)
print("Prediction of DT Entropy: ", prediction2)

print("\n\n")

print("Accuracy of NB: ", accuracy)
print("Accuracy of DT Gini: ", accuracy1)
print("Accuracy of DT Entropy: ", accuracy2)
