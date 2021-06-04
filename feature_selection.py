from sklearn.decomposition import PCA

def pca_analysis(df, nfeatures):
    array = df.values
    X = array[:,0:nfeatures]
    pca = PCA()
    fit = pca.fit(X)
    
    print("Explained Variance: %s" % fit.explained_variance_ratio_)
    print(fit.components_)