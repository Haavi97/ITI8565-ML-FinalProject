from sklearn.decomposition import PCA

def pca_analysis(data, nfeatures):
    try:
        array = data.values
    except:
        array = data
        print('Data is already array-like')
        print(data)
    X = array[:,0:(nfeatures-1)]
    print(X)
    pca = PCA()
    fit = pca.fit(X)
    print("Explained Variance: %s" % fit.explained_variance_ratio_)
    print(fit.singular_values_)
    #print(fit.components_)