from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def pca_analysis(data, nfeatures, n=3):
    try:
        array = data.values
    except:
        array = data
        print('Data is already array-like')
        #print(data)
    X = array[:,0:(nfeatures-1)]
    # print(X)
    pca = PCA()
    fit = pca.fit(X)
    print("Explained Variance: %s" % fit.explained_variance_ratio_)
    print(fit.singular_values_)
    print(fit.get_params())
    #print(fit.components_)
    pca = PCA(n_components=n)
    pca.fit(X)
    return pca.transform(X)

def selec_best(X, y, n=3):
    X_new = SelectKBest(chi2, k=n).fit_transform(X, y)
    return X_new

def print_best(original, best,  tags):
    index = []
    i = 0
    for e in range(len(original[0])):
        if (original[0][e] - best[0][i]) < 0.0000001:
            index.append(e)
            i += 1
            if i >= len(best[0]):
                break
    result = []
    for j in range(i):
        result.append(tags[j])
    print(result)
    return result

def select_and_print_best(X, y, tags, n=3):
    best = selec_best(X, y, n=n)
    print('The {} features that best fit the labels:'.format(n))
    return print_best(X, best, tags)