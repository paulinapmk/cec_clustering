import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def pick_colors(how_many):
    """'pick_colors(how_many) chooses 'how_many' random colors"""
    cols = []
    for i in range(how_many):
        cols.append(np.random.rand(3, 1))
    return cols

COLORS = pick_colors(20)

def multivariate_normal_pdf(y, m, cov, N):
    """ 'multivariate_normal_pdf(y, m, cov, N)' returns the multivariate normal
    density for a given point 'y', with given mean 'm', covariance matrix 'cov'
    and defined dimension of data 'N'"""
    B = np.sqrt(pow(2 * np.pi, N)) * np.sqrt(np.linalg.det(cov))
    delta = y - m
    sigma = np.linalg.inv(cov)
    m = np.dot(np.dot(delta, sigma), delta)
    return np.exp(-0.5*m)/B

def multivariate_normal_pdf_for_data(data, m, cov, N):
    """ 'multivariate_normal_pdf_for_data(data, m, cov, N)' returns an array with
    the densities of the dataset 'data', with given means 'm', 
    covariance matrix 'cov' and defined dimension of data 'N'"""
    z = []
    for row in data:
        temp = []
        for y in row:
            pdf = np.asarray(multivariate_normal_pdf(y, m, cov, N)).flatten()
            temp.append(pdf[0])
        z.append(temp)
    return np.asarray(z)

def divide_points_into_clusters(Y, C, clusters):
    """This function divides the data 'Y' into defined clusters."""
    clustered = zip(Y, C)
    for pair in clustered:
        clusters[pair[1]].append(pair[0])
    return clusters

def assign_clusters(Y, C, k):
    """This function assigns the clusters to the elements of data 'Y'
    basing on the information about clustering 'C'"""
    clusters = []
    [clusters.append([]) for i in range(k)]
    clusters = divide_points_into_clusters(Y, C, clusters)
    return clusters

def clustering(Y, C, k, means, cov_matrix, N, p, cost):
    """This function performs the clustering for each point of the dataset 'Y'
    running the multivariate_normal_pdf function and it returns an array of
    the minimum costs obtained"""
    A = []
    i = 0
    for y in Y:
        cost = []
        for m in range(0, k):
            pdf = multivariate_normal_pdf(y, means[m], cov_matrix[m], N)
            cost.insert(i, -(np.log(p[m])) - np.log(pdf))
        A.append(np.argmin(cost))
        i += 1
    return np.asarray(A)

def percentage(part, whole):
    """ This function calculates the percentage of a given set """
    return float(part)*float(whole)/100

def eigsorted(cov):
    """"'eigsorted(cov)' returns the eigenvalues and the eigenvectors
    basing on the covariance matrix 'cov'"""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def reduce_clusters(clusters, Y, cost, k, C, means, cov_matrix, N, p, percent):
    """This function checks whether a cluster has less than 'percent' of
    total number of elements 'Y' or not. All the clusters with a lesser number
    of elements are removed within their means and covariances from 
    the respective matrices."""
    index = 0
    for c in clusters:
        all_data_length = len(Y)
        if len(c) < percentage(percent, all_data_length):
            clusters.pop(index)
            k -= 1
            means.pop(index)
            cov_matrix.pop(index)
            p.pop(index)
            index -= 1
        C = clustering(Y, C, k, means, cov_matrix, N, p, cost)
        clusters = assign_clusters(Y, C, k)
        index += 1
    return C, clusters, means, cov_matrix, p

def compute_new_values(k, Y, C, N, means, cov_matrix, p, model, trace_matrix, **args):
    """This functions calculates the new values of 'means', probabilities 'p'
    and covariance and trace matrices. In the **args argument 'r' for radial
    model can be transmitted."""
    for i in range(0, k):
        mapped_y = Y[C == i]
        means[i] = np.mean(mapped_y, 0)
        p[i] = float(len(Y[C==i])) / float(len(Y))
        if(model == "gaussian"): #1
            cov_matrix[i] = np.cov(mapped_y[:, 0], mapped_y[:, 1])
            trace_matrix.append(np.trace(np.linalg.inv(cov_matrix[i]).dot(cov_matrix[i])))
        elif(model == "radial"): #2
            r = args['r']
            cov_matrix[i] = r * np.matrix(np.identity(N), copy=False)
            trace_matrix.append(np.trace(np.cov(mapped_y[:, 0], mapped_y[:, 1])))
        elif(model == "spherical"): #3
            trace_matrix.append(np.trace(np.cov(mapped_y[:, 0], mapped_y[:, 1])))
            cov_matrix[i] = trace_matrix[i]/N * np.matrix(np.identity(N), copy=False)
        elif(model == "diagonal"): #4
            cov_matrix[i] = np.diagflat(np.diagonal(np.cov(mapped_y[:, 0], mapped_y[:, 1])))
        elif(model == "ellipsoidal"): #5
            cov_matrix[i] = np.cov(mapped_y[:, 0], mapped_y[:, 1])
        else: #6
            cov_matrix[i] = np.cov(mapped_y[:, 0], mapped_y[:, 1])
    return means, p, cov_matrix, trace_matrix

def calculate_entropy(k, Y, C, N, p, cov_matrix, trace_matrix, model, **args):
    """This function calculates and returns the entropy for each cluster basing 
    on the chosen model. It takes the number of clusters 'k', the dataset 'Y',
    the assigned clusters 'C', the dimension of data 'N', the matrix of
    probabilities for each cluster 'p', the covariance matrix 'cov_matrix',
    the trace matrix and the model chosen. In the **args argument 'r' for radial 
    model and 'expected_eigenvalues' can be transmitted."""
    new_h = 0
    for i in range(0, k):
        if(model == "gaussian"): #1
            new_h += p[i] * (N/2 * np.log(2*np.pi) + 0.5 * trace_matrix[i] + 
                      0.5 * np.log(np.linalg.det(cov_matrix[i])))
        elif(model == "radial"): #2
            r = args['r']
            new_h += p[i] * (N/2 * np.log(2 * np.pi) + N/2 * np.log(r) + 
                      (1./(2*r))*trace_matrix[i])
        elif(model == "spherical"): #3
            new_h += p[i] * (- np.log(p[i]) + N/2 * np.log(trace_matrix[i]) + 
                      N/2 * np.log((2 * np.pi * np.e))/2)
        elif(model == "diagonal"): #4
            mapped_y = Y[C == i]
            new_h += p[i] * (N/2 * np.log(2 * np.pi * np.e) + 
                      0.5 * np.log(np.linalg.det(np.diagflat(np.diagonal(np.cov(mapped_y[:, 0], mapped_y[:, 1]))))))
        elif(model == "ellipsoidal"): #5    
            eigenvals, eigenvects = eigsorted(cov_matrix[i])
            new_h += p[i] * (N/2 * np.log(2 * np.pi) + 0.5 * 
                      np.sum([m/a for (a, m) in zip(eigenvals, args['expected_eigenvalues'])]) + 
                      0.5 * np.log(np.prod(eigenvals)))
        else: #6
            new_h += p[i] * (- np.log(p[i]) + 0.5 * np.log(np.linalg.det(cov_matrix[i])) + 
                      N/2 * np.log(2 * np.pi * np.e))
    return new_h

def lloyd(k, epsilon, Y, N, percent, plot_progress = None, model="normal", **args):
    """This function implements the LLoyd's algorithm and returns the centroids
    and the clusters with all the data related to them: covariance matrices, 
    probabilities, the optimal entropy, the optimal number of clusters and the 
    array 'h' which contains the value of the entropy at each iteration"""
    #initial conditions
    #choose randomly k centers
    centroids = Y[np.random.choice(np.arange(len(Y)), k), :]
    C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in Y])
    p, cov_matrix, h, cost, means = [], [], [], [], []
    h.append(10000)
    for i in range(0, k):
        mapped_y = Y[C==i]
        p.append(float(len(mapped_y)) / float(len(Y)))
        cov_matrix.append(np.cov(mapped_y.reshape(N, len(mapped_y))))
        means.append(np.mean(mapped_y, 0))
    n = 0   
    while True:
        C = clustering(Y, C, k, means, cov_matrix, N, p, cost)  
        clusters = assign_clusters(Y, C, k)
        if plot_progress != None: showclusters(Y, C, np.array(means), k) #show progress
        n += 1
        #delete cluster if Card(Y) < w%
        C, clusters, means, cov_matrix, p = reduce_clusters(clusters, Y, cost, k, C, means, cov_matrix, N, p, percent)
        trace_matrix = []
        means, p, cov_matrix, trace_matrix = compute_new_values(k, Y, C, N, means, cov_matrix, p, model, trace_matrix, **args)
        new_h = calculate_entropy(k, Y, C, N, p, cov_matrix, trace_matrix, model, **args)
        h.append(new_h)
        if h[n] >= h[n-1] - epsilon:
            break
    return np.array(means), clusters, C, cov_matrix, p, new_h, k, h

def showclusters(X, C, centroids, K, keep = False):
    """This functions plots the clusters in different colors but without the 
    contour plot"""
    import time
    time.sleep(0.5)
    plt.cla() #clear the subplot
    plt.clf() #clear figure
    fig = plt.figure(1)
    ax = fig.add_subplot(111, aspect='auto')
    ax.set_xlim(X[:,0].min()-2, X[:,0].max()+2)
    ax.set_ylim(X[:,1].min()-2, X[:,1].max()+2)

    show_points(X, C, centroids, ax, K)    
    plt.show()
    
    if keep :
        plt.ioff() #interactive graphics off
        plt.show()
        
def show_points(X, C, centroids, ax, K):
    """This function plots the scatter plot of data 'X' basing
    on the information about the clusters."""
    for i in range(K):
        ax.plot(X[C==i, 0], X[C==i, 1], '+', c=COLORS[i], alpha=0.5)
    ax.plot(centroids[:,0],centroids[:,1],'*y',markersize=10)
    
def contour_plot(data, means, cov, N, C, K, Pr):
    """This function plots the contour plot of the definitive clustering."""
    plt.cla() #clear the subplot
    plt.clf() #clear figure
    fig = plt.figure(1)
    ax = fig.add_subplot(111, aspect='auto')
    x = np.linspace(data[:,0].min()-0.5, data[:,0].max()+0.5, 50)
    y = np.linspace(data[:,1].min()-0.5, data[:,1].max()+0.5, 50)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    show_points(data, C, means, ax, K)
    for i in range(K):
        z = Pr[i]*multivariate_normal_pdf_for_data(pos, means[i], cov[i], N) 
        plt.contour(x, y, z)     
#        c = plt.contour(x, y, z) #uncomment if you want labeled contour plot 
#        plt.clabel(c) #uncomment if you want labeled contour plot
    plt.axis('equal')
    plt.savefig("samples.png")
    plt.show()

def run(n, data, N=2, K=1, percent=5, epsilon=0.01, model="normal", plot_progress=None, show_entropy=None, show_stats=None, **args):
    """This function runs the main modules of the package. It returns the computed
    clusters in an array corresponding to the elements in 'data'."""
    centroids, C, cov, p = [], [], [], []
    h = 10000
    for i in range(0, n):
        print "RUN ", i+1
        clusters = []
        [clusters.append([]) for i in range(K)]
        means, def_clusters, new_C, cov_matrix, pr, last_h, k, entropia = \
            lloyd(K, epsilon, data, N, percent, plot_progress=plot_progress, model=model, **args)
        if last_h < h:
            h = last_h
            centroids = means
            C = new_C
            cov = cov_matrix
            p = pr
            optimal_k = k
            entropy = entropia
    if show_entropy != None: show_final_entropy(optimal_k, h, entropy)
    contour_plot(data, centroids, cov, N, C, optimal_k, p)
    return C
    
def show_final_entropy(optimal_k, h, entropy):
    """This functions prints the optimal number of clusters 'k' computed by the
    algorithm, it prints the final entropy of the best clustering with its plot"""
    print "Optimal k: ", optimal_k
    print "Final entropy: ", h
    entr=entropy[1:]
    plt.plot(entr)
    #plt.show()
    plt.savefig("entropy.png")
    

def show_evaluations(labels_true, labels_pred):  
    """This functions needs to be run separately. It prints the evaluation scores
    basing on the true labels and on the predicted ones."""
    print "Adjusted Rand index: ", metrics.adjusted_rand_score(labels_true, labels_pred)
    print "Mutual Information based score: ", metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    print "Homogeneity: ", metrics.homogeneity_score(labels_true, labels_pred)
    print "Completeness: ", metrics.completeness_score(labels_true, labels_pred) 
    print "V-measure: ", metrics.v_measure_score(labels_true, labels_pred)
    print "Fowlkes-Mallows: ", metrics.fowlkes_mallows_score(labels_true, labels_pred)
    
def read(filename):
    """The function 'read(filename)' returns the stacked points of the file 'filename'"""
    points = []
    input = open(filename)
    for line in input:
        points.append(map(float, line.split()))
    return np.vstack(points)

def load(name):
    """The function 'load(filename)' loads prepared data basing on its 'name'
    and returns stacked points and the target cluster; this works for 2D data"""
    import os
#    points = []
    script_dir = os.path.dirname(__file__)
    rel_path = "data/"+name+".txt"
    abs_file_path = os.path.join(script_dir, rel_path)
    data = read(abs_file_path)
    X, target = [], []
    for line in data:
        if len(line) == 3:
            target.append(int(line[2]))
        X.append([line[0], line[1]])
    return np.vstack(X), target

def hello():
    print "Hi. I am CEC"