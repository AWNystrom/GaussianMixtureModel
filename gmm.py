from numpy import mean, std, e, pi, log, argmax, array
from numpy.random import uniform

def pdf(x, m, s):
    if s == 0:
        s += 10**-10
    return e**(-(x-m)**2/(2*s**2))/(s*(2*pi)**0.5)

class GMM(object):
    def __init__(self, n_components, trials=1, max_iters=100):
        self.n_components = n_components
        self.max_iters = max_iters
        self.trials = trials
    
    def fit(self, X, Y=None, **fit_params):
        X = array(X)
        #initialize params
        low, high = min(X), max(X)
        s_init = std(X)
        best_likelihood = None
        best_params = None
        
        for trial in xrange(self.trials):
            param_pairs = [(uniform(low, high), s_init) for i in xrange(self.n_components)]
            prev_likelihood = None
        
            for iter in xrange(self.max_iters):
                #Separate points into cluster groups
        
                #EXPECTATION
                cluster_ids = [argmax([pdf(x, m, s) for m, s in param_pairs]) for x in X]
                cluster_to_points = {c: [] for c in xrange(len(param_pairs))}
                for i, c in enumerate(cluster_ids):
                    cluster_to_points[c].append(i)
        
                #MAXIMIZATION
                log_likelihood = 0
                for c in xrange(len(param_pairs)):
                    pts = X[cluster_to_points[c]]
                
                    if len(pts) == 0:
                        continue
                    elif len(pts) == 1:
                        m = pts[0]
                        s = 10**-10
                        param_pairs[c] = (m, s)
                    else:
                        m, s = mean(pts), std(pts)
                        param_pairs[c] = (m, s)
                        
                    log_likelihood += sum(log(pdf(x, m, s)) for x in pts)#/len(pts)
        
                #Stop early?
                if log_likelihood == prev_likelihood:
                    break
                
                prev_likelihood = log_likelihood
            
            #Track best params
            if log_likelihood > best_likelihood:
                best_likelihood = log_likelihood
                best_params = list(param_pairs)
        
        #calculate Bayesian Information Criterion
        self.params = best_params
        self.likelihood = best_likelihood
        self.bic = -2*best_likelihood + 2*self.n_components*log(X.shape[0])
    
    def predict(self, X):
        return array([argmax([pdf(x, m, s) for m, s in self.params]) for x in X])

if __name__ == '__main__':
    from numpy.random import normal, randint, uniform
    import matplotlib.pyplot as plt
    from numpy import arange
    
    abs_max_m, max_s = 100, 1
    true_k = randint(1, 5)
    params = [(uniform(-abs_max_m, abs_max_m), uniform(0, max_s)) for i in xrange(true_k)]
    counts = [randint(20, 100) for i in xrange(true_k)]
    
    X = [[normal(m, s) for i in xrange(c)] for (m,s), c in zip(params, counts)]
    X = array(reduce(lambda a,b: a+b, X))
    
    best_bic = None
    best_k = None
    bics = []
    ks = []
    for k in xrange(1, 5*true_k):
        clf = GMM(k)
        clf.fit(X)
        
        print k, clf.bic
        bics.append(clf.bic)
        ks.append(k)
        if clf.bic > best_bic:
            best_bic = clf.bic
            best_k = k
    
    clf = GMM(best_k)
    clf.fit(X)
    
    plt.scatter(X, [0 for x in X], alpha=0.01)
    l,h = min(X), max(X)
    X = arange(l, h, (h-l)/1000)
    for m, s in clf.params:
        plt.plot(X, [pdf(x, m, s) for x in X])
    
    print 'True', true_k
    print 'chosen', best_k
    plt.show()
    
    plt.scatter(ks, bics)
    plt.show()