from cluster import Cluster
from sklearn.cluster import DBSCAN

class DBScanCluster(Cluster):
    def __init__(self, vitimas, features, eps, min_samples):
        super().__init__(vitimas, features)
        
        self.eps = eps
        self.min_samples = min_samples
        
        self.modelo = None
        self.n_clusters = None
        self.n_ruidos = None
        
    def executar(self):
        self.normalizar_dados()
        
        self.modelo = DBSCAN(
            eps=self.eps, 
            min_samples=self.min_samples
        )
        
        self.clusters = self.modelo.fit_predict(self.X)
        
        clusters = set(self.clusters)
        self.n_ruidos = list(self.clusters).count(-1)
        self.n_clusters = len(clusters - {-1})
        
        return self.clusters