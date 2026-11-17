from cluster import Cluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class KMeansCluster(Cluster):
    def __init__(self, vitimas, features, K):
        super().__init__(vitimas, features)
        
        self.K = K
        self.modelo = None
        self.sse = None
        self.silhueta = None
    
    def executar(self):
        self.normalizar_dados()
        
        self.modelo = KMeans(
            n_clusters=self.K, 
            random_state=67, 
            n_init=10
        )
        
        self.clusters = self.modelo.fit_predict(self.X)
        self.sse = self.modelo.inertia_
        self.silhueta = silhouette_score(self.X, self.clusters)
        
        return self.clusters