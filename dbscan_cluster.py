from cluster import Cluster
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

class DBScanCluster(Cluster):
    def __init__(self, vitimas, features, eps, min_samples):
        super().__init__(vitimas, features)
        
        self.eps = eps
        self.min_samples = min_samples
        
        self.modelo = None
        self.n_clusters = None
        self.n_ruidos = None
        
        self.cobertura = None
        self.silhueta_global = None
        self.score_final = None
        
    def executar(self):
        self.normalizar_dados()
        
        self.modelo = DBSCAN(
            eps=self.eps, 
            min_samples=self.min_samples
        )
        
        self.clusters = self.modelo.fit_predict(self.X)
        
        nao_ruido = self.clusters != -1
        self.n_ruidos = list(self.clusters).count(-1)
        self.n_clusters = len(set(self.clusters) - {-1})
        
        total_pontos = len(self.X)
        pontos_validos = np.sum(nao_ruido)
        
        self.cobertura = pontos_validos / total_pontos if total_pontos > 0 else 0
        
        if self.n_clusters > 1 and pontos_validos > self.n_clusters:
            self.silhueta_global = silhouette_score(self.X[nao_ruido], self.clusters[nao_ruido])
            self.score_final = self.silhueta_global * self.cobertura
        else:
            self.silhueta_global = 0.0
            self.score_final = 0.0
        
        return self.clusters