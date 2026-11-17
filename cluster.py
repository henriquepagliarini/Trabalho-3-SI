from sklearn.preprocessing import StandardScaler

class Cluster():
    def __init__(self, vitimas, features):
        self.vitimas = vitimas.copy()
        self.features = features
        
        self.X = None
        self.clusters = None
        
    def normalizar_dados(self):
        dados = self.vitimas[self.features]
        
        scaler = StandardScaler()
        self.X = scaler.fit_transform(dados)
        
    def executar(self):
        raise NotImplementedError("O método executar deve ser implementado.")
    
    def obter_resultados(self):
        if self.clusters is None:
            raise RuntimeError("A clusterização ainda não foi executada.")
        
        resultado = self.vitimas.copy()
        resultado['cluster'] = self.clusters
        
        return resultado
    
    def salvar_clusters(self, dir='.'):
        if self.clusters is None:
            raise RuntimeError("Execute a clusterização antes de salvar.")
        
        import os
        
        os.makedirs(dir, exist_ok=True)
        
        resultado = self.obter_resultados()
        
        for cluster_id in sorted(set(self.clusters)):
            vitimas_cluster = resultado[resultado['cluster'] == cluster_id]
            
            file_name = f"cluster_{self.__class__.__name__}_{cluster_id}.txt"
            path = os.path.join(dir, file_name)
            
            with open(path, 'w', encoding='utf-8') as arquivo:
                for id_vitima in vitimas_cluster['id']:
                    arquivo.write(f"{id_vitima}\n")