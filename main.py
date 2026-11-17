from dataset import DataSet
from mlmodels import MLModels
from kmeans_cluster import KMeansCluster
from dbscan_cluster import DBScanCluster

def main():
    dataset = DataSet('./datasets/map.csv')
    data_vict = dataset.obter_dados_vitimas()
    
    models = MLModels("models")
    vitimas = models.calcular(dataset.vitimas, data_vict)

    print(vitimas[['id', 'tri', 'sobr']])
    vitimas.to_csv("datasets/vitimas_classificadas.csv", index=False)
    kmeans = KMeansCluster(
        vitimas,
        ['tri', 'pr'],
        K=5
    )

    kmeans.executar()
    kmeans.salvar_clusters('resultados')

    print("SSE:", kmeans.sse)
    print(kmeans.obter_resultados())

    dbscan = DBScanCluster(
        vitimas,
        ['tri', 'pr'],
        eps=0.1,
        min_samples=5
    )

    dbscan.executar()
    dbscan.salvar_clusters('resultados')
    
    print("Clusters:", dbscan.n_clusters)
    print("Ruídos:", dbscan.n_ruidos)
    print(dbscan.obter_resultados())
    
if __name__ == '__main__':
    main()