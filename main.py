from dataset import DataSet
from mlmodels import MLModels
from kmeans_cluster import KMeansCluster
from dbscan_cluster import DBScanCluster

import matplotlib.pyplot as plt
import os

def plotar_nuvem_pontos(vitimas):
    plt.figure(figsize=(7, 5))
    plt.scatter(
        vitimas['tri'],
        vitimas['pr'],
        c='royalblue',
        alpha=0.7,
        edgecolors='k',
        s=40,
    )
    plt.title('Nuvem de Pontos: Características (Triagem vs. Pulso)')
    plt.xlabel('Triagem (tri)')
    plt.ylabel('Pulso (pr)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('resultados/nuvem_pontos_tri_pr.png', dpi=300)
    plt.close()
        
def main():
    dataset = DataSet('./datasets/map.csv')
    data_vict = dataset.obter_dados_vitimas()
    
    models = MLModels("models")
    vitimas = models.calcular(dataset.vitimas, data_vict)
    
    os.makedirs('resultados', exist_ok=True)
    
    plotar_nuvem_pontos(vitimas)
    
if __name__ == '__main__':
    main()