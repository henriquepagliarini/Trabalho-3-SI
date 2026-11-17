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
    
def executar_kmeans_basico(vitimas):
    print('K-Means básico (apenas usando a posição espacial)')
    print(f"{'K':<4} | {'SSE':<12} | {'Coef. Silhueta':<15}")
    print('-' * 36)
    
    ks = list(range(2, 13))
    sses = []
    silhuetas = []
    
    for k in ks:
        kmeans_basico = KMeansCluster(
            vitimas,
            ['x_rel', 'y_rel'], 
            k
        )
        
        kmeans_basico.executar()
        
        sses.append(kmeans_basico.sse)
        silhuetas.append(kmeans_basico.silhueta)
        
        print(f'{k:<4} | {kmeans_basico.sse:<12.4f} | {kmeans_basico.silhueta:<15.4f}')
    
    plt.figure(figsize=(7, 4))
    plt.plot(ks, sses, marker='o', linewidth=2)
    plt.title('K-Means Básico: K por SSE (Posição Espacial)')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('SSE')
    plt.xticks(ks)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('resultados/grafico_k_por_sse.png', dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(ks, silhuetas, marker='s', color='green', linewidth=2)
    plt.title('K-Means Básico: K por Coeficiente de Silhueta')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Coeficiente de Silhueta')
    plt.xticks(ks)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('resultados/grafico_k_por_silhueta.png', dpi=300)
    plt.close()
        
def main():
    dataset = DataSet('./datasets/map.csv')
    data_vict = dataset.obter_dados_vitimas()
    
    models = MLModels("models")
    vitimas = models.calcular(dataset.vitimas, data_vict)
    
    os.makedirs('resultados', exist_ok=True)
    
    plotar_nuvem_pontos(vitimas)
    executar_kmeans_basico(vitimas)
    
if __name__ == '__main__':
    main()