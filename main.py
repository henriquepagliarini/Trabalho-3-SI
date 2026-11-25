from dataset import DataSet
from mlmodels import MLModels
from kmeans_cluster import KMeansCluster
from dbscan_cluster import DBScanCluster

import matplotlib.pyplot as plt
import os

def plotar_nuvem_pontos(vitimas):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        vitimas["x_rel"],
        vitimas["y_rel"],
        vitimas["sobr"],
        s=45,
        edgecolors="k",
        alpha=0.8,
    )

    ax.set_title(f"Nuvem de Pontos K-Means 3D (K=7)\nFeatures: [x_rel, y_rel, sobr]")
    ax.set_xlabel("x_rel (X)")
    ax.set_ylabel("y_rel (Y)")
    ax.set_zlabel("sobr (Prob. Sobrevivência)")
    plt.tight_layout()
    plt.savefig("resultados/kmeans_nuvem_pontos_3d_sobr.png", dpi=300)
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

def executar_kmeans_sobr_espacial(vitimas):
    print("\n" + "=" * 70)
    print("EXECUTANDO K-MEANS: ESPACIAL + SOBREVIVÊNCIA (x_rel, y_rel, sobr)")
    print("=" * 70)

    features = ["x_rel", "y_rel", "sobr"]
    k = 7
    
    kmeans = KMeansCluster(vitimas, features, k)
    kmeans.executar()

    print(f"K = {k:<2} | SSE = {kmeans.sse:<10.4f} | Silhueta = {kmeans.silhueta:<8.4f}")

    os.makedirs("resultados", exist_ok=True)
    kmeans.salvar_clusters("resultados")

def executar_dbscan(vitimas):
    print("\nDBScan - 3 parametrizações diferentes")
    print(
        f"{'Par':<4} | {'eps':<6} | {'min_samp':<8} | {'Clusters':<8} |"
        f" {'Ruídos':<8} | {'Cobertura':<10} | {'Silh. Global':<12} | {'Score'}"
        f" {'Final':<10}"
    )
    print('-' * 88)
    
    parametros = [
        {'eps': 0.2, 'min_samples': 3},
        {'eps': 0.3, 'min_samples': 5},
        {'eps': 1.0, 'min_samples': 10},
    ]
    
    for i, params in enumerate(parametros, start=1):
        dbscan = DBScanCluster(vitimas, ['x_rel', 'y_rel', 'sobr'], eps=params['eps'], min_samples=params['min_samples'])
        dbscan.executar()
        
        print(
            f'{i:<4} | {params["eps"]:<6} | {params["min_samples"]:<8} |'
            f' {dbscan.n_clusters:<8} | {dbscan.n_ruidos:<8} | {dbscan.cobertura:<10.4f} |'
            f' {dbscan.silhueta_global:<12.4f} | {dbscan.score_final:<10.4f}'
        )
        
def main():
    dataset = DataSet('./datasets/map.csv')
    data_vict = dataset.obter_dados_vitimas()
    
    models = MLModels("models")
    vitimas = models.calcular(dataset.vitimas, data_vict)
    
    dataset.df['sobr'] = dataset.df['sobr'].astype('float64')
    dataset.df.update(vitimas[['tri', 'sobr']])
    
    os.makedirs('resultados', exist_ok=True)
    dataset.df.to_csv('./resultados/map_tri_sobr.csv', index=False)
    
    plotar_nuvem_pontos(vitimas)
    executar_kmeans_basico(vitimas)
    executar_dbscan(vitimas)
    executar_kmeans_sobr_espacial(vitimas)
    
if __name__ == '__main__':
    main()