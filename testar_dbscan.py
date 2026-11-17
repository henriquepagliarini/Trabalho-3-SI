
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from dbscan_cluster import DBScanCluster


# DataFrame compartilhado entre as tarefas de cada processo
_vitimas = None


def inicializar_processo(vitimas):
    global _vitimas
    _vitimas = vitimas


def executar_teste(parametros):

    features, eps, min_samples = parametros

    dbscan = DBScanCluster(
        _vitimas,
        list(features),
        eps=eps,
        min_samples=min_samples
    )

    dbscan.executar()

    labels = np.array(dbscan.clusters)
    mascara = labels != -1

    labels_validos = labels[mascara]
    X_validos = dbscan.X[mascara]

    cobertura = np.mean(mascara)
    n_clusters = dbscan.n_clusters

    if 2 <= n_clusters < len(labels_validos):
        silhouette = silhouette_score(
            X_validos,
            labels_validos
        )
    else:
        silhouette = np.nan

    score_final = (
        silhouette * cobertura
        if not np.isnan(silhouette)
        else np.nan
    )

    return {
        'features': ', '.join(features),
        'eps': eps,
        'min_samples': min_samples,
        'clusters': n_clusters,
        'ruidos': dbscan.n_ruidos,
        'cobertura': cobertura,
        'silhouette': silhouette,
        'score_final': score_final
    }

def executar_experimentos(vitimas):

    features_disponiveis = [
        'pr', 'tri'
    ]

    combinacoes = [
        features
        for tamanho in range(2, len(features_disponiveis) + 1)
        for features in combinations(features_disponiveis, tamanho)
        if 'tri' in features or 'sobr' in features
    ]

    valores_eps = [round(x, 1) for x in np.arange(0.1, 1.01, 0.05)]
    valores_min_samples = range(2, 20)

    experimentos = [
        (features, eps, min_samples)
        for features in combinacoes
        for eps in valores_eps
        for min_samples in valores_min_samples
    ]

    total = len(experimentos)

    n_processos = os.cpu_count() - 2 or 2

    # Divide os experimentos em lotes
    quantidade_lotes = n_processos * 8
    tamanho_lote = (total + quantidade_lotes - 1) // quantidade_lotes

    lotes = [
        experimentos[i:i + tamanho_lote]
        for i in range(0, total, tamanho_lote)
    ]

    print(f"Total de experimentos: {total}", flush=True)
    print(f"Processos: {n_processos}", flush=True)
    print(f"Lotes: {len(lotes)}", flush=True)

    resultados = []
    concluidos = 0

    with ProcessPoolExecutor(
        max_workers=n_processos,
        initializer=inicializar_processo,
        initargs=(vitimas,)
    ) as executor:

        tarefas = [
            executor.submit(executar_lote, lote)
            for lote in lotes
        ]

        for tarefa in as_completed(tarefas):
            resultado_lote = tarefa.result()
            resultados.extend(resultado_lote)

            concluidos += len(resultado_lote)

            percentual = (concluidos / total) * 100

            print(
                f"\rProgresso: {concluidos}/{total} "
                f"({percentual:.2f}%)",
                end='',
                flush=True
            )

        print("\nTodos os experimentos foram concluídos!")

    df_resultados = pd.DataFrame(resultados)

    df_resultados = df_resultados.sort_values(
        by='score_final',
        ascending=False,
        na_position='last'
    )

    df_resultados.to_csv(
        'resultados_dbscan.csv',
        index=False,
        float_format='%.4f'
    )

    print("\nTOP 20 RESULTADOS:")
    print(df_resultados.head(20).to_string(index=False))

    print("\nResultados salvos em resultados_dbscan.csv")

    return df_resultados

def executar_lote(experimentos):
    resultados = []

    for experimento in experimentos:
        resultados.append(executar_teste(experimento))

    return resultados