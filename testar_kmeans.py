
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import os

from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from kmeans_cluster import KMeansCluster


_vitimas = None


def inicializar_processo(vitimas):
    global _vitimas
    _vitimas = vitimas


def executar_teste(parametros):

    features, k = parametros

    # Evita que cada processo crie várias threads internas
    with threadpool_limits(limits=1):

        kmeans = KMeansCluster(
            _vitimas,
            list(features),
            K=k
        )

        kmeans.executar()

        labels = np.array(kmeans.clusters)

        # Calcula Silhouette
        silhouette = silhouette_score(
            kmeans.X,
            labels
        )

        return {
            'features': ', '.join(features),
            'k': k,
            'sse': kmeans.sse,
            'silhouette': silhouette
        }


def executar_lote(experimentos):

    resultados = []

    for experimento in experimentos:
        resultados.append(executar_teste(experimento))

    return resultados


def executar_experimentos(vitimas):

    features_disponiveis = [
        coluna
        for coluna in vitimas.select_dtypes(include=np.number).columns
        if coluna not in ['id', 'gcs', 'avpu', 'cluster']
    ]

    # Combinações com pelo menos duas features
    # e obrigatoriamente tri ou sobr
    combinacoes = [
        features
        for tamanho in range(2, len(features_disponiveis) + 1)
        for features in combinations(features_disponiveis, tamanho)
        if 'tri' in features or 'sobr' in features
    ]

    valores_k = range(3, 11)

    experimentos = [
        (features, k)
        for features in combinacoes
        for k in valores_k
    ]

    total = len(experimentos)

    n_processos = max(1, (os.cpu_count() or 2) - 1)

    # Divide os experimentos em lotes
    quantidade_lotes = n_processos * 20
    tamanho_lote = max(1, (total + quantidade_lotes - 1) // quantidade_lotes)

    lotes = [
        experimentos[i:i + tamanho_lote]
        for i in range(0, total, tamanho_lote)
    ]

    print(f"Features disponíveis: {len(features_disponiveis)}", flush=True)
    print(f"Combinações de features: {len(combinacoes)}", flush=True)
    print(f"Total de experimentos: {total}", flush=True)
    print(f"Processos utilizados: {n_processos}", flush=True)
    print(f"Quantidade de lotes: {len(lotes)}", flush=True)

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
                f"Progresso: {concluidos}/{total} "
                f"({percentual:.2f}%)",
                end='',
                flush=True
            )
        print("\nTodos os experimentos foram concluídos!")

    df_resultados = pd.DataFrame(resultados)

    df_resultados = df_resultados.sort_values(
        by='silhouette',
        ascending=False
    )

    df_resultados.to_csv(
        'resultados_kmeans.csv',
        index=False,
        float_format='%.4f'
    )

    print("\nTOP 20 RESULTADOS:")
    print(df_resultados.head(20).to_string(index=False))

    print("\nResultados salvos em resultados_kmeans.csv")

    return df_resultados
