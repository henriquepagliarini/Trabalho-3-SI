from dataset import DataSet
from mlmodels import MLModels

def main():
    dataset = DataSet('./datasets/map.csv')
    data_vict = dataset.obter_dados_vitimas()
    
    models = MLModels("models")
    vitimas = models.calcular(dataset.vitimas, data_vict)
    
    print(vitimas[['id', 'tri', 'sobr']])
    vitimas.to_csv("datasets/vitimas_classificadas.csv", index=False)
    
if __name__ == '__main__':
    main()