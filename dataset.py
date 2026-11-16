import pandas as pd

class DataSet:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

        self.vitimas = self.df[self.df['id'] != -1].copy()
    
    def obter_dados_vitimas(self):
        features = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']
        
        return self.vitimas[features]