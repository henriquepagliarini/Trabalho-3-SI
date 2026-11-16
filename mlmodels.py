import joblib

class MLModels:

    def __init__(self, models_path):
        self.modelo_tri = joblib.load(f"{models_path}/classificador.joblib")
        self.modelo_sobr = joblib.load(f"{models_path}/regressor.joblib")

    def __prever_tri(self, data):
        return self.modelo_tri.predict(data)
    
    def __prever_sobr(self, data):
        return self.modelo_sobr.predict(data)

    def calcular(self, vitimas, data):
        vitimas = vitimas.copy()

        vitimas['tri'] = self.__prever_tri(data)
        vitimas['sobr'] = self.__prever_sobr(data)

        return vitimas
