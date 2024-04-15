
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Cargar los datos
# Cargar datos
data = pd.read_csv('Pasajeros_Transporte_Masivo.csv')

# Seleccionar las variables relevantes
X = data[['Pasajeros_dia_tipico_laboral', 'Pasajeros_dia_sabado', 'Pasajeros_dia_festivo', 'Dia_Semana']]
y = data['Pasajeros_dia']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'Coeficiente de determinación (R^2): {r2:.2f}')

