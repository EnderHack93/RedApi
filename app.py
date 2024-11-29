from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the model and scaler
MODEL_PATH = "modelo.h5"
SCALER_PATH = "scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Define the structure of the input data
class AcademicData(BaseModel):
    Edad: float
    Promedio_Final: float
    Asistencia_Total: float
    Num_Faltas: float
    Porcentaje_Inasistencia: float
    porc_Materias_Aprobadas: float
    porc_Materias_Reprobadas: float
    Genero_M: int
    Especialidad_DIBUJO: int
    Especialidad_ESCULTURA: int
    Especialidad_GRAFICAS: int
    Especialidad_PINTURA: int

# Initialize FastAPI
app = FastAPI()

# Map the classes
CLASSES = ['Activo', 'Alerta', 'Condicional', 'Excelencia', 'Recuperación', 'Reprobado']

@app.post("/predict")
async def predict(data: AcademicData):
    # Convert the input data into a NumPy array
    input_data = np.array([[data.Edad, data.Promedio_Final, data.Asistencia_Total,
                            data.Num_Faltas, data.Porcentaje_Inasistencia,
                            data.porc_Materias_Aprobadas, data.porc_Materias_Reprobadas,
                            data.Genero_M, data.Especialidad_DIBUJO,
                            data.Especialidad_ESCULTURA, data.Especialidad_GRAFICAS,
                            data.Especialidad_PINTURA]])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Perform the prediction
    prediction = model.predict(input_scaled)
    prediction_class = np.argmax(prediction, axis=1)[0]

    # Get the corresponding academic state
    estado_academico = CLASSES[prediction_class]

    # Return the response
    return {
        "estado_academico": estado_academico,
        "probabilidades": prediction.tolist()[0]  # Convert probabilities to a list
    }

# Entry point for running the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
