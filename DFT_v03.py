import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
import openai  # Para la API de OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from io import StringIO  # Para manejar datos en memoria


def obtener_recomendacion_modelo(dataframe, api_key):
    """
    Obtiene una recomendación de modelo de Machine Learning de la API de OpenAI
    basándose en un análisis del dataset proporcionado.

    Args:
        dataframe (pd.DataFrame): El dataset cargado por el usuario.
        api_key (str): La clave de la API de OpenAI.

    Returns:
        tuple: Una tupla que contiene el modelo recomendado (str) y la justificación (str).
               Devuelve (None, None) si hay un error.
    """
    openai.api_key = api_key  # Establecer la clave de la API

    # 1. Preparación de los Datos del Dataset
    try:
        # Extract relevant information from the dataset
        columnas = list(dataframe.columns)
        tipos_de_datos = [str(dataframe.dtypes[col]) for col in columnas]
        estadisticas_numericas = {}
        for col in dataframe.select_dtypes(include='number').columns:
            estadisticas_numericas[col] = {
                'media': dataframe[col].mean(),
                'desviacion_estandar': dataframe[col].std(),
                'minimo': dataframe[col].min(),
                'maximo': dataframe[col].max(),
            }
        valores_unicos_categoricas = {}
        for col in dataframe.select_dtypes(include='object').columns:
            valores_unicos_categoricas[col] = list(dataframe[col].unique())

        # Formatear los datos para el prompt
        informacion_dataset = f"Columnas: {columnas}\n"
        informacion_dataset += f"Tipos de datos: {tipos_de_datos}\n"
        informacion_dataset += "Estadísticas de columnas numéricas:\n"
        for col, stats in estadisticas_numericas.items():
            informacion_dataset += f"  - {col}: {stats}\n"
        informacion_dataset += "Valores únicos en columnas categóricas:\n"
        for col, valores in valores_unicos_categoricas.items():
            informacion_dataset += f"  - {col}: {valores}\n"

    except Exception as e:
        print(f"Error al preparar los datos del dataset: {e}")
        return None, None

    # 2. Creación del Prompt para la API de OpenAI
    prompt = f"""
    Descripción del problema: El objetivo es detectar transacciones fraudulentas en un conjunto de datos.

    Información del dataset:
    {informacion_dataset}

    Modelos de Machine Learning disponibles:
    - Regresión Logística
    - SVM (Support Vector Machine)
    - Redes Neuronales
    - Random Forest

    Tarea:
    1. Analiza el dataset proporcionado.
    2. Recomienda uno de los modelos de Machine Learning disponibles que sea más adecuado para este dataset y el problema de detección de fraudes.
    3. Proporciona una breve justificación de por qué el modelo recomendado es el más adecuado.

    Respuesta:
    """

    # 3. Llamada a la API de OpenAI
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",  # O un modelo similar
            prompt=prompt,
            temperature=0.5,  # Ajustado para respuestas más deterministas
            max_tokens=250,  # Aumentado para permitir justificaciones más extensas
        )
        respuesta_openai = response.choices[0].text.strip()
    except Exception as e:
        print(f"Error al llamar a la API de OpenAI: {e}")
        return None, None

    # 4. Procesamiento de la Respuesta de la API
    try:
        # Extract the recommended model and justification (more robust parsing)
        lineas = respuesta_openai.split('\n')
        modelo_recomendado = None
        justificacion = ""
        encontrado_modelo = False

        for linea in lineas:
            linea = linea.strip()
            if not encontrado_modelo and "Modelo recomendado:" in linea:
                partes = linea.split("Modelo recomendado:")
                if len(partes) > 1:
                    modelo_recomendado = partes[1].strip().replace(".", "")  # Remove trailing period
                    encontrado_modelo = True
            elif encontrado_modelo:
                justificacion += linea + " "

        # Check if the model is valid
        if not modelo_recomendado or modelo_recomendado not in ["Regresión Logística", "SVM", "Redes Neuronales", "Random Forest"]:
            print(f"La API de OpenAI no recomendó un modelo válido. Respuesta de la API: {respuesta_openai}")
            return None, None

    except Exception as e:
        print(f"Error al procesar la respuesta de la API: {e}")
        return None, None

    # 5. Presentación de la Recomendación al Usuario (Esto se hará en la interfaz de usuario)
    return modelo_recomendado, justificacion


def entrenar_modelo(modelo_seleccionado, X_train, y_train):
    """
    Entrena el modelo de Machine Learning seleccionado.

    Args:
        modelo_seleccionado (str): El nombre del modelo a entrenar.
        X_train (pd.DataFrame): Los datos de entrenamiento.
        y_train (pd.Series): Las etiquetas de entrenamiento.

    Returns:
        object: El modelo entrenado. Devuelve None si hay un error.
    """
    try:
        if modelo_seleccionado == "Regresión Logística":
            modelo = LogisticRegression(solver='liblinear', random_state=42)  # Especificar solver
        elif modelo_seleccionado == "SVM":
            modelo = SVC(probability=True, random_state=42)  # probability=True para predict_proba
        elif modelo_seleccionado == "Redes Neuronales":
            modelo = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        elif modelo_seleccionado == "Random Forest":
            modelo = RandomForestClassifier(random_state=42)
        else:
            print(f"Modelo no soportado: {modelo_seleccionado}")
            return None

        modelo.fit(X_train, y_train)
        return modelo
    except Exception as e:
        print(f"Error al entrenar el modelo {modelo_seleccionado}: {e}")
        return None



def realizar_predicciones(modelo_entrenado, X_test):
    """
    Realiza predicciones con el modelo entrenado.

    Args:
        modelo_entrenado (object): El modelo entrenado.
        X_test (pd.DataFrame): Los datos de prueba.

    Returns:
        pd.Series: Las predicciones. Devuelve None si hay un error.
    """
    try:
        if modelo_entrenado:
            return modelo_entrenado.predict(X_test)
        else:
            return None
    except Exception as e:
        print(f"Error al realizar predicciones: {e}")
        return None
def analizar_transaccion_openai(transaccion, modelo_seleccionado, api_key):
    """
    Analiza una transacción de alto riesgo con la API de OpenAI.

    Args:
        transaccion (pd.Series): Los datos de la transacción de alto riesgo.
        modelo_seleccionado (str): El modelo de ML utilizado.
        api_key (str): Clave de la API de OpenAI

    Returns:
        str: El análisis de la IA.
    """
    openai.api_key = api_key
    try:
        prompt = f"""
            Descripción del problema: Determinar si una transacción es fraudulenta.

            Datos de la transacción:
            {transaccion.to_string()}

            Modelo de Machine Learning utilizado: {modelo_seleccionado}

            Tarea:
            1. Analiza los datos de la transacción.
            2. Indica si la transacción es de alto riesgo y explica los factores que contribuyen a tu evaluación.
            3. Sé conciso en tu respuesta.

            Respuesta:
            """
        response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                temperature=0.5,
                max_tokens=150,
            )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error al analizar transacción con OpenAI: {e}")
        return "No se pudo analizar la transacción."

def calcular_metricas(y_true, y_pred, modelo_entrenado, X_test):
    """
    Calcula y muestra varias métricas de rendimiento del modelo.

    Args:
        y_true (pd.Series): Valores reales.
        y_pred (pd.Series): Valores predichos.
        modelo_entrenado (object): Modelo entrenado (para AUC-ROC).
        X_test (pd.DataFrame): Datos de prueba
    """
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, modelo_entrenado.predict_proba(X_test)[:, 1])

        st.subheader("Métricas de Rendimiento del Modelo")
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        st.write(f"AUC-ROC: {auc_roc:.4f}")
    except Exception as e:
        st.error(f"Error al calcular las métricas: {e}")

def main():
    st.title("FraudGuard AI - Detección de Transacciones Fraudulentas")

    # API Key
    api_key = st.text_input("Ingrese su API Key de OpenAI:", type="password")
    if not api_key:
        st.warning("Por favor, ingrese su API Key de OpenAI para utilizar todas las funcionalidades.")
        return

    # Carga de datos
    archivo_csv = st.file_uploader("Cargue su archivo CSV con datos de transacciones", type="csv")
    if archivo_csv is not None:
        try:
            # Leer el archivo CSV usando pandas
            df = pd.read_csv(archivo_csv)
        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")
            return

        # Verificar si el DataFrame está vacío
        if df.empty:
            st.error("El DataFrame está vacío. Por favor, cargue un archivo CSV con datos.")
            return

        # Separación de datos (con manejo de errores)
        if 'target' in df.columns:
            X = df.drop("target", axis=1)  # Suponiendo que 'target' es la columna objetivo
            y = df["target"]
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            except Exception as e:
                st.error(f"Error al dividir los datos: {e}")
                return
        else:
            X = df
            X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
            y_train = None #Lo inicializo para que no de error luego.


        # Obtener recomendación de OpenAI
        modelo_recomendado, justificacion_recomendacion = obtener_recomendacion_modelo(X_train, api_key)
        if modelo_recomendado is None:
            st.error("No se pudo obtener una recomendación de modelo de la API de OpenAI.  Por favor, revise su API key y el formato del archivo CSV.")
            return

        # Mostrar IU de selección de modelos
        st.subheader("Recomendación del Modelo de la IA")
        st.write(f"Modelo Recomendado: **{modelo_recomendado}**")
        with st.expander("Justificación de la Recomendación"):
            st.write(justificacion_recomendacion)

        modelos_disponibles = ["Regresión Logística", "SVM", "Redes Neuronales", "Random Forest"]
        modelo_seleccionado = st.selectbox("Seleccione un modelo", modelos_disponibles, index=modelos_disponibles.index(modelo_recomendado))

        # Entrenar modelo
        modelo_entrenado = entrenar_modelo(modelo_seleccionado, X_train, y_train)
        if modelo_entrenado is None:
            st.error(f"No se pudo entrenar el modelo seleccionado: {modelo_seleccionado}")
            return

        # Realizar predicciones
        predicciones = realizar_predicciones(modelo_entrenado, X_test)
        if predicciones is None:
            st.error("No se pudieron realizar las predicciones.")
            return

        if y_train is not None:
            calcular_metricas(y_test, predicciones, modelo_entrenado, X_test)

        # Analizar transacciones de alto riesgo (ejemplo)
        st.subheader("Análisis de Transacciones de Alto Riesgo")
        if y_train is not None: #Solo tiene sentido si hay un target.
            transacciones_alto_riesgo = X_test[predicciones == 1]  # Suponiendo que 1 indica fraude
            if transacciones_alto_riesgo.empty:
                st.write("No se detectaron transacciones de alto riesgo.")
            else:
                for index, transaccion in transacciones_alto_riesgo.iterrows():
                    analisis_ia = analizar_transaccion_openai(transaccion, modelo_seleccionado, api_key)
                    st.write(f"Transacción {index}: {analisis_ia}")
        else:
            st.write("No se puede analizar transacciones de alto riesgo ya que no se proporcionó una columna objetivo.")
        # Mostrar las predicciones
        st.subheader("Predicciones")
        st.write(predicciones)

if __name__ == "__main__":
    main()
