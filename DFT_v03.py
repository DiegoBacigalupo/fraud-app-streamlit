import streamlit as st
import openai
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Inicializar la API Key de OpenAI
openai.api_key = st.session_state.get("OPENAI_API_KEY")

if 'modelo_ml' not in st.session_state:
    st.session_state['modelo_ml'] = None
if 'feature_columns_entrenamiento' not in st.session_state:
    st.session_state['feature_columns_entrenamiento'] = []
if 'transacciones_fraudulentas' not in st.session_state:
    st.session_state['transacciones_fraudulentas'] = {}
if 'explicaciones' not in st.session_state:
    st.session_state['explicaciones'] = {}
if 'mostrar_transacciones' not in st.session_state:
    st.session_state['mostrar_transacciones'] = False
if 'importancia_caracteristicas' not in st.session_state:
    st.session_state['importancia_caracteristicas'] = {}
if 'modelo_seleccionado' not in st.session_state:
    st.session_state['modelo_seleccionado'] = None
if 'recomendacion_modelo' not in st.session_state:
    st.session_state['recomendacion_modelo'] = None
if 'justificacion_modelo' not in st.session_state:
    st.session_state['justificacion_modelo'] = None
if 'datos_entrenamiento' not in st.session_state:
    st.session_state['datos_entrenamiento'] = None
if 'target_column_entrenamiento' not in st.session_state:
    st.session_state['target_column_entrenamiento'] = None


def obtener_recomendacion_modelo(dataframe):
    prompt = f"""
    Descripción del problema: El objetivo es detectar transacciones fraudulentas en un conjunto de datos.

    Información del dataset:
    {dataframe.head().to_string()}

    Columnas y tipos de datos:
    {dataframe.info()}

    Tarea:
    1. Analiza el dataset proporcionado.
    2. Recomienda el modelo de Machine Learning más adecuado para este problema de clasificación binaria.
    3. Justifica brevemente tu elección (máximo 2 oraciones).
    4. Devuelve solo el nombre del modelo y la justificación, separados por un punto y coma. Por ejemplo: "Random Forest; Es un buen modelo para problemas de clasificación y puede manejar grandes conjuntos de datos".
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "Eres un experto en detección de fraudes y análisis de datos."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
        )
        respuesta = response.choices[0].message.content.strip()
        return respuesta
    except Exception as e:
        error_message = f"Error al obtener recomendación del modelo: {e}"
        st.error(error_message)
        print(error_message)
        return "Error; No se pudo obtener la recomendación del modelo."


def entrenar_modelo(modelo_seleccionado, X_train, y_train):
    modelos = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Neural Network": MLPClassifier(random_state=42),
    }
    if modelo_seleccionado in modelos:
        try:
            modelo = modelos[modelo_seleccionado]
            modelo.fit(X_train, y_train)
            return modelo
        except Exception as e:
            st.error(f"Error al entrenar el modelo {modelo_seleccionado}: {e}")
            return None
    else:
        st.error(f"Modelo no soportado: {modelo_seleccionado}")
        return None


def analizar_transaccion(transaccion, modelo_ml, feature_columns_entrenamiento):
    try:
        transaccion_df = pd.DataFrame([transaccion])
        transaccion_X = transaccion_df[feature_columns_entrenamiento]
        prediccion = modelo_ml.predict(transaccion_X)[0]
        return prediccion
    except Exception as e:
        st.error(f"Error al analizar la transacción: {e}")
        return 0


st.title("FraudGuard AI - Detección de Fraudes")

# Solicitar la API Key de OpenAI al usuario
openai_api_key = st.text_input(
    "Ingresa tu API Key de OpenAI:",
    type="password",
    placeholder="sk-...",
    help="Obtén tu API Key de OpenAI desde https://platform.openai.com/account/api-keys",
)

# Guardar la API Key en la sesión de Streamlit
if openai_api_key:
    openai.api_key = openai_api_key
    st.session_state["OPENAI_API_KEY"] = openai_api_key
    try:
        openai.models.list()
        st.success("API Key de OpenAI configurada correctamente.")
    except Exception as e:
        st.error(
            f"Error: La API Key de OpenAI no es válida: {e}. Por favor, verifica tu API Key.")
        st.stop()
else:
    st.warning(
        "Por favor, ingresa tu API Key de OpenAI para continuar. La aplicación no funcionará correctamente sin ella.")
    st.stop()

# Sección de "Cómo utilizar la app"
st.sidebar.header("Cómo utilizar la app")
st.sidebar.markdown(
    """
    **Objetivo de la app:** Esta aplicación utiliza inteligencia artificial para detectar transacciones fraudulentas en conjuntos de datos financieros.

    **Para qué sirve:** Permite a los usuarios entrenar un modelo de machine learning con sus propios datos históricos de transacciones y luego utilizar ese modelo para analizar nuevas transacciones e identificar posibles fraudes.

    ---
    **Paso 1: Ingresar API Key de OpenAI**
    - Ingresa tu API Key de OpenAI en el campo de texto principal.

    **Paso 2: Entrenar el modelo**
    - Sube un dataset en formato CSV que contenga la columna objetivo (la que indica si la transacción es fraudulenta o no).
    - Selecciona la columna objetivo.
    - Selecciona las columnas de características que el modelo utilizará para el entrenamiento.
    - Opcionalmente, haz clic en "Obtener recomendación de modelo" para obtener una sugerencia de modelo de ML.
    - Selecciona el modelo de ML que deseas entrenar.
    - Haz clic en "Entrenar modelo".

    **Paso 3: Analizar transferencias**
    - Sube un nuevo dataset en formato CSV para predecir fraudes. Este dataset debe contener las mismas columnas de características que usaste para el entrenamiento (pero no la columna objetivo).
    - Haz clic en "Analizar transferencias".

    **Paso 4: Ver resultados**
    - Se mostrará una lista de las transacciones detectadas como fraudulentas.
    - Para cada transacción fraudulenta, puedes hacer clic en "Ver Explicación" para obtener una interpretación de por qué se clasificó como tal.

    ---
    **Datasets de prueba:**
    - Dataset con columna objetivo: [Enlace al dataset CON columna objetivo](TU_URL_CON_OBJETIVO)
    - Dataset sin columna objetivo: [Enlace al dataset SIN columna objetivo](TU_URL_SIN_OBJETIVO)
    """
)

st.subheader("Entrenar modelo desde cero")
uploaded_file_entrenamiento = st.file_uploader(
    "Sube el dataset para entrenar el modelo (debe incluir la columna objetivo)", type=["csv"])

if uploaded_file_entrenamiento is not None:
    try:
        df_entrenamiento = pd.read_csv(uploaded_file_entrenamiento)
        st.session_state['datos_entrenamiento'] = df_entrenamiento
        target_column_entrenamiento = st.selectbox(
            "Selecciona la columna objetivo", df_entrenamiento.columns.tolist())
        st.session_state['target_column_entrenamiento'] = target_column_entrenamiento

        feature_columns_entrenamiento = st.multiselect(
            "Selecciona las columnas de características",
            df_entrenamiento.columns.tolist())
        st.session_state['feature_columns_entrenamiento'] = feature_columns_entrenamiento

        if st.button("Obtener recomendación de modelo"):
            if st.session_state['datos_entrenamiento'] is not None:
                with st.spinner("Obteniendo recomendación del modelo..."):
                    respuesta_modelo = obtener_recomendacion_modelo(
                        st.session_state['datos_entrenamiento'])
                    if "Error;" not in respuesta_modelo:
                        st.session_state['recomendacion_modelo'], st.session_state['justificacion_modelo'] = respuesta_modelo.split(
                            ";")
                        st.write(
                            f"Modelo recomendado: {st.session_state['recomendacion_modelo']}")
                        st.write(
                            f"Justificación: {st.session_state['justificacion_modelo']}")
                    else:
                        st.error(
                            "No se pudo obtener una recomendación del modelo. Por favor, verifica tu conexión a internet y tu clave de API de OpenAI.")
                        st.session_state['recomendacion_modelo'] = None
            else:
                st.warning(
                    "Por favor, sube un dataset primero para obtener la recomendación.")

        modelos_disponibles = ["Random Forest", "Gradient Boosting", "AdaBoost",
                               "Logistic Regression", "Decision Tree", "SVM", "Naive Bayes", "Neural Network"]
        st.session_state['modelo_seleccionado'] = st.selectbox(
            "Selecciona el modelo a entrenar", modelos_disponibles)

        if st.button("Entrenar modelo"):
            if target_column_entrenamiento and feature_columns_entrenamiento:
                X_entrenamiento = df_entrenamiento[feature_columns_entrenamiento]
                y_entrenamiento = df_entrenamiento[target_column_entrenamiento].astype(
                    int)
                st.write(
                    f"Entrenando modelo: {st.session_state['modelo_seleccionado']}")
                st.session_state['modelo_ml'] = entrenar_modelo(
                    st.session_state['modelo_seleccionado'], X_entrenamiento, y_entrenamiento)
                if st.session_state['modelo_ml'] is not None:
                    st.success("Modelo entrenado con éxito.")
                    st.session_state['feature_columns_entrenamiento'] = feature_columns_entrenamiento

                    if hasattr(st.session_state['modelo_ml'], 'feature_importances_'):
                        st.session_state['importancia_caracteristicas'] = dict(
                            zip(feature_columns_entrenamiento,
                                st.session_state['modelo_ml'].feature_importances_))
                    else:
                        st.session_state['importancia_caracteristicas'] = {}

                    # Mostrar métricas de rendimiento
                    y_pred_entrenamiento = st.session_state['modelo_ml'].predict(
                        X_entrenamiento)
                    st.write("Métricas de rendimiento en el conjunto de entrenamiento:")
                    st.write(f"Accuracy: {accuracy_score(y_entrenamiento, y_pred_entrenamiento)}")
                    st.write(f"Precision: {precision_score(y_entrenamiento, y_pred_entrenamiento)}")
                    st.write(f"Recall: {recall_score(y_entrenamiento, y_pred_entrenamiento)}")
                    st.write(f"F1 Score: {f1_score(y_entrenamiento, y_pred_entrenamiento)}")
                    st.write(
                        f"AUC ROC Score: {roc_auc_score(y_entrenamiento, y_pred_entrenamiento)}")

                    # Mostrar reporte de clasificación
                    st.text("Reporte de Clasificación:\n" + classification_report(
                        y_entrenamiento, y_pred_entrenamiento))

                    # Mostrar matriz de confusión
                    st.subheader("Matriz de Confusión en el Conjunto de Entrenamiento:")
                    cm = confusion_matrix(y_entrenamiento, y_pred_entrenamiento)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                    plt.xlabel("Predicho")
                    plt.ylabel("Real")
                    st.pyplot(plt)

                else:
                    st.error("No se pudo entrenar el modelo.")
            else:
                st.warning(
                    "Por favor, selecciona la columna objetivo y las columnas de características.")

    except Exception as e:
        st.error(f"Error al cargar el dataset de entrenamiento: {e}")


st.subheader("Analizar transferencias con modelo entrenado")
uploaded_file_prediccion = st.file_uploader(
    "Sube el dataset para predecir fraudes (sin la columna objetivo)",
    type=["csv"])

if uploaded_file_prediccion is not None:
    try:
        df_prediccion = pd.read_csv(uploaded_file_prediccion)
        if st.button("Analizar transferencias"):
            if st.session_state['modelo_ml'] is not None:
                try:
                    X_prediccion = df_prediccion[st.session_state['feature_columns_entrenamiento']]
                    predicciones = st.session_state['modelo_ml'].predict(
                        X_prediccion)
                    df_prediccion["prediccion_fraude"] = predicciones
                    st.session_state['transacciones_fraudulentas'] = df_prediccion[
                        df_prediccion["prediccion_fraude"] == 1].to_dict(orient='index')
                    st.session_state['mostrar_transacciones'] = True

                    # Mostrar métricas de rendimiento en el conjunto de predicción (si hay etiquetas reales disponibles)
                    if st.session_state['target_column_entrenamiento'] in df_prediccion.columns:
                        y_real_prediccion = df_prediccion[st.session_state['target_column_entrenamiento']].astype(int)
                        st.write("Métricas de rendimiento en el conjunto de predicción:")
                        st.write(
                            f"Accuracy: {accuracy_score(y_real_prediccion, predicciones)}")
                        st.write(
                            f"Precision: {precision_score(y_real_prediccion, predicciones)}")
                        st.write(
                            f"Recall: {recall_score(y_real_prediccion, predicciones)}")
                        st.write(
                            f"F1 Score: {f1_score(y_real_prediccion, predicciones)}")
                        st.write(
                            f"AUC ROC Score: {roc_auc_score(y_real_prediccion, predicciones)}")

                        # Mostrar reporte de clasificación
                        st.text("Reporte de Clasificación:\n" +
                                classification_report(y_real_prediccion, predicciones))

                        # Mostrar matriz de confusión
                        st.subheader("Matriz de Confusión en el Conjunto de Predicción:")
                        cm_prediccion = confusion_matrix(
                            y_real_prediccion, predicciones)
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(cm_prediccion, annot=True, fmt="d", cmap="Blues")
                        plt.xlabel("Predicho")
                        plt.ylabel("Real")
                        st.pyplot(plt)
                    else:
                        st.write(
                            "No se dispone de la columna objetivo en el dataset de predicción, por lo que no se pueden mostrar las métricas de rendimiento.")

                except Exception as e:
                    st.error(f"Error al predecir fraudes: {e}")
            else:
                st.warning("Por favor, reentrena el modelo primero.")
    except Exception as e:
        st.error(f"Error al cargar el dataset de predicción: {e}")


if st.session_state['mostrar_transacciones']:
    if not st.session_state['transacciones_fraudulentas']:
        st.write("No se detectaron transacciones fraudulentas.")
    else:
        st.write("Transacciones fraudulentas detectadas:")
        for index, row in st.session_state['transacciones_fraudulentas'].items():
            if row['prediccion_fraude'] == 1:
                st.write(f"Transacción {index}: Probabilidad de Fraude: ", end="")
                probabilidad_fraude = "Alta" # Esto se puede mejorar con la probabilidad real del modelo si está disponible
                st.write(probabilidad_fraude)

                if st.button("Ver Explicación", key=f"explicacion_{index}"):
                    if index not in st.session_state['explicaciones']:
                        importancia_transaccion = {k: st.session_state['importancia_caracteristicas'][k] for k in
                                                    st.session_state['feature_columns_entrenamiento']}
                        explicacion_arbol = ""
                        if hasattr(st.session_state['modelo_ml'], 'estimators_'):
                            primer_arbol = st.session_state['modelo_ml'].estimators_[0]
                            explicacion_arbol = export_text(primer_arbol,
                                                            feature_names=st.session_state['feature_columns_entrenamiento'])

                        prompt_openai = f"""
                            Analiza en detalle la siguiente transacción y determina si es fraudulenta.
                            Proporciona una explicación concisa, limitándote a un máximo de 2-3 oraciones.
                            Incluye la probabilidad de fraude (alta, media o baja) y los factores más importantes que contribuyeron a la clasificación.

                            Especificamente, considera los siguientes campos y sus valores:
                            {', '.join(st.session_state['feature_columns_entrenamiento'])}.

                            Aquí está la importancia de cada característica:
                            {importancia_transaccion}.

                            Aquí está la explicación de un árbol de decisión para esta transacción:
                            {explicacion_arbol}.

                            Datos de la transacción:
                            {row}
                            """
                        try:
                            response = openai.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "Eres un experto en detección de fraudes con un profundo conocimiento de los indicadores de riesgo, el análisis de datos transaccionales, la importancia de las características en modelos de machine learning y la lógica de los árboles de decisión. Proporciona explicaciones concisas y precisas.",
                                    },
                                    {"role": "user", "content": prompt_openai},
                                ],
                                max_tokens=200,
                            )
                            st.session_state['explicaciones'][index] = response.choices[
                                0].message.content.strip()
                        except Exception as e:
                            st.error(f"Error al obtener explicación de OpenAI: {e}")
                            st.session_state['explicaciones'][index] = "No se pudo obtener la explicación."
                    if index in st.session_state['explicaciones']:
                        st.write(st.session_state['explicaciones'][index])