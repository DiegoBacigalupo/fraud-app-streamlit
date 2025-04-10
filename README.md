## Versión 3: FraudGuard AI - Detección de Fraudes (Versión 3)

## Descripción

FraudGuard AI es una aplicación de Streamlit diseñada para detectar transacciones fraudulentas utilizando modelos de Machine Learning e Inteligencia Artificial. La aplicación permite a los usuarios entrenar modelos personalizados a partir de sus propios datos y analizar nuevas transacciones para identificar posibles fraudes.
El modelo de ML actúa como un sistema de detección, identificando transacciones sospechosas y notificando al Agente de IA. El Agente de IA sirve como intermediario, presentando esta información a los usuarios y permitiéndoles tomar decisiones informadas. El Agente de IA entonces ejecuta estas decisiones, interactuando con otros sistemas según sea necesario.

## Funcionalidades

* **Entrenamiento de Modelos Personalizados:** Los usuarios pueden subir sus propios datasets para entrenar modelos de Machine Learning. Se admiten archivos CSV.
* **Selección de Características y Objetivo:** Los usuarios pueden seleccionar qué columnas de sus datos se utilizarán como características y cuál será la columna objetivo.
* **Recomendación de Modelo:** La aplicación puede recomendar el modelo de Machine Learning más adecuado para los datos del usuario, utilizando la API de OpenAI.
* **Soporte para Múltiples Modelos:** Los usuarios pueden elegir entre varios modelos de Machine Learning populares, incluyendo Random Forest, Gradient Boosting, Regresión Logística, y más.
* **Análisis de Transacciones:** Los usuarios pueden subir un dataset de transacciones para ser analizado por el modelo entrenado. La aplicación marcará las transacciones que se consideren fraudulentas.
* **Explicaciones Detalladas:** Para cada transacción marcada como fraudulenta, la aplicación proporciona una explicación detallada de los factores que contribuyeron a la decisión, utilizando la API de OpenAI.
* **Métricas de Rendimiento:** Se muestran métricas de rendimiento del modelo, como accuracy, precision, recall, F1-score y AUC-ROC.
* **Matriz de Confusión:** Se visualiza la matriz de confusión para evaluar el rendimiento del modelo.

## Cómo Funciona

1.  **Entrenamiento del Modelo:**
    * El usuario sube un dataset en formato CSV.
    * El usuario selecciona la columna objetivo y las columnas de características.
    * Opcionalmente, la aplicación recomienda un modelo de Machine Learning.
    * El usuario selecciona un modelo y lo entrena.
2.  **Análisis de Transacciones:**
    * El usuario sube un dataset de transacciones en formato CSV.
    * La aplicación utiliza el modelo entrenado para predecir qué transacciones son fraudulentas.
    * Se muestran las transacciones fraudulentas detectadas.
    * El usuario puede solicitar una explicación detallada para cada transacción fraudulenta.


  ### Comunicación con el Agente de IA

**1. Comunicación del Modelo ML con el Agente de IA**

* **Detección de Transacciones Sospechosas:** El modelo de ML es el componente principal responsable de analizar las transacciones y determinar su nivel de riesgo. Cuando el modelo detecta una transacción que supera un umbral de riesgo predefinido, se activa una alerta.
* **Envío de Alertas al Agente de IA:** En lugar de tomar acción directa (como bloquear la transacción), el modelo de ML se comunica con el Agente de IA. Envía los detalles de la transacción sospechosa, incluyendo:
    * Identificador de la transacción
    * Usuario involucrado
    * Monto de la transacción
    * Características extraídas por el modelo
    * Puntaje de riesgo calculado
* **Formato de los Datos:** La comunicación entre el modelo de ML y el Agente de IA se realiza mediante un formato de datos estandarizado (por ejemplo, JSON). Esto asegura que ambos componentes puedan entender e interpretar la información correctamente.
* **API de Comunicación:** El modelo de ML utiliza una API (Interfaz de Programación de Aplicaciones) para enviar las alertas al Agente de IA. Esta API define los puntos finales (endpoints), los métodos de solicitud (por ejemplo, POST), y el formato de los datos.

**2. Comunicación del Usuario con el Agente de IA**

* **Interfaz de Usuario (UI):** Los usuarios interactúan con el Agente de IA a través de una interfaz de usuario, que puede ser una aplicación web, una aplicación móvil, o un sistema de mensajería.
* **Recepción de Alertas:** El Agente de IA presenta las alertas de transacciones sospechosas a los usuarios a través de la UI. La información mostrada incluye los detalles proporcionados por el modelo de ML.
* **Acciones del Usuario:** Los usuarios pueden realizar varias acciones en respuesta a una alerta, tales como:
    * **Aprobar la transacción:** Si el usuario determina que la transacción es legítima.
    * **Rechazar la transacción:** Si el usuario confirma que la transacción es fraudulenta.
    * **Solicitar más información:** Si el usuario necesita más detalles para tomar una decisión.
    * **Investigar al usuario:** Si el usuario considera que el usuario involucrado es sospechoso.
* **Envío de Acciones al Agente de IA:** Las acciones que el usuario realiza en la UI se envían al Agente de IA.
* **Lógica del Agente de IA:** El Agente de IA procesa las acciones del usuario y toma las medidas necesarias. Por ejemplo, si el usuario rechaza una transacción, el Agente de IA puede bloquear la transacción, notificar al sistema de procesamiento de pagos, y registrar la acción para fines de auditoría.
* **Confirmación al Usuario:** El Agente de IA envía una confirmación al usuario a través de la UI para indicar que su acción ha sido procesada.

