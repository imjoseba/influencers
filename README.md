# PROYECTO DE ANALÍTICA DE INFLUENCERS Y MODELO DE MACHINE LEARNING PARA CONOCER LA PUNTUACIÓN DE UNA CUENTA DE INSTAGRAM. 

![Influencer](foto2.jpg)

Este proyecto integral se enfocó en la recopilación y análisis de datos de influencers en Instagram con el objetivo de entender las tendencias y comportamientos clave. Utilizando técnicas de análisis de datos y machine learning, creamos una aplicación interactiva destinada a mejorar la experiencia en redes sociales.

Los aspectos destacados incluyen la recopilación detallada de datos de influencers, análisis exploratorio y confirmatorio, y la creación de un modelo predictivo utilizando Random Forest Regressor. Además, se ha desarrollado una aplicación interactiva con Streamlit para visualizar resultados y que ha derivado en recomendaciones prácticas para mejorar la presencia y la interacción en las redes sociales. 

## Recopilación de Datos:
Se realizó un exhaustivo proceso de webscraping para recopilar datos de influencers en Instagram, abarcando información clave como seguidores, interacciones, y características del perfil.

## Análisis de Datos:
Implementamos análisis exploratorio y confirmatorio para extraer patrones significativos. Identificamos factores de influencia y relaciones que afectan la interacción de los influencers con su audiencia.

## Machine Learning:
Desarrollamos modelos de machine learning, elegimos el uso de Random Forest Regressor para prever la puntuación de influencia de los perfiles. Estos modelos contribuyeron a comprender mejor las métricas clave de éxito.

## Desarrollo de Aplicación:
Creamos una aplicación interactiva utilizando Streamlit, proporcionando una interfaz amigable para explorar y entender los resultados del análisis. La aplicación ofrece insights valiosos sobre la estrategia de influencers y cómo mejorar la presencia en redes sociales.

## Recomendaciones Personalizadas:
Derivamos recomendaciones prácticas basadas en hallazgos estadísticos y el peso que tienen las diferentes variables sobre la puntuación de influencer según nuestro modelo de machine learning. Además de recomendaciones como la calidad del contenido, el uso estratégico de hashtags y la autenticidad en la interacción con la audiencia.

## Guía de los arhivos: 
**1_primer-tratamiento**: notebook donde cargamos dataV1.csv.  
Limpieza de columnas y filas.  
Traducción de inglés a castellano.  
Completar países faltantes, añadir idioma principal de la cuenta.  
**2_proyecto**: notebook donde se realiza el proyecto.  
Limpieza y visualización de datos.  
Tratamientos de datos.  
Análisis Exploratorio de Datos (EDA).  
Análisis de Datos Confirmatorio (CDA).  
Machine Learning.  

**dataV1.csv**: csv original generado con webscraping.  
**dataV2.csv**: generado al final de 1_primer-tratamiento.  
**dataV3.csv**: csv con datos transformados y con columna ‘Total_interacciones’.  
**dataV4.csv**: csv con datos para subir a Azure y entrenar el modelo.  
**RF_model.pkl**: entrenamiento del modelo usando Random Forest Regressor  

**punt.py**: app streamlit únicamente con la predicción de la puntuación.  
**main.py**: web streamlit con todo!  
Las fotos son para esta parte.  
