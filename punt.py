import streamlit as st
import pandas as pd
from pycaret.regression import predict_model, load_model

# Cargo el modelo previamente entrenado
model = load_model('RF_model')

# Función para hacer la predicción
def predict_puntuacion_influence(seguidores, media_likes, posts, tasa_interaccion_60d, media_likes_post_nuevo, total_likes, total_interacciones, país, idioma):
    input_data = {
        'Seguidores': seguidores,
        'Media_likes': media_likes,
        'Posts': posts,
        'Tasa_interacción_60d': tasa_interaccion_60d,
        'Media_Likes_Post_Nuevo': media_likes_post_nuevo,
        'Total_Likes': total_likes,
        'Total_interacciones': total_interacciones,
        'País': país,
        'Idioma_principal': idioma
    }

    # Crear dataframe con los datos de entrada
    input_df = pd.DataFrame([input_data])

    # Uso del modelo para hacer una predicción
    prediction = predict_model(model, data=input_df)

    # Imprimir la predicción
    st.write("Predicción:", prediction)

    # Extracción de la puntuación de influence
    puntuacion_influence = prediction.iloc[0, -1]

    return puntuacion_influence

# "Front-end" de la aplicación
st.title("¿Cuál es la puntuación de mi cuenta?")

# Introducir los datos de entrada
seguidores = st.number_input("Seguidores", min_value=36700000)
media_likes = st.number_input("Media likes", min_value=54800)
posts = st.number_input("Posts", min_value=1)
tasa_interaccion_60d = st.number_input("Tasa interacción 60 días", min_value=0.0)
media_likes_post_nuevo = st.number_input("Media Likes Post Nuevo", min_value=0)
total_likes = st.number_input("Total Likes", min_value=2900000)
total_interacciones = st.number_input("Total interacciones", min_value=0.0)
# Añadimos las opciones de 'País' e 'Idioma_principal
país_options = ['Portugal', 'Argentina', 'Estados Unidos', 'Canada', 'India',
                'Brasil', 'España', 'Holanda', 'Francia', 'Tailandia', 'Colombia',
                'Reino Unido', 'Corea del Sur', 'Italia', 'Nueva Zelanda',
                'Uruguay', 'Indonesia', 'Turquía', 'Suecia', 'Egipto', 'Australia',
                'Puerto Rico', 'Suiza', 'Costa de Marfil', 'Anguilla', 'Alemania',
                'China', 'Mexico', 'Islas Vírgenes Británicas']
país = st.selectbox("País", país_options)

idioma_options = ['Inglés', 'Español', 'Portugés', 'Francés', 'Indonesio',
                  'Italiano', 'Turco', 'Coreano']
idioma = st.selectbox("Idioma principal", idioma_options)

# Botón de predicción
if st.button("¡Descubre tu puntuación!"):
    result = predict_puntuacion_influence(seguidores, media_likes, posts, tasa_interaccion_60d, media_likes_post_nuevo, total_likes, total_interacciones, país, idioma)
    st.success(f"La puntuación es: {result}")
