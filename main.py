import streamlit as st
import matplotlib as plt
import pandas as pd
import pydeck as pdk
import numpy as np
import seaborn as sns
import plotly.express as px
from geopy.geocoders import Nominatim
from pycaret.regression import predict_model, load_model


# Carga de datos
df = pd.read_csv('dataV2.csv')
df3 = pd.read_csv('dataV3.csv')

# Configura la página
st.set_page_config(page_title="Análisis de Influencers en Instagram", page_icon=":bar_chart:", layout="wide")

# Título de la página

st.markdown("<h1 style='color: #FF69B4;'>Análisis Profesional de Influencers en Instagram</h1>", unsafe_allow_html=True)

# Estilo personalizado para aumentar el tamaño de la letra
st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Número total de seguidores
total_seguidores = df3['Seguidores'].sum()

# Media de likes
media_likes = df3['Media_likes'].mean()

# Tasa de interacción promedio
tasa_interaccion_promedio = df3['Tasa_interacción_60d'].mean()

# Crear tres columnas
col1, col2, col3 = st.columns(3)
blue_light_color = "#2f9cb3"

with col1:
    st.markdown('<p class="big-font"> Suma de Seguidores del top 200</p>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:{blue_light_color};">{total_seguidores:,}</h1>', unsafe_allow_html=True)

with col2:
    st.markdown('<p class="big-font">Media de Likes</p>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:{blue_light_color};">{media_likes:,.2f}</h1>', unsafe_allow_html=True)

with col3:
    st.markdown('<p class="big-font">Media de tasa de Interacción (últimos 60 días)</p>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:{blue_light_color};">{tasa_interaccion_promedio:.2f}%</h1>', unsafe_allow_html=True)

# Subtítulo y descripción
st.markdown("<h2 style='text-align: ; text-decoration: underline;'>Bienvenido al Dashboard de Análisis de Influencers en Instagram</h2>", unsafe_allow_html=True)

st.write("""
Esta plataforma está diseñada para ofrecer insights y análisis detallados sobre los influencers más destacados en Instagram. 
         
Utilizamos el análisis de datos y la inteligencia artificial para proporcionar una comprensión profunda de las tendencias, el engagement y el rendimiento global de los influencers.
""")

# Espacio para otros elementos como imágenes o gráficos
st.image('https://www.coynepr.com/wp-content/uploads/2022/06/Coyne_BlogEmail_SocialMediaIndustry_SiteSlate1280x1080_opt.gif')

# Mensaje de bienvenida o descripción adicional
st.markdown("**Descubre patrones y métricas clave para  sacarle el mayor rendimiento a tu perfil de Instagram.**")

st.write("""
        Aquí podrás:
        - Analizar las 200 mejores cuentas de Instagram.
        - Comprender la relación entre la cantidad de seguidores, posts, interacciones y mucho más.
        - Comparar tu perfil con el top 200 para negociar tu caché en base a tus métricas.
        """)




# Estilización del DataFrame
styled_df = df.style.set_properties(**{
    'background-color': 'black',   # Color de fondo de las celdas
    'color': 'white',              # Color del texto
    'border-color': 'green'         # Color de los bordes
})



#################################

columnas_para_grafico = ['País', 'Usuario']
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
    st.write("Predicción:", prediction)

    # Extracción de la puntuación de influence
    puntuacion_influence = prediction.iloc[0, -1]
    return puntuacion_influence



# Utiliza el método .head() para seleccionar solo las primeras 5 filas si es necesario
df_grafico = df[columnas_para_grafico].head()
with st.expander("**1. INFORMACIÓN DE DATOS**"):
    st.markdown("<h1 style='color: #FF69B4;'>¿Qué hay en las características?</h1>", unsafe_allow_html=True)
    st.write("""
        

- Usuario: Información de usuario.
- Puntuación_influence: Se calcula en base a menciones, importancia y popularidad.
- Seguidores: Número de seguidores del usuario.
- Media_likes : Total de "Me Gusta :heart: " que el usuario ha recibido en sus publicaciones.
- Posts: Número de publicaciones que han hecho hasta ahora.
- Tasa_interacción_60d: Tasa de reacciones en 60 dias.
- Media_Likes_Post_Nuevo: Cantidad promedio de "me gusta".
- Total_Likes: Suma total de "me gusta" acumulados.
- País: País de origen del usuario.
- Idioma: Idioma principal del influencer.
- Total_interacciones: Total de reacciones. """)
    
    st.dataframe(df3)
    st.metric('Usuarios Totales', 200 )

    st.image('imagen.jpg')


with st.expander("**2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)**"):
    st.markdown("<h1 style='color: #FF69B4;'>Gráficas: 📊</h1>", unsafe_allow_html=True)
    
      # Pie Chart para la distribución de usuarios por país
    st.subheader('Gráfico de distribución de Usuarios por País')
    fig = px.pie(df3, names='País', title='Porcentaje de Usuarios por País')
    st.plotly_chart(fig)
        # Aquí estamos suponiendo que 'df' es tu DataFrame y ya está definido.
    # Obtén una lista de todos los países únicos y el rango de índices animados
    st.subheader('Gráfico de distribución de Posts por País')
    paises_unicos = df3['País'].unique()
    indices_animados = range(2)  # Por ejemplo, 10 pasos en la animación

    # Crea un nuevo DataFrame que tenga una entrada para cada país en cada índice animado
    # Inicializa el conteo de 'Posts' a 0
    df_animado = pd.DataFrame([(pais, indice, 0) for pais in paises_unicos for indice in indices_animados],
                            columns=['País', 'Índice_Animado', 'Posts'])

    # Agrupa el DataFrame original por 'País' y cuenta los 'Posts'
    conteos_originales = df3.groupby('País')['Posts'].sum().reset_index()

    # Actualiza el conteo de 'Posts' en el nuevo DataFrame con los valores del DataFrame original
    for _, row in conteos_originales.iterrows():
        df_animado.loc[(df_animado['País'] == row['País']) & (df_animado['Índice_Animado'] == indices_animados[-1]), 'Posts'] = row['Posts']

    # Ahora crea la gráfica animada usando el nuevo DataFrame
    fig = px.bar(df_animado, x='País', y='Posts', color='País',
                animation_frame='Índice_Animado',
                range_y=[0, df_animado['Posts'].max() + 5],
                title='Evolución de Posts por País')

    # Mostrar la gráfica animada en Streamlit
    st.plotly_chart(fig, use_container_width=True)
   
    st.subheader('Gráfico Top 10 Usuarios por Puntuación de Influencia')
    # Display the figure in Streamlit
    fig = px.bar(df3.nlargest(10, 'Puntuación_influence'), x='Usuario', y='Puntuación_influence', title='Top 10 Influencers')

    # Update the layout for a better visual appearance
    fig.update_layout(
        xaxis_title='Nombre de usuario',
        yaxis_title='Puntuación de influencia',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 100]),  # Assuming the score is out of 100
        xaxis=dict(tickangle=-45),
        title_font=dict(size=25, color='rosybrown'),
        title_x=0.5
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig)
    st.subheader('Gráfico Top 10 Usuarios por Posts')
    # Display the figure in Streamlit
    fig = px.bar(df3.nlargest(10, 'Posts'), x='Usuario', y='Posts', title='Top 10 Influencers')

    # Update the layout for a better visual appearance
    fig.update_layout(
        xaxis_title='Nombre de usuario',
        yaxis_title='Posts',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=False),  # Assuming the score is out of 100
        xaxis=dict(tickangle=-45),
        title_font=dict(size=25, color='rosybrown'),
        title_x=0.5
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig)

    st.subheader('Gráfico Top 10 Usuarios por Total de Likes ')
    # Display the figure in Streamlit
    fig = px.bar(df3.nlargest(10, 'Total_Likes'), x='Usuario', y='Total_Likes', title='Top 10 Influencers')

    # Update the layout for a better visual appearance
    fig.update_layout(
        xaxis_title='Nombre de usuario',
        yaxis_title='Total_Likes',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=False),  # Assuming the score is out of 100
        xaxis=dict(tickangle=-45),
        title_font=dict(size=25, color='rosybrown'),
        title_x=0.5
    )


    # Display the figure in Streamlit
    st.plotly_chart(fig)
    st.subheader('Gráfico Top 10 Usuarios por Seguidores')
    # Display the figure in Streamlit
    fig = px.bar(df3.nlargest(10, 'Seguidores'), x='Usuario', y='Seguidores', title='Top 10 Influencers')

    # Update the layout for a better visual appearance
    fig.update_layout(
        xaxis_title='Nombre de usuario',
        yaxis_title='Seguidores',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=False),  # Assuming the score is out of 100
        xaxis=dict(tickangle=-45),
        title_font=dict(size=25, color='rosybrown'),
        title_x=0.5
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig)
   ###################

    st.subheader('¿Outliers en las diferentes variables?')
    st.image('detalles_outliers.png')
    st.write("Tras observar de cerca, concluimos que por la naturaleza de los mismos no se pueden considerar outliers a reparar.")
    st.subheader('Histograma de distiribución de las distintas variables')
    st.image('graficasmulti.png')

    st.subheader('Gráfico de Usuarios por Seguidores-Posts')
    # Ejemplo de un gráfico de barras con Plotly
    fig_bar = px.bar(df3, x='Usuario', y='Seguidores', color='Posts',
                    title='Seguidores vs Puntuación de Influencia')
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader('Gráfico de Usuarios por Media de Likes y Países')
    # Ejemplo de un gráfico de dispersión con Plotly
    fig_scatter = px.scatter(df3, x='Seguidores', y='Media_likes', size='Posts',
                            color='País', hover_name='Usuario', size_max=40,
                            title='Relación entre Seguidores y Me Gusta Media')
    st.plotly_chart(fig_scatter, use_container_width=True)
    

    # Ejemplo de un mapa de calor con Plotly
    
######################################################################################################

##############################################################################################################
    fig = px.scatter_matrix(df3)

    
    st.subheader('Heatmap de Correlación entre variables ')
    # Título de la aplicación
# Suponemos que df3 es tu DataFrame y ya está definido
   # Suponemos que df3 es tu DataFrame y ya está definido
    numeric_columns = df3.select_dtypes(include='number')
    df_corr = numeric_columns.corr()

    # Generar una máscara para la parte superior derecha
    mask = np.triu(np.ones_like(df_corr, dtype=bool))

    # Aplicar la máscara para reemplazar los valores con NaN
    df_corr_masked = df_corr.mask(mask)

    # Definir una escala de colores personalizada
    custom_color_scale = [
        (0.0, '#00BFFF'),   # correlación -1
        (0.5, 'white'),  # correlación 0
        (1.0, '#FF1493')    # correlación 1
    ]

    # Crear el heatmap con Plotly, usando la data con la máscara aplicada
    fig = px.imshow(df_corr_masked,
                    text_auto=True,
                    labels=dict(x='Variable X', y='Variable Y', color='Correlación'),
                    x=df_corr.columns,
                    y=df_corr.columns,
                    aspect='auto',
                    color_continuous_scale=custom_color_scale)  # Usa la escala de color personalizada

    # Mostrar el título
    fig.update_layout(title_text='Heatmap de Correlación', title_x=0.5)

    # Mostrar el heatmap en Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Mostrar el gráfico en Streamlit
    

with st.expander("**3. TEST ESTADÍSTICO**"):
    st.markdown("<h1 style='color: #FF69B4;'> Análisis de Datos Confirmatorio (CDA)</h1>", unsafe_allow_html=True)
    
    st.write("**¿Tienen más seguidores los usuarios que están en el 10% con más total likes?.**")
    
        # Creación del gráfico de dispersión con Plotly Express
    fig = px.scatter(
        df3, x='Total_Likes', y='Seguidores',
        trendline='ols',  # OLS significa Ordinary Least Squares (Mínimos Cuadrados Ordinarios)
        title='Relación entre Total de Likes y Seguidores'
    )

    # Personalización adicional si es necesario (opcional)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
        xaxis_title='Total de Likes',
        yaxis_title='Seguidores',
        title_x=0.5,  # Centrar el título
    )
    fig.update_traces(line=dict(color='rgba(255, 20, 147, 1)'), selector=dict(type='scatter', mode='lines'))
    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)

      


    st.write("Tras realizar un test de Mann-Whitney, el p-valor obtenido (1.581540771327198e-08) indica que **rechazamos la hipótesis nula**, por lo que hay una diferencia significativa entre el 10% de usuarios con más likes y el total de seguidores")

    st.write("**¿Tienen más interacciones que están en el 10% con más posts?**")
      # Creación del gráfico de dispersión con Plotly Express
    fig = px.scatter(
        df3, x='Posts', y='Total_interacciones',
        trendline='ols',  # OLS significa Ordinary Least Squares (Mínimos Cuadrados Ordinarios)
        title='Relación entre Posts y Total de Interacciones'
    )

    # Personalización adicional si es necesario (opcional)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
        xaxis_title='Posts',
        yaxis_title='Total de Interacciones',
        title_x=0.5,  # Centrar el título
    )
    fig.update_traces(line=dict(color='rgba(255, 20, 147, 1)'), selector=dict(type='scatter', mode='lines'))

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)
    st.write("Tras realizar un test de Mann-Whitney, el p-valor obtenido (1.1394432769938303e-07) **rechazamos la hipótesis nula**, por lo que hay una diferencia significativa entre el 10% de usuarios con más posts y el total de interacciones")
 
    st.dataframe(df3)
with st.expander("**4. APP**"):
    
    st.markdown("<h1 style='color: #FF69B4;'>¿Cuál la puntuación de mi cuenta? </h1>", unsafe_allow_html=True)
    st.title(" :rocket:")
    seguidores = st.number_input("Seguidores", min_value=36700000)
    media_likes = st.number_input("Media likes", min_value=54800)
    posts = st.number_input("Posts", min_value=1)
    tasa_interaccion_60d = st.number_input("Tasa interacción 60 días", min_value=0.0)
    media_likes_post_nuevo = st.number_input("Media Likes Post Nuevo", min_value=0)
    total_likes = st.number_input("Total Likes", min_value=2900000)
    total_interacciones = st.number_input("Total interacciones", min_value=0.0)
    
    
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
    
    if st.button("¡Descubre tu puntuación!"):
        result = predict_puntuacion_influence(seguidores, media_likes, posts, tasa_interaccion_60d, media_likes_post_nuevo, total_likes, total_interacciones, país, idioma)
        st.success(f"La puntuación es: {result}")
        st.write(":smile:")
        st.image('https://media.tenor.com/WVPNXedSf_YAAAAM/im-an-influencer-kim-kardashian.gif')


with st.expander("**5. RECOMENDACIONES**"):
    
    
    st.markdown("<h1 style='color: #FF69B4;'>Recomendaciones para mejorar tu puntuación de influencer</h1>", unsafe_allow_html=True)
    

    st.markdown("""
        **1. Seguidores**: Estrategias para aumentar la visibilidad y la participación.

        **2. Tasa de Interacción en 60 días**: Interactuar con la audiencia a través de comentarios, encuestas y preguntas puede aumentar la participación.

        **3. Posts**: Publicar regularmente mantiene la presencia en la mente de los seguidores.

        **4. Total de Interacciones**: Diversificar el tipo de interacciones.

        **5. Total de Likes**: Similar a las interacciones totales, diversificar los "me gusta".

        **6. Media de Likes por Post Nuevo  y Media de Likes**: Analizar qué tipo de contenido recibe más likes y adaptar la estrategia en consecuencia.

        **7. País e Idioma Principal**: Personalizar el contenido según la ubicación y el idioma de la audiencia.


        """)

    st.image('chart.png', caption='La importancia de diferentes variables del modelo.')

    st.sidebar.header("Acerca de esta Aplicación")
    st.sidebar.info("Esta aplicación brinda recomendaciones para mejorar tu perfil de Instagram y aumentar tu influencia de manera orgánica y auténtica.")
    st.sidebar.header("Autores")  
    st.sidebar.info("""
                    [**Joseba** Moreno Iriarte](https://www.linkedin.com/in/imjoseba/)

                    [**Lorena** Marchante Rodríguez](https://www.linkedin.com/in/lorena-marchante/)
                    
                    [**Manuel** Montes Trujillo](https://www.linkedin.com/in/manuel-montes-trujillo-051217142/)
                    """)
    
    path_to_gif = 'https://hashtag.com.pk/filemanager/photos/1/617ba1e4bcfed.gif'

    # Show the GIF with a specific width
    st.sidebar.image(path_to_gif, width=250)
    
    st.sidebar.header("Fuente de los datos")
    st.sidebar.info("""Los datos utilizados en esta aplicación se obtuvieron de la plataforma [Socialbook](https://socialbook.io/instagram-channel-rank/top-200-instagrammers).
            
                    22/11/2023
                    
                    """)

st.title("Geolocalizador de Influencers en Instagram 🌐")

def get_location(country):
    geolocator = Nominatim(user_agent="streamlit_geolocator")
    location = geolocator.geocode(country)
    if location:
        return location.latitude, location.longitude
    return None, None

# Datos de ejemplo basados en tu imagen (deberás reemplazar esto con tus datos reales)
data = {
    'country': ['Portugal', 'Argentina', 'United States', 'Canada', 'India', 'Brazil', 'Spain', 'Netherlands', 'Thailand', 'Colombia', 'United Kingdom', 'South Korea', 'Italy', 'New Zealand', 'Uruguay', 'Indonesia', 'Turkey', 'Sweden', 'Egypt', 'Australia', 'Puerto Rico', 'Switzerland', 'Anguilla', 'Germany', 'China', 'Mexico', 'British Virgin Islands'],
    'num_influencers': [1, 2, 82, 26, 12, 7, 2, 1, 5, 7, 3, 3, 1, 2, 3, 11, 3, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1]
}

df = pd.DataFrame(data)
df['location'] = df['country'].apply(get_location)
df[['latitude', 'longitude']] = pd.DataFrame(df['location'].tolist(), index=df.index)

    # Configura la página de Streamlit


 


    # Asegúrate de que las coordenadas estén correctas
df = df.dropna()

    # Crear un mapa con pydeck
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=0,
        longitude=0,
        zoom=1
        ),
        layers=[
        pdk.Layer(
        "ScatterplotLayer",
                data=df,
                get_position=["longitude", "latitude"],
                get_color="[180, 0, 200, 140]",
                get_radius="num_influencers * 30000",  # El tamaño del radio puede ser una función del número de influencers
                pickable=True
            ),
        ],
    ))



# Sección de Mejoras a Futuro
with st.expander("**6. MEJORAS A FUTURO**"):
    st.markdown("<h1 style='color: #FF69B4;'>Mejoras a Futuro</h1>", unsafe_allow_html=True)

    st.markdown("""
               Este equipo es consciente de que la aplicación tiene un gran potencial de mejora. Como puntos a mejorar, destacamos los siguientes:
                - **Aumentar el conjunto de datos**: el conjunto de datos utilizado para el análisis es de 200 usuarios. Para mejorar la precisión de los resultados, se recomienda aumentar el conjunto de datos.
                - **Añadir otras métricas como**: el número de comentarios, el número de seguidos, la frecuencia de publicación, el precio por post, en otros.
                - **Utilizable por cualquier usuario**: actualmente, la aplicación es precisa únicamente para los usuarios pertenecientes al top 200.
                - **Recomendaciones personalizadas**: poder ofrecer recomendaciones específicas en base a las métricas del usuario para mejorar el perfil de Instagram. 
                - **Añadir un apartado de contacto para que los usuarios puedan ponerse en contacto con el equipo de desarrollo.**
                - **Así mismo, esta aplicación podría ser utilizada por marcas para encontrar influencers que se ajusten a sus necesidades.**
                """)
    

with st.expander("**7. AGRADECIMIENTOS**"):
    st.markdown("<h1 style='color: #FF69B4;'>¡¡A TODOS!!</h1>", unsafe_allow_html=True)

    st.markdown("""
               Nos gustaría agradecer:
                - Al equipo de [UpgradeHub](https://www.upgrade-hub.com/) (en especial a [Demetrio](https://github.com/demstalferez) y Andrés) por su dedicación y paciencia durante todo el curso.
                - Agradecer a nuestros **compañeros** de clase por su apoyo y ayuda. Nos toca volar! :airplane:
                
                """)
    
    # Crear tres columnas
    
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("""Enhorabuena a todos! :clap: y feliz analísis! :computer: :chart_with_upwards_trend:""")

    # ponemos mask.png en la col2
    with col2:
        st.image('mask.png')
