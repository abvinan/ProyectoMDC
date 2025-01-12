import streamlit as st
import pandas as pd
import numpy as np
import gdown
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split

# AUTENTICACIÓN
USER_CREDENTIALS = {"username": "admin", "password": "password123"}

# CSS para ajustar el diseño de la ventana de inicio de sesión
st.markdown("""
    <style>
    body {
        margin: 0;
        padding: 0;
        background-color: #f4f4f4;
    }
    .main-container {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        height: 4vh;
        margin-top: 4vh;
    }
    .login-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        width: 320px;
        text-align: left;
        margin-left: 180 px;
    }
    .login-box h1 {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #333;
        text-align: center;
    }
    .login-box label {
        font-size: 22px !important;
        font-weight: bold;
        color: #555;
        display: block;
        margin-bottom: 8px;
    }
    .login-box input[type="text"], 
    .login-box input[type="password"] {
        width: 100%;
        padding: 10px;
        margin-bottom: 15px;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-size: 18px;
        box-sizing: border-box;
    }
    .login-box input:focus {
        border-color: #6c63ff;
        outline: none;
    }
    .login-box button {
        background-color: #6c63ff;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-size: 18px;
        cursor: pointer;
        width: 100%;
    }
    .login-box button:hover {
        background-color: #5750d9;
    }
    .login-box .extras {
        text-align: center;
        margin-top: 10px;
        font-size: 18px;
    }
    .login-box .extras a {
        color: #6c63ff;
        text-decoration: none;
    }
    .login-box .extras a:hover {
        text-decoration: underline;
    }
    .login-box .remember-me {
        display: flex;
        align-items: center;
        font-size: 18px;
    }
    .login-box .remember-me input {
        margin-right: 5px;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <div class="login-box">
        <label for="username" style="font-size: 22px; font-weight: bold; color: #555;">Usuario</label>
        <input id="username" type="text" placeholder="Ingrese su usuario" style="width: 100%; padding: 10px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 5px; font-size: 18px;">
        <label for="password" style="font-size: 22px; font-weight: bold; color: #555;">Contraseña</label>
        <input id="password" type="password" placeholder="Ingrese su contraseña" style="width: 100%; padding: 10px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 5px; font-size: 18px;">
    </div>
""", unsafe_allow_html=True)


# Función para manejar la autenticación
def autenticar_usuario():
    if "autenticado" not in st.session_state:
        st.session_state["autenticado"] = False

    if not st.session_state["autenticado"]:
        # Contenedor principal
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown('<h1>Iniciar sesión</h1>', unsafe_allow_html=True)

        # Capturar credenciales de usuario y contraseña
        username = st.text_input("Usuario", placeholder="Ingrese su usuario")
        password = st.text_input("Contraseña", type="password", placeholder="Ingrese su contraseña")

        # Casilla de "Recuérdame"
        remember_me = st.checkbox("Recuérdame")

        # Botón para iniciar sesión
        if st.button("Iniciar Sesión"):
            if username == USER_CREDENTIALS["username"] and password == USER_CREDENTIALS["password"]:
                st.session_state["autenticado"] = True
                st.success("Inicio de sesión exitoso. Redirigiendo...")
            else:
                st.error("Usuario o contraseña incorrectos.")

        # Enlace de "¿Olvidaste tu contraseña?"
        st.markdown('<div class="extras"><a href="#">¿Olvidaste tu contraseña?</a></div>', unsafe_allow_html=True)

        # Cerrar contenedores
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


  



# Función para cargar datos
@st.cache_data
def cargar_datos():
    url = 'https://drive.google.com/uc?id=1NmAZBoSj8YqWFbypAm8HYMj2YHbRyggT'
    output = 'datos.csv'
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

@st.cache_data
def cargar_ventas_mensuales():
    url = 'https://drive.google.com/uc?id=1-21lc0LEqQLeph9YmnqIv5dhnDMzV15q'
    output = 'ventas_mensuales.csv'
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

# Función para filtrar productos por categoría seleccionada
def filtrar_por_categoria(df, categoria_seleccionada):
    seccion = secciones.get(categoria_seleccionada)  # Buscar la sección según la categoría
    if seccion is not None:
        return df[df['SECCION'] == seccion]  # Filtrar por columna 'SECCION'
    else:
        return pd.DataFrame()  # Retornar un DataFrame vacío si no hay coincidencias

def sistema_recomendacion():
    st.title("Bienvenido a la Aplicación de Recomendación")
    st.write("¡La aplicación está funcionando correctamente!")

    # Menú lateral
    menu_seleccion = st.sidebar.radio(
        "Seleccione una ventana:", 
        ["Seleccionar Productos", "Recomendaciones", "Resumen de Combos Seleccionados"]
    )
    
    # Ventana 1: Selección de Productos
    if menu_seleccion == "Seleccionar Productos":
        st.header("Selecciona los Productos para Recomendación")
        categoria_seleccionada = st.selectbox("Seleccione una Categoría", list(secciones.keys()))
        df_categoria = filtrar_por_categoria(df, categoria_seleccionada)
        st.session_state['df_categoria'] = df_categoria
        subcategorias_disponibles = df_categoria['DESC_CLASE'].unique()
        subcategoria_seleccionada = st.selectbox("Seleccione una Subcategoría", subcategorias_disponibles)
        productos_disponibles = df_categoria[df_categoria['DESC_CLASE'] == subcategoria_seleccionada]['DESC_PRODUCTO'].unique()
        productos_seleccionados = st.multiselect("Seleccione hasta 4 productos:", productos_disponibles, max_selections=4)
        st.session_state.productos_seleccionados = productos_seleccionados

    # Ventana 2: Mostrar Combos Recomendados
    elif menu_seleccion == "Recomendaciones":
        st.header("Combos Recomendados")
        if 'productos_seleccionados' in st.session_state and st.session_state.productos_seleccionados:
            df_categoria = st.session_state['df_categoria']
            productos_seleccionados_ids = [df[df['DESC_PRODUCTO'] == nombre]['COD_PRODUCTO'].values[0] for nombre in st.session_state.productos_seleccionados]
            # Aquí sigue el resto del código para calcular las recomendaciones
            st.write("Recomendaciones generadas...")

    # Ventana 3: Resumen de Combos Seleccionados
    elif menu_seleccion == "Resumen de Combos Seleccionados":
        st.header("Resumen de Combos Seleccionados")
        # Aquí sigue el resto del código para mostrar el resumen
        st.write("Resumen generado...")

# Lógica principal: decide si mostrar la autenticación o la aplicación
if "autenticado" not in st.session_state or not st.session_state["autenticado"]:
    autenticar_usuario()
else:
    # Asegurarse de cargar los datos antes de ejecutar el sistema
    if "df" not in st.session_state:
        st.session_state["df"] = cargar_datos()
    if "df_ventas" not in st.session_state:
        st.session_state["df_ventas"] = cargar_ventas_mensuales()

    # Variables globales necesarias
    secciones = {
        'Limpieza del Hogar': 14,
        'Cuidado Personal': 16,
        'Bebidas': 24,
        'Alimentos': 25
    }

    df = st.session_state["df"]
    df_ventas = st.session_state["df_ventas"]

    sistema_recomendacion()  # Inicia la aplicación
