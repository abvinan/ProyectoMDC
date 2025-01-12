import streamlit as st
import pandas as pd
import numpy as np
import gdown
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split

# AUTENTICACIÓN
USER_CREDENTIALS = {"username": "admin", "password": "password123"}

# Función para manejar la autenticación
def autenticar_usuario():
    if "autenticado" not in st.session_state:
        st.session_state["autenticado"] = False

    if not st.session_state["autenticado"]:
        # Usar HTML y CSS para encapsular todos los elementos dentro de un contenedor
        st.markdown('''
            <div class="main-container">
                <div class="login-box">
                    <h1>Iniciar sesión</h1>
                    <form>
                        <!-- Campo de Usuario -->
                        <label for="username">Usuario</label>
                        <input id="username" name="username" type="text" placeholder="Ingrese su usuario">
                        
                        <!-- Campo de Contraseña -->
                        <label for="password">Contraseña</label>
                        <input id="password" name="password" type="password" placeholder="Ingrese su contraseña">
                        
                        <!-- Casilla "Recuérdame" -->
                        <div class="remember-me">
                            <input type="checkbox" id="remember" name="remember">
                            <label for="remember">Recuérdame</label>
                        </div>
                        
                        <!-- Botón de Inicio de Sesión -->
                        <button type="button" id="login-button">Iniciar Sesión</button>
                        
                        <!-- Enlace de "¿Olvidaste tu contraseña?" -->
                        <div class="extras">
                            <a href="#">¿Olvidaste tu contraseña?</a>
                        </div>
                    </form>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        # CSS para ajustar el diseño
        st.markdown('''
            <style>
                .main-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background-color: #f4f4f4;
                }
                .login-box {
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                    width: 320px;
                    text-align: left;
                }
                .login-box h1 {
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 15px;
                    color: #333;
                    text-align: center;
                }
                .login-box label {
                    font-size: 14px;
                    font-weight: bold;
                    color: #555;
                    display: block;
                    margin-bottom: 5px;
                }
                .login-box input[type="text"], 
                .login-box input[type="password"] {
                    width: 100%;
                    padding: 10px;
                    margin-bottom: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    font-size: 14px;
                    box-sizing: border-box;
                }
                .login-box input:focus {
                    border-color: #6c63ff;
                    outline: none;
                }
                .login-box .remember-me {
                    display: flex;
                    align-items: center;
                    margin-bottom: 15px;
                }
                .login-box .remember-me input {
                    margin-right: 5px;
                }
                .login-box button {
                    background-color: #6c63ff;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    font-size: 16px;
                    cursor: pointer;
                    width: 100%;
                }
                .login-box button:hover {
                    background-color: #5750d9;
                }
                .login-box .extras {
                    text-align: center;
                    margin-top: 10px;
                    font-size: 14px;
                }
                .login-box .extras a {
                    color: #6c63ff;
                    text-decoration: none;
                }
                .login-box .extras a:hover {
                    text-decoration: underline;
                }
            </style>
        ''', unsafe_allow_html=True)

        # Lógica para autenticar con Streamlit (dentro del contenedor)
        username = st.text_input("Usuario", placeholder="Ingrese su usuario", label_visibility="collapsed")
        password = st.text_input("Contraseña", type="password", placeholder="Ingrese su contraseña", label_visibility="collapsed")
        remember_me = st.checkbox("Recuérdame")

        if st.button("Iniciar Sesión"):
            if username == "admin" and password == "password123":
                st.session_state["autenticado"] = True
                st.success("Inicio de sesión exitoso. Redirigiendo...")
            else:
                st.error("Usuario o contraseña incorrectos.")

# Inicializar estado de autenticación
if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False

# Llamar a la función de autenticación
autenticar_usuario()

# Mostrar contenido solo si está autenticado
if st.session_state["autenticado"]:
    st.title("Bienvenido a la Aplicación de Recomendación")
    st.write("¡La aplicación está funcionando correctamente!")








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
