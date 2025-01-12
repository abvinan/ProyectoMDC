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
  


# Funciones para cargar datos
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
    seccion = secciones.get(categoria_seleccionada)
    return df[df['SECCION'] == seccion]

# Obtener el top 200 productos más vendidos
def obtener_top_200_productos(df_categoria):
    if df_categoria.empty:
        return pd.DataFrame()
    else:
        top_200_productos = df_categoria.groupby('COD_PRODUCTO')['CANTIDAD'].sum().nlargest(200).index
        return df_categoria[df_categoria['COD_PRODUCTO'].isin(top_200_productos)]

# Preparar datos para entrenar el modelo ALS
def preparar_datos_para_entrenar(df):
    if len(df) > 0:
        df_train, _ = train_test_split(df, test_size=0.3, random_state=42)
        df_train_compras = df_train.groupby(['COD_FACTURA', 'COD_PRODUCTO'])['CANTIDAD'].sum().unstack().fillna(0)
        return df_train_compras
    else:
        st.error("No hay suficientes datos para dividir en entrenamiento y prueba.")
        return None

# Entrenar el modelo ALS
def entrenar_modelo_als(df_train_compras):
    if df_train_compras is not None:
        df_train_sparse = csr_matrix(df_train_compras.values)
        als_model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=30)
        als_model.fit(df_train_sparse)
        return als_model, df_train_sparse
    else:
        return None, None

# Generar recomendaciones
def generar_recomendaciones_seleccionados(df_train_compras, als_model, df_train_sparse, productos_seleccionados_ids):
    als_recommendations = {}
    for product_id in productos_seleccionados_ids:
        try:
            product_idx = df_train_compras.columns.get_loc(product_id)
            recommended_products = als_model.recommend(product_idx, df_train_sparse.T, N=6)
            recommended_products_list = [df_train_compras.columns[i] for i, _ in zip(recommended_products[0], recommended_products[1])]
            als_recommendations[product_id] = recommended_products_list
        except KeyError:
            st.warning(f"El producto con ID {product_id} no se encontró en el modelo.")
    return als_recommendations

# Configuración del sistema de recomendación
def sistema_recomendacion():
    st.title("Sistema de Recomendación")
    menu_seleccion = st.sidebar.radio("Seleccione una ventana:", ["Seleccionar Productos", "Recomendaciones", "Resumen de Combos Seleccionados"])

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

    elif menu_seleccion == "Recomendaciones":
        st.header("Combos Recomendados")
        if 'productos_seleccionados' in st.session_state and st.session_state.productos_seleccionados:
            df_categoria = st.session_state['df_categoria']
            productos_seleccionados_ids = [df[df['DESC_PRODUCTO'] == nombre]['COD_PRODUCTO'].values[0] for nombre in st.session_state.productos_seleccionados]
            df_top_200 = obtener_top_200_productos(df_categoria)
            df_train_compras = preparar_datos_para_entrenar(df_top_200)
            modelo_als, df_train_sparse = entrenar_modelo_als(df_train_compras)
            recomendaciones = generar_recomendaciones_seleccionados(df_train_compras, modelo_als, df_train_sparse, productos_seleccionados_ids)
            st.write(recomendaciones)
        else:
            st.warning("Por favor, seleccione productos en la sección anterior.")

# Lógica principal
if "autenticado" not in st.session_state or not st.session_state["autenticado"]:
    autenticar_usuario()
else:
    if "df" not in st.session_state:
        st.session_state["df"] = cargar_datos()
    if "df_ventas" not in st.session_state:
        st.session_state["df_ventas"] = cargar_ventas_mensuales()

    df = st.session_state["df"]
    secciones = {
        'Limpieza del Hogar': 14,
        'Cuidado Personal': 16,
        'Bebidas': 24,
        'Alimentos': 25
    }
    sistema_recomendacion()
