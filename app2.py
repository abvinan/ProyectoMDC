import streamlit as st
import pandas as pd
import numpy as np
import gdown
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split
import streamlit as st

# AUTENTICACIÓN
USER_CREDENTIALS = {"username": "admin", "password": "password123"}

# CSS para ajustar la posición y tamaño del cuadro
st.markdown("""
    <style>
    body {
        margin: 300;
        padding: 300;
        background-color: #f4f4f4;
    }
    .main-container {
        display: flex;
        justify-content: flex-start; /* Cambia el centrado horizontal a la izquierda */
        align-items: flex-start; /* Centrado vertical en la parte superior */
        height: 100vh; /* Altura de la pantalla */
        margin-top: 4vh; /* Ajusta el desplazamiento desde arriba */
        margin-left: 20px; /* Aumenta el margen izquierdo */
}
    .login-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        width: 320px; /* Tamaño fijo para evitar crecimiento */
        text-align: left; /* Texto alineado a la izquierda */
    }
    .login-box h1 {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #333;
        text-align: center; /* Centrar el título */
    }
    .login-box label {
        font-size: 16px;
        font-weight: bold;
        display: block;
        margin-bottom: 8px;
        color: #555;
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
        font-size: 14px;
        margin-top: 10px;
        text-align: center;
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
        font-size: 14px;
    }
    .login-box .remember-me input {
        margin-right: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Función para manejar la autenticación
def autenticar_usuario():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('''
        <div class="login-box">
            <h1>Iniciar sesión</h1>
            <form>
                <label for="username">Usuario</label>
                <input id="username" type="text" placeholder="Ingrese su usuario">
                <label for="password">Contraseña</label>
                <input id="password" type="password" placeholder="Ingrese su contraseña">
                <div class="remember-me">
                    <input type="checkbox" id="remember">
                    <label for="remember">Recuérdame</label>
                </div>
                <button type="submit">Iniciar Sesión</button>
                <div class="extras">
                    <a href="#">¿Olvidaste tu contraseña?</a>
                </div>
            </form>
        </div>
    ''', unsafe_allow_html=True)
    return st.session_state.get("autenticado", False)

# Inicializar estado de autenticación
if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False

# Llamar a la función de autenticación
autenticado = autenticar_usuario()

# Detener la aplicación si no está autenticado
if not autenticado:
    st.stop()

# Código principal de la aplicación
st.title("Bienvenido a la Aplicación de Recomendación")
st.write("¡La aplicación está funcionando correctamente!")












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

# Cargar datos
df = cargar_datos()
df_ventas = cargar_ventas_mensuales()

# Diccionario para mapear categorías con sus valores de sección
secciones = {
    'Limpieza del Hogar': 14,
    'Cuidado Personal': 16,
    'Bebidas': 24,
    'Alimentos': 25
}

# Función para filtrar productos por categoría seleccionada
def filtrar_por_categoria(df, categoria_seleccionada):
    seccion = secciones.get(categoria_seleccionada)
    return df[df['SECCION'] == seccion]

# Obtener el top 200 productos más vendidos de la categoría seleccionada
def obtener_top_200_productos(df_categoria):
    if df_categoria.empty:
        return pd.DataFrame()
    else:
        top_200_productos = df_categoria.groupby('COD_PRODUCTO')['CANTIDAD'].sum().nlargest(200).index
        return df_categoria[df_categoria['COD_PRODUCTO'].isin(top_200_productos)]

# Preparar datos para entrenar el modelo ALS
def preparar_datos_para_entrenar(df):
    if len(df) > 0:
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
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

# Generar recomendaciones para los productos seleccionados
def generar_recomendaciones_seleccionados(df_train_compras, als_model, df_train_sparse, productos_seleccionados_ids):
    als_recommendations = {}
    for product_id in productos_seleccionados_ids:
        try:
            product_idx = df_train_compras.columns.get_loc(product_id)
            recommended_products = als_model.recommend(product_idx, df_train_sparse.T, N=6, filter_already_liked_items=False)
            recommended_products_list = [df_train_compras.columns[i] for i, _ in zip(recommended_products[0], recommended_products[1]) if df_train_compras.columns[i] != product_id]
            als_recommendations[product_id] = recommended_products_list[:5]
        except KeyError:
            st.warning(f"El producto con ID {product_id} no se encontró en el modelo.")
    return als_recommendations

# Configuración de la aplicación Streamlit
menu_seleccion = st.sidebar.radio("Seleccione una ventana:", ["Seleccionar Productos", "Recomendaciones", "Resumen de Combos Seleccionados"])

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
        df_top_200 = obtener_top_200_productos(df_categoria)
        df_train_compras = preparar_datos_para_entrenar(df_top_200)
        modelo_als, df_train_sparse = entrenar_modelo_als(df_train_compras)
        recomendaciones = generar_recomendaciones_seleccionados(df_train_compras, modelo_als, df_train_sparse, productos_seleccionados_ids)
        combos = []
        for producto_id in productos_seleccionados_ids:
            descripcion_a = df[df['COD_PRODUCTO'] == producto_id]['DESC_PRODUCTO'].values[0]
            for recomendacion_id in recomendaciones.get(producto_id, []):
                descripcion_b = df[df['COD_PRODUCTO'] == recomendacion_id]['DESC_PRODUCTO'].values[0]
                precio_a = df[df['COD_PRODUCTO'] == producto_id]['VALOR_PVSI'].values[0]
                precio_b = df[df['COD_PRODUCTO'] == recomendacion_id]['VALOR_PVSI'].values[0]
                costo_a = df[df['COD_PRODUCTO'] == producto_id]['COSTO'].values[0]
                costo_b = df[df['COD_PRODUCTO'] == recomendacion_id]['COSTO'].values[0]
                precio_combo = precio_a + precio_b
                margen_combo = round(((precio_combo - (costo_a + costo_b)) / precio_combo) * 100, 2)
                combos.append({
                    'Producto A': descripcion_a,
                    'Producto B': descripcion_b,
                    'Precio Combo': f"${precio_combo:.2f}",
                    'Margen Combo': f"{margen_combo}%"
                })
        df_combos = pd.DataFrame(combos)
        st.table(df_combos)
        seleccion_indices = st.multiselect("Seleccione los índices de los combos que desea considerar:", df_combos.index.tolist())
        st.session_state.combos_seleccionados = df_combos.loc[seleccion_indices]
# Ventana 3: Resumen de Combos Seleccionados
elif menu_seleccion == "Resumen de Combos Seleccionados":
    st.header("Resumen de Combos Seleccionados")
    
    if 'combos_seleccionados' in st.session_state and not st.session_state.combos_seleccionados.empty:
        resumen = []
        for _, row in st.session_state.combos_seleccionados.iterrows():
            producto_a = row['Producto A']
            producto_b = row['Producto B']
            
            # Obtener los códigos de producto correspondientes
            try:
                producto_a_id = df[df['DESC_PRODUCTO'] == producto_a]['COD_PRODUCTO'].values[0]
                producto_b_id = df[df['DESC_PRODUCTO'] == producto_b]['COD_PRODUCTO'].values[0]
            except IndexError:
                st.error(f"No se encontró el código para '{producto_a}' o '{producto_b}'.")
                continue  # Saltar este combo si falta alguno de los códigos

            # Filtrar las ventas mensuales de cada producto usando los nombres de columna correctos
            ventas_a = df_ventas[df_ventas['COD_PRODUCTO'] == producto_a_id]
            ventas_b = df_ventas[df_ventas['COD_PRODUCTO'] == producto_b_id]

            # Verificar si hay datos disponibles para ambos productos
            if not ventas_a.empty and not ventas_b.empty:
                # Calcular las métricas requeridas utilizando los nombres de columna correctos y formatear con separadores de miles
                cantidad_estimada = "{:,.0f}".format(ventas_a['Cantidad Vendida'].mean() + ventas_b['Cantidad Vendida'].mean())
                venta_estimada = "{:,.0f}".format(ventas_a['Precio Total'].mean() + ventas_b['Precio Total'].mean())
                ganancia_estimada = "{:,.0f}".format(
                    (ventas_a['Precio Total'].mean() - ventas_a['Costo total'].mean()) +
                    (ventas_b['Precio Total'].mean() - ventas_b['Costo total'].mean())
                )

                # Agregar los datos del combo al resumen
                resumen.append({
                    'Combo': f"{producto_a} + {producto_b}",
                    'Cantidad estimada de venta': cantidad_estimada,
                    'Venta estimada ($)': venta_estimada,
                    'Ganancia estimada ($)': ganancia_estimada
                })
            else:
                st.warning(f"No hay datos de ventas mensuales para '{producto_a}' o '{producto_b}'.")

        # Mostrar el resumen si se generaron datos
        if resumen:
            df_resumen = pd.DataFrame(resumen)
            st.table(df_resumen)
        else:
            st.write("No se han generado datos de resumen para los combos seleccionados.")
    else:
        st.write("No se han seleccionado combos para mostrar el resumen.")

