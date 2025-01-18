import streamlit as st
import gdown
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Funciones para cargar datos

def cargar_datos():
    url = 'https://drive.google.com/uc?id=1NmAZBoSj8YqWFbypAm8HYMj2YHbRyggT'
    output = 'datos.csv'
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

def cargar_ventas_mensuales():
    url = 'https://drive.google.com/uc?id=1-21lc0LEqQLeph9YmnqIv5dhnDMzV15q'
    output = 'ventas_mensuales.csv'
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

# Verificar si los datos ya están cargados en st.session_state
if "datos" not in st.session_state:
    st.session_state.datos = cargar_datos()

if "ventas_mensuales" not in st.session_state:
    st.session_state.ventas_mensuales = cargar_ventas_mensuales()

# Asignar los datos a variables
df = st.session_state.datos
df_ventas = st.session_state.ventas_mensuales

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

users = {
    "admin": "1234",
    "user1": "password",
    "user2": "pass123"
}

# Función para validar el login
def login(username, password):
    if username in users and users[username] == password:
        return True
    return False

# Configuración de la aplicación Streamlit
# Inicializar el estado de sesión
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "page" not in st.session_state:
    st.session_state.page = "login"

def show_login():
    # Configurar la página
    st.set_page_config(page_title="Inicio de sesión", page_icon="🔒", layout="wide")

    # Crear columnas vacías a los lados para centrar el formulario
    col1, col2, col3 = st.columns([0.35, 0.3, 0.35])

    with col2:  # Columna central donde estará el formulario
        # Contenedor del formulario con ancho fijo
        with st.container(border=True):
            st.title("Inicio de sesión")

            # Campos de entrada
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")

            # Botón de inicio de sesión centrado
            col_btn1, col_btn2, col_btn3 = st.columns([0.2, 0.6, 0.2])
            with col_btn2:
                login_button = st.button("Iniciar sesión",use_container_width=True, type="primary")

            # Lógica del login
            if login_button:
                if login(username, password):
                    st.session_state.authenticated = True
                    with st.spinner("Logeando..."):
                        time.sleep(3)
                    st.session_state.page = "app"
                    st.rerun()  
                else:
                    st.error("Usuario o contraseña incorrectos.")

# Funciones para cambiar de página
def home():
    st.session_state.pagina_actual = 1
    st.session_state.modelo_ejecutado = False
    st.session_state.seleccion_indices = []

def avanzar_pagina():
    if st.session_state.pagina_actual < 3:
        st.session_state.pagina_actual += 1


def retroceder_pagina():
    if st.session_state.pagina_actual > 1:
        st.session_state.pagina_actual -= 1
        if st.session_state.pagina_actual == 1:
            st.session_state.modelo_ejecutado = False
            st.session_state.seleccion_indices = []

def show_app():
    if "pagina_actual" not in st.session_state:
        st.session_state.pagina_actual = 1

    # Ventana 1: Selección de Productos
    if st.session_state.pagina_actual == 1:
        col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
        with col2:
            st.title("Selecciona los Productos para Recomendación")
            st.subheader("Seleccione una Categoría")
            categoria_seleccionada = st.selectbox("", list(secciones.keys()), label_visibility="collapsed")
            df_categoria = filtrar_por_categoria(df, categoria_seleccionada)
            st.session_state['df_categoria'] = df_categoria
            subcategorias_disponibles = df_categoria['DESC_CLASE'].unique()
            st.subheader("Seleccione una Subcategoría")
            subcategoria_seleccionada = st.selectbox("", subcategorias_disponibles, label_visibility="collapsed")
            productos_disponibles = df_categoria[df_categoria['DESC_CLASE'] == subcategoria_seleccionada]['DESC_PRODUCTO'].unique()
            st.subheader("Seleccione hasta 4 productos:")
            productos_seleccionados = st.multiselect("", productos_disponibles, max_selections=4, label_visibility="collapsed")
            st.session_state.productos_seleccionados = productos_seleccionados
            col_bttn1, col_bttn2, col_bttn3 = st.columns([0.4, 0.2, 0.4])

            with col_bttn2:
                st.button("Siguiente", on_click=avanzar_pagina ,use_container_width=True, type="primary")

    # Ventana 2: Mostrar Combos Recomendados
    elif st.session_state.pagina_actual == 2:
        if 'productos_seleccionados' in st.session_state and st.session_state.productos_seleccionados:
            if "modelo_ejecutado" not in st.session_state:
                st.session_state.modelo_ejecutado = False
            if "df_combos" not in st.session_state:
                st.session_state['df_combos'] = None
            if not st.session_state.modelo_ejecutado:
                df_categoria = st.session_state['df_categoria']
                productos_seleccionados_ids = [df[df['DESC_PRODUCTO'] == nombre]['COD_PRODUCTO'].values[0] for nombre in st.session_state.productos_seleccionados]
                df_top_200 = obtener_top_200_productos(df_categoria)
                df_train_compras = preparar_datos_para_entrenar(df_top_200)
                with st.spinner("Aplicando un modelo de inteligencia artificial para generación de combos..."):
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
                            'Margen': f"{margen_combo}%"
                        })
                st.session_state['df_combos'] = pd.DataFrame(combos)
                st.session_state.modelo_ejecutado = True
            df_combos = st.session_state['df_combos']

            def generar_barra_visual(valor, longitud=20):
                num_I = int((valor / 100) * longitud)
                return '[' + 'I' * num_I + '-' * (longitud - num_I) + ']'

            df_combos['Barra'] = df_combos['Margen'].str.rstrip('%').astype(float).apply(generar_barra_visual)

            df_combos.index = range(1, len(df_combos) + 1)

            if "seleccion_indices" not in st.session_state:
                st.session_state.seleccion_indices = []

            st.header("Combos Recomendados")
            st.subheader("Seleccione los índices de los combos que desea considerar:")
            col1, col2, col3, col4, col5 = st.columns([0.4,0.3, 0.1, 0.1,0.1],vertical_alignment="bottom")
            with col1:
                seleccion_indices = st.multiselect(
                    "",
                    df_combos.index.tolist(), label_visibility="hidden"
                )
            with col2:
                col2_1, col2_2 = st.columns(2,vertical_alignment="bottom")
                with col2_1:
                    if st.button("Guardar Selección", use_container_width=True, type="secondary"):
                        st.session_state.seleccion_indices = seleccion_indices
                        st.session_state.combos_seleccionados = df_combos.loc[seleccion_indices]
                        with col2_2:
                            st.success("¡Selección guardada!")
            with col4:
                st.button("Anterior", on_click=retroceder_pagina, use_container_width=True, type="primary")
            with col5:
                st.button("Siguiente", on_click=avanzar_pagina, use_container_width=True, type="primary")

            def resaltar_filas(fila):
                if fila.name in st.session_state.seleccion_indices:
                    return ["font-weight: bold; text-align: center;" for _ in fila]
                else:
                    return ["text-align: center;" for _ in fila]

            styled_df = (
                df_combos.style
                .apply(resaltar_filas, axis=1)
                .set_table_styles([
                    {"selector": "table", "props": [("border-collapse", "collapse"), ("margin", "auto")]},
                    {"selector": "th", "props": [
                        ("text-align", "center"),
                        ("font-weight", "bold"),
                        ("border", "2px solid black"),
                        ("padding", "8px"),
                        ("background-color", "#FFDAB9"),
                        ("color", "#4E342E"),
                        ("font-size", "20px")
                    ]},
                    {"selector": "td", "props": [("border", "2px solid black"), ("padding", "8px")]}
                ])
            )

            st.markdown(
                """
                <style>
                    .container {
                        display: flex;
                        justify-content: center;
                        width:100%
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="container">
                    {styled_df.to_html(escape=False)}
                """,
                unsafe_allow_html=True
            )

        else:
            st.warning("No hay productos seleccionados para generar combos.")
            col1, col2 = st.columns(2)
            with col1:
                st.button("Anterior", on_click=retroceder_pagina)
            with col2:
                st.button("Siguiente", on_click=avanzar_pagina)

    # Ventana 3: Resumen de Combos Seleccionados
    elif st.session_state.pagina_actual == 3:
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

                if not ventas_a.empty and not ventas_b.empty:
                    cantidad_estimada = "{:,.0f}".format(ventas_a['Cantidad Vendida'].mean() + ventas_b['Cantidad Vendida'].mean())
                    venta_estimada = "{:,.0f}".format(ventas_a['Precio Total'].mean() + ventas_b['Precio Total'].mean())
                    ganancia_estimada = "{:,.0f}".format(
                        (ventas_a['Precio Total'].mean() - ventas_a['Costo total'].mean()) +
                        (ventas_b['Precio Total'].mean() - ventas_b['Costo total'].mean())
                    )

                    resumen.append({
                        'Combo': f"{producto_a} + {producto_b}",
                        'Cantidad estimada de venta': cantidad_estimada,
                        'Venta estimada ($)': venta_estimada,
                        'Ganancia estimada ($)': ganancia_estimada
                    })
                else:
                    st.warning(f"No hay datos de ventas mensuales para '{producto_a}' o '{producto_b}'.")

            if resumen:
                df_resumen = pd.DataFrame(resumen)
                lista_combos = ["Combo "+ str(i+1) for i in range(len(df_resumen))]

                styled_df = (
                    df_resumen.style
                    .set_table_styles([
                        {"selector": "table", "props": [("border-collapse", "collapse"), ("margin", "auto")]},
                        {"selector": "th", "props": [
                            ("text-align", "center"),
                            ("font-weight", "bold"),
                            ("border", "2px solid black"),
                            ("padding", "8px"),
                            ("background-color", "#FFDAB9"),
                            ("color", "#4E342E"),
                            ("font-size", "20px")
                        ]},
                        {"selector": "td", "props": [("border", "2px solid black"), ("padding", "8px")]}
                    ])
                )

                st.markdown(
                    """
                    <style>
                        .container {
                            display: flex;
                            justify-content: center;
                            width:100%
                        }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="container">
                        {styled_df.to_html(escape=False)}
                    """,
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns([0.5, 0.5])
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(lista_combos, df_resumen["Venta estimada ($)"].str.replace(",", "", regex=False).astype(int) , label="Venta estimada ($)", marker="o")
                    ax.plot(lista_combos, df_resumen["Ganancia estimada ($)"].str.replace(",", "", regex=False).astype(int) , label="Ganancia estimada ($)", marker="o")
                    ax.set_title("Proyección de Ventas y Ganancias", fontsize=16)
                    ax.set_ylabel("Monto ($)", fontsize=12)
                    ax.legend()
                    plt.xticks(rotation=45, ha="right")
                    ax.grid(True)
                    fig.set_facecolor("none")

                    st.pyplot(fig)

                with col2:
                    df_c2 = df_resumen.copy()
                    df_c2['Combo'] = lista_combos.copy()
                    df_c2['Ganancia estimada ($) num'] =  df_c2['Ganancia estimada ($)'].str.replace(",", "", regex=False).astype(int)
                    df_c2 = df_c2.sort_values(by="Ganancia estimada ($) num", ascending=False)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(df_c2["Combo"], df_c2["Ganancia estimada ($) num"], color="skyblue")
                    ax.set_title("Ganancia Estimada por Combo", fontsize=16)
                    ax.set_ylabel("Ganancia Estimada ($)", fontsize=12)
                    plt.xticks(rotation=45, ha="right")
                    ax.grid(axis="y", linestyle="--", alpha=0.7)
                    fig.set_facecolor("none")

                    st.pyplot(fig)

            else:
                st.write("No se han generado datos de resumen para los combos seleccionados.")
        else:
            st.write("No se han seleccionado combos para mostrar el resumen.")

        col1, col2, col3, col4 = st.columns([0.3, 0.1, 0.1, 0.3])

        with col2:
            st.button("Anterior", on_click=retroceder_pagina, use_container_width=True, type="primary")
        with col3:
            st.button("Home", on_click=home, use_container_width=True, type="primary")

if st.session_state.page == "login":
    show_login()
elif st.session_state.page == "app":
    show_app()