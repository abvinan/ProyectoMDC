import streamlit as st
import pandas as pd
import numpy as np
import gdown
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split

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
        df_test_compras = df_test.groupby(['COD_FACTURA', 'COD_PRODUCTO'])['CANTIDAD'].sum().unstack().fillna(0)
        return df_train_compras, df_test_compras
    else:
        st.error("No hay suficientes datos para dividir en entrenamiento y prueba.")
        return None, None

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
    if df_train_compras is not None and als_model is not None:
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
    else:
        st.error("No se pudieron generar recomendaciones debido a la falta de datos o error en el entrenamiento.")
        return {}

# Configuración de la aplicación Streamlit
menu_seleccion = st.sidebar.radio("Seleccione una ventana:", ["Seleccionar Productos", "Recomendaciones", "Resumen de Combos Seleccionados"])

# Ventana 1: Selección de Productos
if menu_seleccion == "Seleccionar Productos":
    st.header("Selecciona los Productos para Recomendación")

    # Selección de categoría basada en el diccionario de secciones
    categoria_seleccionada = st.selectbox("Seleccione una Categoría", list(secciones.keys()))

    # Filtrar el DataFrame para obtener solo los datos de la categoría seleccionada
    df_categoria = filtrar_por_categoria(df, categoria_seleccionada)

    # Guardar df_categoria en el estado de la sesión
    st.session_state['df_categoria'] = df_categoria

    # Filtrar subcategorías según la categoría seleccionada
    subcategorias_disponibles = df_categoria['DESC_CLASE'].unique()
    subcategoria_seleccionada = st.selectbox("Seleccione una Subcategoría", subcategorias_disponibles)

    # Filtrar productos según la subcategoría seleccionada
    productos_disponibles = df_categoria[df_categoria['DESC_CLASE'] == subcategoria_seleccionada]['DESC_PRODUCTO'].unique()
    productos_seleccionados = st.multiselect("Seleccione hasta 4 productos:", productos_disponibles, max_selections=4)

    # Guardar la selección en el estado de sesión
    st.session_state.productos_seleccionados = productos_seleccionados

    # Crear una tabla para mostrar los productos seleccionados sin el índice
    if st.session_state.productos_seleccionados:
        productos_seleccionados_df = pd.DataFrame({
            "Productos seleccionados": st.session_state.productos_seleccionados
        })
        productos_seleccionados_df.index = [""] * len(productos_seleccionados_df)  # Eliminar el índice
        st.table(productos_seleccionados_df)
    else:
        st.write("No se han seleccionado productos.")

# Ventana 2: Mostrar Combos Recomendados
elif menu_seleccion == "Recomendaciones":
    st.header("Combos Recomendados")

    # Verificar que haya productos seleccionados y `df_categoria` esté definido en la sesión
    if 'productos_seleccionados' in st.session_state and st.session_state.productos_seleccionados and 'df_categoria' in st.session_state:
        df_categoria = st.session_state['df_categoria']

        # Obtener los IDs de los productos seleccionados
        productos_seleccionados_ids = [
            df[df['DESC_PRODUCTO'] == nombre]['COD_PRODUCTO'].values[0]
            for nombre in st.session_state.productos_seleccionados
        ]

        # Obtener el top 200 productos de la categoría seleccionada para entrenar el modelo
        df_top_200 = obtener_top_200_productos(df_categoria)

        # Preparar los datos y entrenar el modelo ALS
        df_train_compras, df_test_compras = preparar_datos_para_entrenar(df_top_200)
        modelo_als, df_train_sparse = entrenar_modelo_als(df_train_compras)

        # Generar las recomendaciones para los productos seleccionados
        recomendaciones = generar_recomendaciones_seleccionados(df_train_compras, modelo_als, df_train_sparse, productos_seleccionados_ids)

        # Mostrar las recomendaciones en una tabla
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

        # Convertir la lista de combos en un DataFrame y mostrarla
        df_combos = pd.DataFrame(combos)
        st.table(df_combos)
    else:
        st.write("No se han seleccionado productos en la ventana anterior.")
