import streamlit as st
import pandas as pd
import numpy as np
import gdown
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# Funciones para cargar datos
@st.cache_data
def cargar_datos():
    url = 'https://drive.google.com/uc?id=1NmAZBoSj8YqWFbypAm8HYMj2YHbRyggT'
    output = 'datos.csv'
    gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    return df  # Asegurarse de devolver el DataFrame

@st.cache_data
def cargar_ventas_mensuales():
    url = 'https://drive.google.com/uc?id=1-21lc0LEqQLeph9YmnqIv5dhnDMzV15q'
    output = 'ventas_mensuales.csv'
    gdown.download(url, output, quiet=False)
    df_ventas = pd.read_csv(output)
    return df_ventas  # Asegurarse de devolver el DataFrame

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

# Ventana de Selección: Categoría, Subcategoría y Productos
st.header("Selecciona los Productos para Recomendación")

# Selección de categoría basada en el diccionario de secciones
categoria_seleccionada = st.selectbox("Seleccione una Categoría", list(secciones.keys()))

# Filtrar el DataFrame para obtener solo los datos de la categoría seleccionada
df_categoria = filtrar_por_categoria(df, categoria_seleccionada)

# Filtrar subcategorías según la categoría seleccionada
subcategorias_disponibles = df_categoria['DESC_CLASE'].unique()
subcategoria_seleccionada = st.selectbox("Seleccione una Subcategoría", subcategorias_disponibles)

# Filtrar productos según la subcategoría seleccionada
productos_disponibles = df_categoria[df_categoria['DESC_CLASE'] == subcategoria_seleccionada]['DESC_PRODUCTO'].unique()
productos_seleccionados = st.multiselect("Seleccione hasta 4 productos:", productos_disponibles, max_selections=4)

# Guardar la selección en el estado de sesión
if 'productos_seleccionados' not in st.session_state:
    st.session_state.productos_seleccionados = []
st.session_state.productos_seleccionados = productos_seleccionados

# Mostrar selección actual para verificar
st.write("Productos seleccionados:", st.session_state.productos_seleccionados)

# Importamos librerías adicionales para la segunda y tercera ventana
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# Función para preparar la matriz dispersa y entrenar el modelo ALS
def entrenar_modelo_als(df):
    # Crear matriz dispersa con facturas en las filas y productos en las columnas
    user_item_matrix = df.pivot(index='COD_FACTURA', columns='COD_PRODUCTO', values='CANTIDAD').fillna(0)
    sparse_matrix = csr_matrix(user_item_matrix.values)

    # Entrenar el modelo ALS
    modelo = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=30)
    modelo.fit(sparse_matrix)

    # Devolver el modelo y la matriz para su uso posterior
    return modelo, user_item_matrix

# Función para generar recomendaciones basadas en el modelo ALS
def generar_recomendaciones(modelo, user_item_matrix, producto_id, N=5):
    try:
        # Obtener el índice del producto en la matriz
        producto_idx = user_item_matrix.columns.get_loc(producto_id)
        
        # Generar recomendaciones
        recomendaciones = modelo.recommend(producto_idx, user_item_matrix.values.T, N=N, filter_already_liked_items=False)
        
        # Devolver los códigos de producto recomendados
        return [user_item_matrix.columns[i] for i, _ in recomendaciones]
    except KeyError:
        st.error(f"El producto con ID {producto_id} no se encontró en el modelo.")
        return []
    except Exception as e:
        st.error(f"Error en la función de recomendaciones: {e}")
        return []

# Función para calcular margen y precio de los combos
def calcular_combos(df, productos_seleccionados, als_recommendations):
    combos = []
    for producto_a in productos_seleccionados:
        for producto_b in als_recommendations.get(producto_a, []):
            prod_a_data = df[df['COD_PRODUCTO'] == producto_a]
            prod_b_data = df[df['COD_PRODUCTO'] == producto_b]
            
            if not prod_a_data.empty and not prod_b_data.empty:
                descripcion_a = prod_a_data['DESC_PRODUCTO'].values[0]
                descripcion_b = prod_b_data['DESC_PRODUCTO'].values[0]
                precio_a = prod_a_data['VALOR_PVSI'].values[0]
                costo_a = prod_a_data['COSTO'].values[0]
                precio_b = prod_b_data['VALOR_PVSI'].values[0]
                costo_b = prod_b_data['COSTO'].values[0]
                
                precio_combo = precio_a + precio_b
                margen_combo = round(((precio_combo - (costo_a + costo_b)) / precio_combo) * 100, 2)
                
                combos.append({
                    'Producto A': descripcion_a,
                    'Producto B': descripcion_b,
                    'Precio Combo': precio_combo,
                    'Margen Combo': f"{margen_combo}%",
                    'Check': False
                })
    return pd.DataFrame(combos)

# Función para calcular las métricas de la tercera ventana
def calcular_resumen_combos(df_ventas, combos_seleccionados):
    resumen = []
    for _, combo in combos_seleccionados.iterrows():
        prod_a_data = df_ventas[df_ventas['DESC_PRODUCTO'] == combo['Producto A']]
        prod_b_data = df_ventas[df_ventas['DESC_PRODUCTO'] == combo['Producto B']]
        
        if not prod_a_data.empty and not prod_b_data.empty:
            cantidad_a = prod_a_data['Cantidad Vendida'].mean()
            cantidad_b = prod_b_data['Cantidad Vendida'].mean()
            precio_total_a = prod_a_data['Precio Total'].mean()
            precio_total_b = prod_b_data['Precio Total'].mean()
            costo_total_a = prod_a_data['Costo total'].mean()
            costo_total_b = prod_b_data['Costo total'].mean()
            
            cantidad_venta_estimada = int(np.ceil(cantidad_a + cantidad_b))
            venta_estimada = round(precio_total_a + precio_total_b, 2)
            ganancia_estimada = round((venta_estimada - (costo_total_a + costo_total_b)), 2)
            
            resumen.append({
                'Producto A': combo['Producto A'],
                'Producto B': combo['Producto B'],
                'Cantidad estimada de venta': cantidad_venta_estimada,
                'Venta estimada ($)': f"${venta_estimada}",
                'Ganancia estimada ($)': f"${ganancia_estimada}"
            })
    return pd.DataFrame(resumen)

# Configuración de la aplicación Streamlit
menu_seleccion = st.sidebar.radio("Seleccione una ventana:", ["Seleccionar Productos", "Recomendaciones", "Resumen de Combos Seleccionados"])

# Ventana 1: Selección de Productos (ya implementada)

# Ventana 2: Mostrar Combos Recomendados
if menu_seleccion == "Recomendaciones":
    st.header("Combos Recomendados")

    if st.session_state.get('productos_seleccionados'):
        # Entrenar el modelo ALS y generar recomendaciones
        modelo, user_item_matrix = entrenar_modelo_als(df)
        als_recommendations = {producto: generar_recomendaciones(modelo, user_item_matrix, producto) for producto in st.session_state.productos_seleccionados}
        
        # Crear el dataframe de combos
        df_combos = calcular_combos(df, st.session_state.productos_seleccionados, als_recommendations)
        
        # Mostrar tabla de combos con opción de selección (checkbox)
        st.write("Seleccione los combos que desea incluir en el resumen:")
        
        # Agregamos una columna de checkboxes en la tabla de recomendaciones
        df_combos['Seleccionado'] = df_combos.apply(lambda x: st.checkbox(f"Combo {x.name+1}", key=f"combo_{x.name}"), axis=1)
        
        # Guardar los combos seleccionados en st.session_state
        if 'combos_seleccionados' not in st.session_state:
            st.session_state.combos_seleccionados = pd.DataFrame()
        st.session_state.combos_seleccionados = df_combos[df_combos['Seleccionado']]

# Ventana 3: Resumen de Combos Seleccionados
elif menu_seleccion == "Resumen de Combos Seleccionados":
    st.header("Resumen de Combos Seleccionados")

    # Verificamos que 'combos_seleccionados' esté definido y no esté vacío
    if 'combos_seleccionados' in st.session_state and not st.session_state.combos_seleccionados.empty:
        # Calculamos y mostramos el resumen de combos seleccionados
        resumen_df = calcular_resumen_combos(df_ventas, st.session_state.combos_seleccionados)
        st.table(resumen_df)
    else:
        st.write("No se han seleccionado combos.")
