# -*- coding: utf-8 -*-
"""dashboard.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18HfzUkdRBYqOcv7fofhiIXbsXU379HT3
"""

import streamlit as st

# Diccionario con las categorías y sus subcategorías
categorias = {
    "Limpieza del Hogar": [
        "LAVADO DE ROPA", "TRATAMIENTO PARA LA ROPA", "LAVAVAJILLAS", "PAPEL HIGIENICO",
        "PLAGUICIDAS DOMESTICOS", "LIMPIADORES DE SUPERFICIES HOGAR", "LIMPIADORES LIQUIDOS",
        "DESCARTABLES DE COCINA", "DESODORANTES DE AMBIENTE", "IMPLEMENTOS HIGIENE DE HOGAR",
        "LIMPIADORES DE BANO", "VAJILLA DESCARTABLES", "LIMPIADORES DE HOGAR",
        "LIMPIEZA Y CUIDADO DEL AUTO", "AMBIENTALES PARA BANO", "CUIDADO Y ACCESORIOS DE CALZADO"
    ],
    "Cuidado Personal": [
        "SHAMPOO", "DESODORANTE CORPORAL", "JABONES", "TOALLAS HUMEDAS", "PROTECCION FEMENINA",
        "CREMAS DENTALES", "PERFUMERIA BEBE", "TRATAMIENTO CAPILAR", "PANALES DE BEBES",
        "SALUD RESPIRATORIA", "CEPILLOS DENTALES", "AFEITADO", "CUIDADO FACIAL", "TALCO PARA PIES",
        "FIJADORES", "BOTIQUIN", "CREMAS HIDRATANTES", "COMPLEMENTOS DE HIGIENE BUCAL", "FRAGANCIAS",
        "COLORACION", "NUTRICION Y VITAMINAS", "ACONDICIONADOR", "SALUD GENERAL", "ESMALTES Y QUITA ESMALTE",
        "SALUD DIGESTIVA", "PROTECCION SOLAR", "REPELENTES", "SALUD INTIMA", "INCONTINENCIA", "MAQUILLAJE"
    ],
    "Bebidas": [
        "JUGOS Y TE", "BEBIDAS NO ALCOHOLICAS SABORIZADAS", "CERVEZAS", "GASEOSAS", "AGUAS",
        "COCTELES", "ESPUMANTE", "WHISKY", "DESTILADAS", "VINOS", "CERVEZAS SIN ALCOHOL", "ENVASES BEBIDAS"
    ],
    "Alimentos": [
        "CAFE", "PESCADOS EN CONSERVAS", "ACEITES", "CHOCOLATES", "GRASAS", "FIDEOS", "PANIFICADOS",
        "REPOSTERIA", "GRANOS", "ENDULZANTES", "ARROZ", "CONDIMENTOS", "HARINAS Y COLADAS",
        "ALIMENTOS EN CONSERVAS", "POSTRES", "MANJAR DE LECHE", "CEREALES PARA EL DESAYUNO", "FRUTAS SECAS",
        "UNTABLES", "SALSAS", "HUEVOS", "INFUSIONES", "ALIMENTOS INFANTILES", "FRUTAS EN CONSERVAS"
    ]
}

# Título del dashboard
st.title("Dashboard de Categorías y Subcategorías")

# Crear un menú desplegable para seleccionar una categoría
categoria_seleccionada = st.selectbox("Selecciona una categoría", list(categorias.keys()))

# Mostrar las subcategorías basadas en la categoría seleccionada
if categoria_seleccionada:
    subcategorias = categorias[categoria_seleccionada]
    subcategoria_seleccionada = st.selectbox("Selecciona una subcategoría", subcategorias)

    # Mostrar la subcategoría seleccionada
    st.write(f"Has seleccionado la subcategoría: {subcategoria_seleccionada}")