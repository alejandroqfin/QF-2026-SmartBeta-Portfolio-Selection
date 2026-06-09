# Optimización de Carteras Smart Beta: Gestión del Riesgo Asimétrico y Clústering Jerárquico

> *Repositorio de código (Python) para la evaluación empírica de 96 estrategias de inversión sobre un universo de 100 ETFs Smart Beta en un entorno de costes de transacción por operar en el mercado.*

---

## Estructura del Repositorio

El código está estructurado de forma que se pueda garantizar la replicabilidad total del Trabajo de Fin de Máster (TFM). 

### 1. Universo de Datos
* `100_ETFs_Smart_Beta.xlsx`: Base de datos. Contiene la serie temporal de precios ajustados (2013-2026) del universo de 100 ETFs Smart Beta agrupados en familias.

### 2. Scripts Principales
Scripts principales para la ejecución del análisis:
* `descriptive_stats.py`: Genera la estadística descriptiva de los datos, frontera eficiente y ejecuta la regresión OLS del Modelo de Mercado frente a índices de referencia (S&P 500 y CRSP Total Market).
* `main1.py`: **Parte 1 (Screening).** Evaluación *Out-of-Sample* de ventanas móviles para evaluar los 12 ratios de performance, analizando redundancia topológica, matriz de coincidencias y rotación diaria.
* `main2.py`: **Parte 2 (Asset Allocation).** Simula la estrategia de asignación de capital cruzando los subconjuntos de activos seleccionados en la etapa anterior con los 8 modelos de pesos bajo costes de transacción.

### 3. Funciones auxiliares o Utils
Librerías internas con la matemática vectorizada del modelo:
* `metrics.py`: Métricas financieras, rentabilidades, descomposición de riesgo de Euler e índices de concentración (MHI).
* `screening.py`: Formulación matemática de los ratios de selección. Incluye medidas de momentos centrales, *Lower/Upper Partial Moments* y riesgo asimétrico de cola (VaR, Expected Shortfall, Generalized Rachev).
* `portfolios.py`: Optimizadores de asignación del capital o reparto de pesos. Contiene las formulaciones de los modelos paramétricos resueltos vía SciPy (GMV, MVS, ERC) y heurísticos (EW, VT, RRT).
* `hrp.py`: Modelos de Machine Learning. Implementa los algoritmos de *Hierarchical Risk Parity* (HRP) y *Hierarchical Equal Risk Contribution* (HERC).
* `plots.py`: Visualización gráfica del análisis.

---

## Replicabilidad del Trabajo
Para ejecutar la simulación completa y reproducir los resultados exactos documentados en la investigación, ejecute los códigos en orden secuencial: `descriptive_stats.py` -> `main1.py` -> `main2.py`.
