# dp261-g3

## Miembros del Proyecto

| Nombre | Rol |
|--------|-----|
| Pedro Shiguihara  | Product Owner (P.O)       |
| Marcelo Condori   | Project Manager           |
| Elena Agüero      | Business Analyst          |
| Diana Ochoa       | Data Engineer, Data Cleaner|
| Elena Agüero      | Data Analyst , Class Balance           |
| Jesus Choque            | Prototype Developer       |

## GitHub Project

Tablero del proyecto: [dp261-g3 Project](https://github.com/orgs/pucp-datanalytics/projects/3)

## Labels

Los labels del repositorio siguen una convencion de prefijos para organizar el trabajo:

| Prefijo | Color | Descripcion |
|---------|-------|-------------|
| `scrum/` | `#0052CC` | `scrum/user-story`, `scrum/task`, `scrum/bug`, `scrum/backlog`, `scrum/in-progress`, `scrum/blocked`, `scrum/review`, `scrum/done` |
| `crisp/` | `#006B75` | `crisp/sprint1-business-data-understanding`, `crisp/sprint2-data-preparation`, `crisp/sprint3-modeling-baseline`, `crisp/sprint4-modeling-advanced`, `crisp/sprint5-evaluation`, `crisp/sprint6-deployment` |
| `rol/` | `#E65100` | `rol/pm`, `rol/ba`, `rol/de`, `rol/da`, `rol/pd` |
| `entrega/` | `#5319E7` | `entrega/notebook`, `entrega/dataset`, `entrega/environment`, `entrega/dashboard`, `entrega/model`, `entrega/report`, `entrega/docs`, `entrega/api-rest`, `entrega/aws-deploy` |
| `prioridad/` | `#B60205` | `prioridad/high`, `prioridad/medium`, `prioridad/low` |

### Detalle de Labels

**`scrum/`**

| Label | Descripcion |
|-------|-------------|
| `scrum/user-story` | Historia de usuario |
| `scrum/task` | Tarea técnica del sprint |
| `scrum/bug` | Error o fallo en el entregable |
| `scrum/backlog` | Pendiente en el product backlog |
| `scrum/in-progress` | En desarrollo activo |
| `scrum/blocked` | Bloqueado, requiere atención |
| `scrum/review` | En revisión del equipo |
| `scrum/done` | Completado y aprobado |

**`crisp/`**

| Label | Descripcion |
|-------|-------------|
| `crisp/sprint1-business-data-understanding` | Sprint 1: Business & Data Understanding |
| `crisp/sprint2-data-preparation` | Sprint 2: Data Preparation |
| `crisp/sprint3-modeling-baseline` | Sprint 3: Modeling baseline |
| `crisp/sprint4-modeling-advanced` | Sprint 4: Modeling avanzado + tuning |
| `crisp/sprint5-evaluation` | Sprint 5: Evaluation + Business Value |
| `crisp/sprint6-deployment` | Sprint 6: Deployment MVP en AWS |

**`rol/`**

| Label | Descripcion |
|-------|-------------|
| `rol/pm` | Project Manager |
| `rol/ba` | Business Analyst |
| `rol/de` | Data Engineer |
| `rol/da` | Data Analyst |
| `rol/pd` | Prototype Developer |

**`entrega/`**

| Label | Descripcion |
|-------|-------------|
| `entrega/notebook` | Jupyter Notebook |
| `entrega/dataset` | Conjunto de datos (DVC) |
| `entrega/environment` | environment.yml / entorno Conda |
| `entrega/dashboard` | Visualización / Dashboard interactivo |
| `entrega/model` | Modelo entrenado y serializado |
| `entrega/report` | Informe o presentación ejecutiva |
| `entrega/docs` | Documentación del proyecto |
| `entrega/api-rest` | API REST del modelo desplegada |
| `entrega/aws-deploy` | Despliegue en AWS (EC2/Lambda/SageMaker) |

**`prioridad/`**

| Label | Descripcion |
|-------|-------------|
| `prioridad/high` | Prioridad alta |
| `prioridad/medium` | Prioridad media |
| `prioridad/low` | Prioridad baja |

## Milestones

| Sprint | Titulo | Fecha de entrega |
|--------|--------|------------------|
| 1 | Sprint 1 — Business & Data Understanding | 10/04/2026 |
| 2 | Sprint 2 — Data Preparation | 17/04/2026 |
| 3 | Sprint 3 — Modeling (Baseline) | 24/04/2026 |
| 4 | Sprint 4 — Modeling (Avanzado + Tuning) | 08/05/2026 |
| 5 | Sprint 5 — Evaluation + Business Value | 15/05/2026 |
| 6 | Sprint 6 — Deployment MVP en AWS | 22/05/2026 |

### Detalle de Milestones

**Sprint 1 — Business & Data Understanding** (entrega: 10/04/2026)

Comprender el problema de negocio y explorar los datos.
Entregables:
- Repositorio GitHub configurado con board Kanban, Issues creados y asignados, README (PM)
- 01_business.ipynb: problema de negocio, variable objetivo, KPIs y criterios de éxito (BA)
- environment.yml + 02_data_loading.ipynb: entorno Conda, DVC configurado, carga y verificación de datos (DE)
- 03_eda.ipynb: análisis de calidad, estadísticas descriptivas, visualizaciones y hallazgos (DA)
- 04_prototype.ipynb: prototipo interactivo con ipywidgets (PD)

**Sprint 2 — Data Preparation** (entrega: 17/04/2026)

Limpiar, transformar y construir features del dataset.
Entregables:
- Pipeline de limpieza y transformación de datos
- Ingeniería de características (feature engineering)
- Dataset final listo para modelado versionado con DVC
- Notebook documentado con cada decisión de preprocesamiento

**Sprint 3 — Modeling (Baseline)** (entrega: 24/04/2026)

Entrenar y comparar modelos de clasificación baseline.
Entregables:
- Notebook de experimentación con modelos baseline (Logistic Regression, Decision Tree, etc.)
- Métricas comparativas: accuracy, precision, recall, F1, ROC-AUC
- Selección del mejor modelo baseline con justificación
- Registro de experimentos documentado

**Sprint 4 — Modeling (Avanzado + Tuning)** (entrega: 08/05/2026)

Optimizar hiperparámetros y aplicar modelos avanzados.
Entregables:
- Notebook con modelos avanzados (Random Forest, XGBoost, etc.)
- Optimización de hiperparámetros (GridSearchCV / Optuna)
- Comparativa final de modelos con métricas detalladas
- Modelo final seleccionado y serializado (pickle/joblib)

**Sprint 5 — Evaluation + Business Value** (entrega: 15/05/2026)

Evaluar el Business Value y documentar resultados finales.
Entregables:
- Evaluación del modelo frente a los KPIs definidos en Sprint 1
- Análisis de errores e interpretabilidad (SHAP, feature importance)
- Dashboard interactivo con resultados para stakeholders
- Informe ejecutivo con conclusiones y recomendaciones

**Sprint 6 — Deployment MVP en AWS** (entrega: 22/05/2026)

Desplegar el mejor modelo como MVP en AWS.
Entregables:
- API REST del modelo desplegada en AWS (EC2, Lambda o SageMaker)
- Dashboard interactivo accesible en producción
- Documentación de arquitectura y guía de uso
- Demo funcional del MVP presentada al stakeholder

