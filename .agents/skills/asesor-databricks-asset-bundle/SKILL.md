# Copilot Instructions — Machine Learning con Databricks Asset Bundle (DAB)

> **Ubicación requerida en el proyecto:** `.github/copilot-instructions.md`
> GitHub Copilot leerá este archivo automáticamente en cada sesión del workspace.

---

## Rol del agente

Eres un ingeniero senior de Machine Learning especializado en **Databricks Asset Bundle (DAB)**.
Tu stack es: Python · PySpark · MLflow · Delta Lake · Databricks Workflows.
Escribe código limpio, reproducible y alineado con la estructura DAB definida en este archivo.

---

## 1. Estructura del Proyecto DAB

Toda sugerencia de código debe respetar esta estructura de carpetas:

```
project-root/
├── .github/
│   └── copilot-instructions.md   ← este archivo
├── databricks.yml                 ← bundle root config
├── resources/
│   ├── jobs/                      ← definición de workflows .yml
│   └── clusters/                  ← cluster policies .yml
├── src/
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   └── features.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py
│   ├── serving/
│   │   ├── __init__.py
│   │   └── serve.py
│   └── utils/
│       └── helpers.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_eng.ipynb
│   ├── 03_training.ipynb
│   └── 04_serving.ipynb
├── tests/
│   └── unit/
├── conf/
│   ├── dev.yml
│   ├── staging.yml
│   └── prod.yml
└── requirements.txt
```

### Reglas de estructura
- Nunca mezclar lógica de negocio en notebooks; los notebooks solo **orquestan** llamadas a `src/`.
- Cada etapa del pipeline debe ser un **task independiente** en el job DAB.
- Los parámetros de entorno van siempre en `conf/<env>.yml`, nunca hardcodeados.

---

## 2. Setup & Configuración DAB

### `databricks.yml` — plantilla base

```yaml
bundle:
  name: ml-project-name

variables:
  env:
    default: dev
  catalog:
    default: dev_catalog
  schema:
    default: ml_schema

workspace:
  host: ${var.workspace_host}

targets:
  dev:
    mode: development
    workspace:
      host: https://<dev-workspace>.azuredatabricks.net
    variables:
      env: dev
      catalog: dev_catalog

  prod:
    mode: production
    workspace:
      host: https://<prod-workspace>.azuredatabricks.net
    variables:
      env: prod
      catalog: prod_catalog

resources:
  jobs:
    ml_pipeline:
      name: ml-pipeline-${var.env}
      job_clusters:
        - job_cluster_key: ml_cluster
          new_cluster:
            spark_version: 15.4.x-scala2.12
            node_type_id: Standard_DS3_v2
            num_workers: 2
      tasks:
        - task_key: feature_engineering
          job_cluster_key: ml_cluster
          python_wheel_task:
            package_name: ml_project
            entry_point: run_features
          libraries:
            - whl: ./dist/*.whl

        - task_key: training
          depends_on:
            - task_key: feature_engineering
          job_cluster_key: ml_cluster
          python_wheel_task:
            package_name: ml_project
            entry_point: run_training

        - task_key: serving
          depends_on:
            - task_key: training
          job_cluster_key: ml_cluster
          python_wheel_task:
            package_name: ml_project
            entry_point: run_serving
```

### Comandos DAB esenciales

```bash
# Validar bundle
databricks bundle validate

# Deploy a dev
databricks bundle deploy --target dev

# Ejecutar job
databricks bundle run ml_pipeline --target dev

# Deploy a prod
databricks bundle deploy --target prod
```

---

## 2.4 Manejo de catalog y schema para evitar [SCHEMA_NOT_FOUND]

Para prevenir errores como:
`[SCHEMA_NOT_FOUND] The schema `workspace`.`geo_mate2010` cannot be found`, sigue siempre estas reglas:

- `USE CATALOG <catalog>` antes de operar con schema.
- `CREATE SCHEMA IF NOT EXISTS <catalog>.<schema>` para tolerancia a schema nuevo.
- `USE SCHEMA <catalog>.<schema>` para establecer el contexto de la sesión.

En notebooks y scripts deben usarse valores inyectados por parámetros (widgets o argumentos) y nunca hardcodeo literal.

Ejemplo (aplicable en `src/sample_notebook.ipynb` y `src/dab_project/main.py`):

```python
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"USE SCHEMA {catalog}.{schema}")
```

---

## 3. Feature Engineering

### Convenciones
- Usar **Delta Lake** como formato de almacenamiento; nunca CSV o Parquet crudo.
- Las feature tables se registran en **Unity Catalog**: `<catalog>.<schema>.<table>`.
- Toda función de transformación debe ser **pura** (sin side effects) y testeable.

### Plantilla `src/feature_engineering/features.py`

```python
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from databricks.feature_engineering import FeatureEngineeringClient
import logging

logger = logging.getLogger(__name__)


def compute_features(spark: SparkSession, input_table: str) -> DataFrame:
    """
    Calcula features a partir de la tabla de entrada.
    
    Args:
        spark: SparkSession activa
        input_table: Nombre completo de la tabla fuente (catalog.schema.table)
    Returns:
        DataFrame con features calculadas
    """
    df = spark.table(input_table)

    df_features = (
        df
        .withColumn("feature_ratio", F.col("col_a") / F.col("col_b"))
        .withColumn("feature_log", F.log(F.col("col_c").cast(DoubleType())))
        .withColumn("feature_date_diff",
                    F.datediff(F.current_date(), F.col("date_col")))
        .dropna(subset=["feature_ratio", "feature_log"])
    )

    logger.info(f"Features calculadas: {df_features.count()} filas")
    return df_features


def write_feature_table(
    fe_client: FeatureEngineeringClient,
    df: DataFrame,
    table_name: str,
    primary_keys: list[str]
) -> None:
    """Escribe o actualiza una Feature Table en Unity Catalog."""
    try:
        fe_client.write_table(
            name=table_name,
            df=df,
            mode="merge"
        )
        logger.info(f"Feature table actualizada: {table_name}")
    except Exception:
        fe_client.create_table(
            name=table_name,
            primary_keys=primary_keys,
            df=df,
            description="Feature table generada por DAB pipeline"
        )
        logger.info(f"Feature table creada: {table_name}")


def run_features() -> None:
    """Entry point para DAB python_wheel_task."""
    spark = SparkSession.builder.getOrCreate()
    fe = FeatureEngineeringClient()

    input_table = spark.conf.get("pipeline.input_table")
    feature_table = spark.conf.get("pipeline.feature_table")
    primary_keys = spark.conf.get("pipeline.primary_keys").split(",")

    df_features = compute_features(spark, input_table)
    write_feature_table(fe, df_features, feature_table, primary_keys)
```

---

## 4. Entrenamiento de Modelos

### Convenciones
- **Siempre** usar `mlflow.autolog()` al inicio del training script.
- Registrar modelos en **MLflow Model Registry** con el stage `Staging` post-training.
- Usar `mlflow.set_experiment()` con nombre estructurado: `/<catalog>/<project>/<model_name>`.
- Incluir métricas de evaluación y parámetros como artefactos MLflow.

### Plantilla `src/training/train.py`

```python
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from pyspark.sql import SparkSession
from databricks.feature_engineering import FeatureEngineeringClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd
import logging

logger = logging.getLogger(__name__)

FEATURE_COLS = ["feature_ratio", "feature_log", "feature_date_diff"]
TARGET_COL = "label"


def load_training_data(
    fe: FeatureEngineeringClient,
    feature_table: str,
    label_table: str
) -> pd.DataFrame:
    """Carga features y labels unificados para entrenamiento."""
    spark = SparkSession.builder.getOrCreate()
    df_labels = spark.table(label_table).select("id", TARGET_COL)

    training_set = fe.create_training_set(
        df=df_labels,
        feature_lookups=[
            mlflow.entities.FeatureLookup(
                table_name=feature_table,
                feature_names=FEATURE_COLS,
                lookup_key="id"
            )
        ],
        label=TARGET_COL
    )
    return training_set.load_df().toPandas()


def train_model(df: pd.DataFrame, params: dict) -> tuple:
    """Entrena el modelo y retorna (model, metrics, X_test, y_test)."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "f1_score": f1_score(y_test, y_pred)
    }

    logger.info(f"Métricas: {metrics}")
    return model, metrics, X_test, y_test


def run_training() -> None:
    """Entry point para DAB python_wheel_task."""
    spark = SparkSession.builder.getOrCreate()
    fe = FeatureEngineeringClient()

    catalog = spark.conf.get("pipeline.catalog")
    project = spark.conf.get("pipeline.project")
    model_name = spark.conf.get("pipeline.model_name")
    feature_table = spark.conf.get("pipeline.feature_table")
    label_table = spark.conf.get("pipeline.label_table")

    params = {
        "n_estimators": int(spark.conf.get("model.n_estimators", "100")),
        "max_depth": int(spark.conf.get("model.max_depth", "5")),
        "learning_rate": float(spark.conf.get("model.learning_rate", "0.1"))
    }

    mlflow.set_experiment(f"/{catalog}/{project}/{model_name}")
    mlflow.autolog(disable=False)

    with mlflow.start_run(run_name=f"{model_name}_training"):
        df = load_training_data(fe, feature_table, label_table)
        model, metrics, X_test, y_test = train_model(df, params)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_test, model.predict(X_test))
        fe.log_model(
            model=model,
            artifact_path=model_name,
            flavor=mlflow.sklearn,
            training_set=fe.create_training_set(
                df=spark.table(label_table).select("id", TARGET_COL),
                feature_lookups=[],
                label=TARGET_COL
            ),
            registered_model_name=f"{catalog}.{project}.{model_name}",
            signature=signature
        )

    logger.info(f"Modelo registrado: {catalog}.{project}.{model_name}")
```

---

## 5. Deployment & Serving

### Convenciones
- Usar **Databricks Model Serving** para endpoints REST en tiempo real.
- Transicionar modelos de `Staging` → `Production` solo si métricas superan threshold.
- El endpoint name sigue el patrón: `<project>-<model_name>-<env>`.

### Plantilla `src/serving/serve.py`

```python
import mlflow
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize
)
from pyspark.sql import SparkSession
import logging

logger = logging.getLogger(__name__)


def promote_model_to_production(
    model_name: str,
    min_roc_auc: float = 0.75
) -> str:
    """
    Promueve a Production si las métricas superan el threshold.
    Retorna la versión promovida.
    """
    client = MlflowClient()
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

    if not staging_versions:
        raise ValueError(f"No hay versiones en Staging para {model_name}")

    latest = staging_versions[-1]
    run = client.get_run(latest.run_id)
    roc_auc = float(run.data.metrics.get("roc_auc", 0))

    if roc_auc < min_roc_auc:
        raise ValueError(
            f"ROC AUC {roc_auc:.3f} < threshold {min_roc_auc}. "
            "No se promueve a Production."
        )

    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True
    )

    logger.info(f"Modelo {model_name} v{latest.version} promovido a Production")
    return latest.version


def deploy_model_endpoint(
    workspace_client: WorkspaceClient,
    endpoint_name: str,
    model_name: str,
    model_version: str,
    workload_size: str = "Small"
) -> None:
    """Crea o actualiza un Databricks Model Serving endpoint."""
    config = EndpointCoreConfigInput(
        name=endpoint_name,
        served_models=[
            ServedModelInput(
                name=f"{model_name}-{model_version}",
                model_name=model_name,
                model_version=model_version,
                workload_size=ServedModelInputWorkloadSize(workload_size),
                scale_to_zero_enabled=True
            )
        ]
    )

    try:
        existing = workspace_client.serving_endpoints.get(endpoint_name)
        workspace_client.serving_endpoints.update_config(
            name=endpoint_name,
            served_models=config.served_models
        )
        logger.info(f"Endpoint actualizado: {endpoint_name}")
    except Exception:
        workspace_client.serving_endpoints.create(
            name=endpoint_name,
            config=config
        )
        logger.info(f"Endpoint creado: {endpoint_name}")


def run_serving() -> None:
    """Entry point para DAB python_wheel_task."""
    spark = SparkSession.builder.getOrCreate()
    w = WorkspaceClient()

    catalog = spark.conf.get("pipeline.catalog")
    project = spark.conf.get("pipeline.project")
    model_name = spark.conf.get("pipeline.model_name")
    env = spark.conf.get("pipeline.env")
    min_roc_auc = float(spark.conf.get("serving.min_roc_auc", "0.75"))

    full_model_name = f"{catalog}.{project}.{model_name}"
    endpoint_name = f"{project}-{model_name}-{env}"

    version = promote_model_to_production(full_model_name, min_roc_auc)
    deploy_model_endpoint(w, endpoint_name, full_model_name, version)
```

---

## 6. Patrones que Copilot debe seguir SIEMPRE

| Contexto | Patrón obligatorio |
|---|---|
| Lectura de datos | `spark.table("catalog.schema.table")` — nunca paths directos |
| Escritura | `.write.format("delta").mode("merge/overwrite")` |
| Logging | `logging.getLogger(__name__)` — nunca `print()` |
| Config | `spark.conf.get("key")` — nunca variables de entorno hardcodeadas |
| Tests | `pytest` + `pyspark.testing.assertDataFrameEqual` |
| Secrets | `dbutils.secrets.get(scope, key)` — nunca strings en código |
| Imports MLflow | Siempre `import mlflow` + `mlflow.set_tracking_uri("databricks")` si es local |

---

## 7. Anti-patrones — Copilot NO debe sugerir

```python
# ❌ MAL — pandas en cluster distribuido para datos grandes
df_pandas = spark_df.toPandas()  # solo para datasets pequeños (<1M filas)

# ❌ MAL — credenciales hardcodeadas
token = "dapi1234abcd..."

# ❌ MAL — paths absolutos DBFS legacy
df = spark.read.csv("dbfs:/FileStore/data.csv")

# ❌ MAL — sin control de errores en MLflow runs
mlflow.start_run()  # siempre usar como context manager: with mlflow.start_run():

# ✅ BIEN — Unity Catalog + Delta
df = spark.table("prod_catalog.ml_schema.raw_data")

# ✅ BIEN — secrets con dbutils
token = dbutils.secrets.get(scope="ml-secrets", key="api-token")
```

---

## 8. Checklist antes de hacer deploy

- [ ] `databricks bundle validate` sin errores
- [ ] Tests unitarios en `tests/unit/` pasando
- [ ] Feature table registrada en Unity Catalog
- [ ] Run MLflow con métricas registradas
- [ ] Modelo en Model Registry (stage: Staging mínimo)
- [ ] Endpoint name sigue convención `<project>-<model>-<env>`
- [ ] Parámetros en `conf/<env>.yml`, no hardcodeados