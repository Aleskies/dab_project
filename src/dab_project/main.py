import argparse
from databricks.sdk.runtime import spark
from dab_project import taxis


def ensure_catalog_and_schema(catalog: str, schema: str) -> None:
    """Asegura que el catalog y schema existan antes de usarlos."""
    spark.sql(f"USE CATALOG {catalog}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    # spark.sql(f"USE SCHEMA {catalog}.{schema}")


def main():
    # Process command-line arguments
    parser = argparse.ArgumentParser(
        description="Databricks job with catalog and schema parameters",
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    args = parser.parse_args()

    # Ensure catalog/schema exist and are in use
    ensure_catalog_and_schema(args.catalog, args.schema)

    # Example: just find all taxis from a sample catalog
    taxis.find_all_taxis().show(5)


if __name__ == "__main__":
    main()
