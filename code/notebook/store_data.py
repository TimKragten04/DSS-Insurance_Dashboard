import pandas as pd
import sqlalchemy as sa
from typing import Any

data_path: str = "../data/cleaned/child_poverty/poverty_all_data.csv"
port: str = "5432"
name: str = "children_poverty" 


def store_data():
    print("test")

    engine: sa.Engine = sa.create_engine(f"postgresql://student:infomdss@dashboard:{port}/dashboard")

    df: pd.DataFrame = pd.read_csv(f"{data_path}")

    with engine.connect() as connection:
        result: sa.CursorResult[Any] = connection.execute(sa.text(f"DROP TABLE IF EXISTS {name} CASCADE;"))

    df.to_sql("income_inequality", engine, if_exists="replace", index=True)


if __name__ == "__main__":
    store_data()