#!/usr/local/bin/python3.10

# Importing the required dependencies
import os
import abc
import json
import requests
import cbsodata
import pandas as pd
import sqlalchemy as sa
from typing import Any
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod

# Path where all the datasets will be stored
data_path: str = r"../data"
port: str = "5432"

# Some helper functions
def dataframe_to_csv(df: pd.DataFrame, save_folder: Path, file_name: str) -> None:
    """ Converts a DataFrame to a csv file and saves it at a specific location

    Args:
        df (pd.DataFrame): DataFrame to be converted to CSV
        save_folder (Path): The folder in which the file needs to be saved
        file_name (str): The actual name of the file
    """
    # Setup required folders
    check_and_create_folder(save_folder)

    df.to_csv(f"{save_folder}/{file_name}.csv", ",", index=False, encoding="utf-8")


def check_and_create_folder(folder_path: Path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")


class Dataset(ABC):

    @abstractmethod
    def retrieve_data(self):
        print("not implemented yet")

    @abstractmethod
    def clean_data(self):
        print("not implemented yet")

    @abstractmethod
    def store_data(self):
        print("not implemented yet")


class PoliceDataset(Dataset):

    police_url: str = "https://api.politie.nl/v4/gezocht"
    wanted_persons_parameters: dict[str, str] = {
        "uid": None,
        "language": "nl",
        "query": None,
        "lat": None,
        "lon": None,
        "radius": None,
        "maxnumberofitems": "25"
    }

    def retrieve_data(self):
        police_datadata: pd.DataFrame = self.get_police_data(self.police_url, 3, self.wanted_persons_parameters)
        dataframe_to_csv(police_datadata, data_path, "wanted_persons")
    
    def clean_data(self):
        return super().clean_data()
    
    def store_data(self):
        return super().store_data()
    
    def retrieve_police_data(target_url: str, max_requests: int = 10, parameters: dict[str, str] = {}) -> pd.DataFrame:
        """ Gets the data from the police API back in a dataframe. Since the API is limited to only returning 25 records,
        the API gets queried for a specified numer of times. While making the API call, it is possible to add additional parameters

        Args:
            target_url (str): The url for the specific data that you want to retrieve from the police
            max_requests (int, optional): Maximum number of requests made to the police API. Defaults to 10
            parameters (dict[str, str], optional): Additional parameters that can be added to the API request. Defaults to {}.

        Returns:
            pd.DataFrame: DataFrame containing the desired data
        """
        df: pd.DataFrame = pd.DataFrame()
        request_index: int = 1

        # Adding the parameters to the target url
        if parameters != {}:
            target_url = f"{target_url}?"

            for parameter, value in parameters.items():
                if value != None:
                    target_url = f"{target_url}{parameter}={value}&"

        base_url: str = target_url

        # Making the requests
        while request_index <= max_requests:
            print(f"Starting request {str(request_index)}/{str(max_requests)}...")
            # Calculate the offset
            offset: int = (request_index - 1) * 25
            
            if parameter != {}:
                target_url = f"{base_url}offset={offset}"
            else:
                target_url = f"{base_url}&offset={offset}"

            r = requests.get(target_url).json()
            df = pd.concat([df, pd.DataFrame(r["opsporingsberichten"])], ignore_index=True)

            request_index = request_index + 1

        print("Finished requests")

        return df
    

class CBSDataset(ABC):

    @abc.abstractproperty
    def name(self) -> str:
        return "This is the name of the dataset"

    @abc.abstractproperty
    def identifier(self) -> str:
        return "This is the identifier for the dataset"

    @abstractmethod
    def retrieve_data(self, identifier: str, name: str):
        print(f"Started: Retrieving dataset {self.name}")
        cbsodata.get_data(identifier, dir=f"{data_path}/{name}")
        print(f"Ended: Retrieving dataset {self.name}")

    @abstractmethod
    def clean_data(self):
        print("not implemented yet")

    @abstractmethod
    def store_data(self, engine: sa.Engine):
        df: pd.DataFrame = pd.read_csv(f"{data_path}/cleaned/{self.name}/{self.name}_cleaned.csv")

        with engine.connect() as connection:
            result: sa.CursorResult[Any] = connection.execute(sa.text(f"DROP TABLE IF EXISTS {self.name} CASCADE;"))

        df.to_sql("income_inequality", engine, if_exists="replace", index=True)


class HouseholdIncomeDataset(CBSDataset):

    @property
    def name(self) -> str:
        return "income_houshold"

    @property
    def identifier(self) -> str:
        return "85342NED"

    def retrieve_data(self):
        super().retrieve_data(self.identifier, self.name)

    def clean_data(self):
        self.__clean_income_dataset(f"{data_path}/{self.name}", f"{data_path}/cleaned/{self.name}", f"{self.name}_cleaned")

    def store_data(self, engine: sa.Engine):
        super().store_data(engine)

    def __clean_income_dataset(self, dataset_path: Path, save_folder: Path, file_name: str) -> None:
        """
        Clean the income dataset. First all redundant values are removed from the JSON. Next, all region codes are translated
        to actual region names, income values are changed to the right format and extra information is added. The results are
        stored in a csv file, which can put in a database.

        Args:
            dataset_path (Path): The path to the folder that contains all JSONs.
            save_folder (Path): Folder where the cleaned csv needs to be saved.
            file_name (str): Name of the resulting file.
        """
        print("Started: Cleaning income dataset")

        # Create some variables for the names of JSONs that will be used
        dataset_income_name: str = "TypedDataSet.json"
        dataset_regions_name: str = "RegioS.json"

        # First store the necessary datasets in variables
        with open(f"{dataset_path}/{dataset_income_name}", "r") as f:
            income_dataset = json.load(f)

        with open(f"{dataset_path}/{dataset_regions_name}", "r") as f:
            regions_dataset = json.load(f)
        
        # Remove all redundant values from the dataset
        income_dataset = self.__remove_redundant_values_income_dataset(income_dataset)
        income_dataset = self.__transform_values_income_dataset(income_dataset, regions_dataset)

        income_df: pd.DataFrame = pd.DataFrame(income_dataset)

        dataframe_to_csv(income_df, save_folder, file_name)

        print("Ended: Cleaning income dataset")
        
    def __remove_redundant_values_income_dataset(self, dataset: dict[str, Any]) -> dict[str, Any]:
        """
        In the income dataset there are some values that need to be removed. These are the values that don't contain the right
        region level or not the right houshold features.

        Args:
            dataset (dict[str, Any]): Income dataset to be cleaned.

        Returns:
            dict[str, Any]: Cleaned income dataset.
        """
        print("Started: Removing redundant values income dataset")

        # remove all data that does not have to do with private housholds
        dataset: dict[str, Any] = [item for item in dataset if item["KenmerkenVanHuishoudens"] == "1050010"]
        dataset: dict[str, Any] = [item for item in dataset if item["Populatie"] == "1050010"]

        # keep all data that has the correct region type (national level, province level or municipality level)
        dataset: dict[str, Any] = [item for item in dataset if  item["RegioS"].startswith("NL") or
                                                                item["RegioS"].startswith("PV") or
                                                                item["RegioS"].startswith("GM")]
        
        # Remove all redundant data
        for item in dataset:
            item.pop("ID", None)
            item.pop("KenmerkenVanHuishoudens", None)
            item.pop("ParticuliereHuishoudensRelatief_2", None)
            item.pop("GemiddeldBesteedbaarInkomen_5", None)
            item.pop("MediaanBesteedbaarInkomen_6", None)
            item.pop("GestandaardiseerdInkomen1e10Groep_7", None)
            item.pop("GestandaardiseerdInkomen2e10Groep_8", None)
            item.pop("GestandaardiseerdInkomen3e10Groep_9", None)   
            item.pop("GestandaardiseerdInkomen4e10Groep_10", None)
            item.pop("GestandaardiseerdInkomen5e10Groep_11", None)
            item.pop("GestandaardiseerdInkomen6e10Groep_12", None)
            item.pop("GestandaardiseerdInkomen7e10Groep_13", None)
            item.pop("GestandaardiseerdInkomen8e10Groep_14", None)
            item.pop("GestandaardiseerdInkomen9e10Groep_15", None)
            item.pop("GestandaardiseerdInkomen10e10Groep_16", None)

        print("Ended: Removing redundant values income dataset")

        return dataset

    def __transform_values_income_dataset(self, income_dataset: dict[str, Any], region_dataset: dict[str, Any]) -> dict[str, Any]:
        """
        Transforms all data in the correct format. This means transforming the population code, transforming the region code,
        updating the houshold amount and updating the income values.

        Args:
            income_dataset (dict[str, Any]): The income dataset, where values still need to be transformed.
            region_dataset (dict[str, Any]): The regions dataset, to translate region codes to actual region names.

        Returns:
            dict[str, Any]: The income dataset with transformed values.
        """
        print("Started: Transforming values income dataset")

        # Create a dictionary from the regions dataset, where the key is the region-key and the values are the corresponsing values
        converted_region_dataset: dict[str, Any] = {}

        for item in region_dataset:
            key = item.get("Key")
            if key is not None:
                item.pop("Key")
                converted_region_dataset[key] = item

        # Transforms data in the right format and deletes redundant data
        for item in income_dataset:
            
            # Check the region for each item in the dataset and set the region type and name
            match item["RegioS"][0:2]:
                case "NL":
                    region_type: str = "Country"
                    region_name: str = converted_region_dataset[item["RegioS"]]["Title"]
                case "PV":
                    region_type: str = "Province"
                    region_name: str = converted_region_dataset[item["RegioS"]]["Title"][:-5]
                case "GM":
                    region_type: str = "Municipality"
                    region_name: str = converted_region_dataset[item["RegioS"]]["Title"]

            item["dataset"] = "income_inequality"
            item["time_period"] = item["Perioden"][0:4]
            item["region_name"] = region_name
            item["region_type"] = region_type
            item["region_code"] = item["RegioS"].strip()
            item["population"] = "Private housholds"
            item["private_houshold_amount"] = round(item["ParticuliereHuishoudens_1"] * 1000, 1)
            item["average_standardized_income"] = round(item["GemiddeldGestandaardiseerdInkomen_3"] * 1000, 1)
            item["median_standardized_income"] = round(item["MediaanGestandaardiseerdInkomen_4"] * 1000, 1)
            item["inequality_income"] = round(item["average_standardized_income"] - item["median_standardized_income"], 1)

            # Delete remaining redundant data
            item.pop("Populatie", None)
            item.pop("RegioS", None)
            item.pop("Perioden", None)
            item.pop("ParticuliereHuishoudens_1", None)
            item.pop("GemiddeldGestandaardiseerdInkomen_3", None)
            item.pop("MediaanGestandaardiseerdInkomen_4", None)

        print("Ended: Transforming values income dataset")

        return income_dataset


def data_processing_pipeline(datasets: list[CBSDataset]) -> None:
    """
    Here all the steps for retrieving, cleaning and storing the data will take place
    """
    engine: sa.Engine = sa.create_engine(f"postgresql://student:infomdss@dashboard:{port}/dashboard")

    for dataset in datasets:
        #dataset.retrieve_data()
        dataset.clean_data()
        dataset.store_data(engine)


if __name__ == "__main__":
    # Setup for all datasets that will be scraped
    datasets: list[CBSDataset] = [HouseholdIncomeDataset()]

    data_processing_pipeline(datasets)