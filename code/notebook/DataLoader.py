import requests
import pandas as pd
import cbsodata
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import os
from fnmatch import fnmatch
import concurrent.futures
import logging

data_path: str = r"../data/crime"

class DataLoader:
    def __init__(self):
        
        self.max_requests = 100
        self.chunk_size = 9999
        self.api_url= {"crimes": {"url":"dataderden.cbs.nl", 
                                  "table_id": "47013NED"},
                        "population": { "url":"opendata.cbs.nl", 
                                       "table_id": "37230ned"}}  
        current_path = os.path.abspath(__file__)
        self.data_path = os.path.join(os.path.dirname(current_path), data_path)
        self.save_path = {"crimes": "crime_counts"}


    def __dataframe_to_csv(self, df: pd.DataFrame, save_folder: Path, file_name: str) -> None:
        """ Converts a DataFrame to a csv file and saves it at a specific location

        Args:
            df (pd.DataFrame): DataFrame to be converted to CSV
            save_folder (Path): The folder in which the file needs to be saved
            file_name (str): The actual name of the file
        """
        df.to_csv(f"{save_folder}/{file_name}.csv", ",", index=False, encoding="utf-8")
        return

    def __request_from_api(self, target_url):
        response = requests.get(target_url)
        r = response.json()
        return pd.DataFrame(r['value'])

    def __fetch_raw_data(self, base_url:str, target_field:str, save_file:str)->pd.DataFrame:
        """
        This function retrieves the original data of crime counts and population.

        Args:
            base_url (str): The base URL for the API request.
            target_field (str): The specific field or endpoint for the data.
            save_file (str): The name of the file to save the data.

        Returns:
            pd.DataFrame: Original dataset.
        """
        with concurrent.futures.ThreadPoolExecutor(self.max_requests) as executor:
            futures = []
            skip_count = 0
            while True:
                filter = f"?$top={self.chunk_size}&$skip={skip_count}"
                target_url = base_url + target_field + filter
                future = executor.submit(self.__request_from_api, target_url)
                data = future.result()
                if len(data)==0:
                    break
                futures.append(future)
                skip_count += self.chunk_size

            data_frames = [future.result() for future in futures]

        original_data = pd.concat(data_frames, ignore_index=True)
        # 
        print("Data requests finished with total rows:", len(original_data))

        self.__dataframe_to_csv(original_data, f"{data_path}/crime_rates", save_file)

        return original_data
        
    def __clean_crimes_dataset(self, ori_data: pd.DataFrame, crimecodes: list, 
                             period:str, save_folder: Path, file_name: str)-> pd.DataFrame:
        """
        Clean the crimes dataset.

        Parameters:
        - ori_data (pd.DataFrame): Original crime dataset.
        - crimecodes (list): List of crime codes to focus on.
        - period (str): Period to filter the data.
        - save_folder (Path): Folder to save the cleaned dataset.
        - file_name (str): Name of the saved file.

        Returns:
        - pd.DataFrame: Cleaned crimes dataset.
        """
        print("Started: Cleaning  crime counts dataset.")

        # remove the sapces in the column RegioS
        ori_data["RegioS"] = ori_data["RegioS"].str.replace(" ","")
        ori_data["SoortMisdrijf"] = ori_data["SoortMisdrijf"].str.replace(" ","")

        # remove those "Niet in te delen" and "Buitenland" region etc
        ori_data = ori_data[(ori_data["RegioS"] != "RE99")&(ori_data["RegioS"] != "GM0998")
                            & (ori_data["RegioS"] != "LD99")&(ori_data["RegioS"] != "PV99")
                            & (ori_data["RegioS"] != "GM0999")]
        ori_data = ori_data.drop(columns=['Aangiften_2','Internetaangiften_3','ID'])
        ori_data = ori_data.fillna(0)
        data = ori_data.rename(columns={"SoortMisdrijf": "crime_code",
                    "RegioS": "region_code",
                    "Perioden":"period",
                    "GeregistreerdeMisdrijven_1":"crime_counts"})
        
        # Select the monthly/yearly data and convert the period to date format
        data = data[data["period"].str.contains(str(period))]
        data['period'] = pd.to_datetime(data['period'].str.replace(str(period),'',regex=True),format='%Y%m').dt.to_period(str(period))
        #
        # filter out the crime types we focus on
        data = data[data['crime_code'].isin(crimecodes)]

        print("Ended: Cleaning crime counts dataset")
        self.__dataframe_to_csv(data, save_folder, file_name)
        return data
    

    def __clean_population_dataset(self, ori_data: pd.DataFrame, period:str, 
                                   save_folder: Path, file_name: str)->pd.DataFrame:
        """
        Clean the population dataset.

        Parameters:
        - ori_data (pd.DataFrame): Original population dataset.
        - period (str): Period to filter the data.
        - save_folder (Path): Folder to save the cleaned dataset.
        - file_name (str): Name of the saved file.

        Returns:
        - pd.DataFrame: Cleaned population dataset.
        """
        print("Started: Cleaning population counts dataset.")
        # remove the sapces in the column RegioS
        ori_data["RegioS"] = ori_data["RegioS"].str.replace(" ","")
        
        # filter the PopulationAtTheEndofThePeriod and rename
        ori_data = ori_data.filter(items=["RegioS","Perioden","BevolkingAanHetEindeVanDePeriode_15"])
        data = ori_data.rename(columns={
        "RegioS":"region_code",
        "Perioden":"period",
        "BevolkingAanHetEindeVanDePeriode_15":"population"})

        # Select the monthly/yearly data and transfer the period to date format 
        data = data[data["period"].str.contains(str(period))]
        data['period'] = pd.to_datetime(data['period'].str.replace(str(period),'',regex=True),format='%Y%m').dt.to_period(str(period))
        print("Ended: Cleaning population counts dataset")
        self.__dataframe_to_csv(data, save_folder, file_name)
        return data

    def __log_update(self, log_table:pd.DataFrame, table_name:str, current_source_date:datetime=None, 
                    source_url:str="dataderden.cbs.nl", table_id:str="47013NED"):
        """
        This method logs updates for a specific table in the given log table.

        Args:
            log_table (pd.DataFrame): The log table to which updates are recorded.
            table_name (str): The name of the table being updated.
            current_source_date (datetime): The current date of the data source (optional).
            source_url (str): The URL of the data source (default is "dataderden.cbs.nl").
            table_id (str): The identifier of the table (default is "47013NED").

        Returns:
            pd.DataFrame: The updated log table.
        """
        # Get the current date from the data source if not provided
        if current_source_date is None:
            latest_source_info = cbsodata.get_info(table_id=table_id,catalog_url=source_url)
            current_source_date = datetime.strptime(latest_source_info["Modified"].split('T')[0], '%Y-%m-%d')

        # generate new record
        update_id = str(len(log_table[log_table["table_name"]==table_name])+1)
        new_record = pd.DataFrame({
            "update_id": [update_id],  
            "table_name": [table_name],
            "update_date": [current_source_date]
        })

        # add the record to the log
        log_table = pd.concat([log_table,new_record], ignore_index=True)

        return log_table

    def __generate_crimerates_dataset(self,crimes:pd.DataFrame,population:pd.DataFrame, 
                                    save_folder: Path, file_name: str)->pd.DataFrame:
        """
        Generate the crime rate dataset.

        Parameters:
        - crimes (pd.DataFrame): DataFrame containing crime data.
        - population (pd.DataFrame): DataFrame containing population data.
        - save_folder (Path): Folder to save the cleaned dataset.
        - file_name (str): Name of the saved file.

        Returns:
        - pd.DataFrame: Generated crime rate dataset.
        """
        print("Started: generating the crime rate dataset")
        # Merge the crime counts with population
        data = pd.merge(crimes, population, on=["region_code","period"])
        data = data.dropna()
        # Add crime rate column
        data["crime_rate"] = (data["crime_counts"].div(data["population"])*10000).round(5)

        # Process the region code data
        region = self.__get_region()
        # Merge with region code
        data = pd.merge(data, region,left_on="region_code", right_on="region_code")
        self.__dataframe_to_csv(data, save_folder, file_name)
        print("Ended: generating the crime rate dataset")
        return data

    def check_files_in_folder(self, folder_path, substring):
        """
        This method checks for files in a specified folder that contain a specific substring.

        Args:
            folder_path (str): The path to the folder to be searched.
            substring (str): The substring to be matched in the file names.

        Returns:
            list: A list of file names in the folder that contain the specified substring.
        """
        files = os.listdir(folder_path)
        # Filename matching
        matching_files = [file for file in files if fnmatch(file, f'*{substring}*')]
        return matching_files

    def __initialize_dataset(self):
        """
        This method initializes the dataset by fetching raw data from specified API URLs,
        cleaning the data, generating crime rates, and updating the dataset log.

        Returns:
            None
        """
        # Extract crime and population API URLs and table IDs from the configuration
        urls = self.api_url
        crimes_url = urls.get("crimes").get("url")
        crimes_table = urls.get("crimes").get("table_id")

        ppl_url = urls.get("population").get("url")
        ppl_table = urls.get("population").get("table_id")

        # Fetch raw crime data
        print("Start to request crimes dataset...(it may take up to 5 minutes)")
        crimes = self.__fetch_raw_data(f"https://{crimes_url}/ODataApi/odata/{crimes_table}/", 
                                "TypedDataSet", "crime_counts_ori")
        # Fetch raw population data
        print("Start to request population dataset...")
        population = self.__fetch_raw_data(f"https://{ppl_url}/ODataApi/odata/{ppl_table}/", 
                                     "TypedDataSet","population_ori")
        # Load crime codes from a JSON file
        with open(f"{data_path}/selected_crimes_codes.json",'r') as file:
            code = json.load(file)
        crimecodes=[]
        for item in code.items():
            crimecodes += list(item[1].keys())
        # Clean and save the crime dataset
        cleaned_crimes = self.__clean_crimes_dataset(crimes,
                            crimecodes = crimecodes,
                            period = "M",
                            save_folder=self.data_path,
                            file_name="crime_counts")
        # Clean and save the population dataset
        cleaned_population = self.__clean_population_dataset(population,
                            period = "M",
                            save_folder=self.data_path,
                            file_name="population")
        # Generate and save the crime rates dataset
        self.__generate_crimerates_dataset(crimes = cleaned_crimes, 
                                    population = cleaned_population,
                                    save_folder=self.data_path,
                                    file_name="crime_rates")
        
        # Initialize the table log which stores the date the dataset updated
        log = pd.DataFrame({
            "update_id": [],
            "table_name": [],
            "update_date": []
        })
        # Log updates for each dataset specified in the API configuration
        for dataset_name, details in self.api_url.items():
            url = details["url"]
            table_id = details["table_id"]
            log = self.__log_update(log, 
                    table_name = dataset_name,
                    source_url = url,
                    table_id = table_id)
        # Log the update for the crime_rates dataset
        log = self.__log_update(log_table=log,
                        table_name="crime_rates",
                        current_source_date= datetime.utcnow().date()
                        )
        # Save the update log to a CSV file
        self.__dataframe_to_csv(log,save_folder=self.data_path,file_name="update_log")
        print("Update log initialized")
        return

    def __fetch_data_from_api(self):
        """
        This method checks if the crime rates dataset is initialized.
        If not, it initializes the dataset by calling __initialize_dataset().

        Returns:
            None
        """
        if not self.check_files_in_folder(self.data_path, "crime_rates.csv"):
            self.__initialize_dataset()
        else:
            print("Crime rate dataset is already initialized.")
        return

    def __generate_date_filter(self, dataset_date, source_date):
        """
        This method generates a date filter string for querying data between two dates.

        Args:
            dataset_date (datetime): The date in the dataset for which the filter is generated.
            source_date (datetime): The latest available date in the data source.

        Returns:
            str: The date filter string for the specified date range.
        """
        # Convert the dataset date to a string in the format "YYYYMM"
        date_year_month = dataset_date.strftime("%YMM%m")
        # Initialize the date filter with the first date
        date_filter = f"$filter=(Perioden eq '{date_year_month}')"

        # Loop through each month until reaching the source date
        current_date = dataset_date
        while current_date < source_date:
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, current_date.day)
            else:
                next_month = current_date.month + 1
                current_date = current_date.replace(month=next_month)
            # Add the current year and month to the date filter
            current_year_month = current_date.strftime("%YMM%m")
            date_filter += f" or (Perioden eq '{current_year_month}')"
        
        return date_filter


    def __detect_updates(self, table_name:str, data_cat_url:str="dataderden.cbs.nl", dataset_id:str="47013NED"):
        """
        This method checks if a specific dataset is up to date by comparing its last update date with the data source's last update date.

        Args:
            table_name (str): The name of the dataset being checked.
            data_cat_url (str): The URL of the data source catalog (default is "dataderden.cbs.nl").
            dataset_id (str): The identifier of the dataset (default is "47013NED").

        Returns:
            pd.DataFrame or None: If updates are available, returns the new data; otherwise, returns None.
        """
        print(f"Check if the {table_name} dateset is up to date.")
        # Get the date of the dateset last updated
        log = pd.read_csv(f"{self.data_path}/update_log.csv")
        table_log= log[log["table_name"]==table_name]
        table_update = table_log[table_log["update_id"]==table_log["update_id"].max()]
        table_update_date =  pd.to_datetime(table_update.iloc[0,2])

        # Get the date of the date source last upadted
        cbsodata.options.catalog_url =  data_cat_url
        latest_info = cbsodata.get_info(table_id=dataset_id)
        source_update_date = datetime.strptime(latest_info["Modified"].split('T')[0], '%Y-%m-%d')

        # Check if the dataset is up to date
        if table_update_date < source_update_date:
            #  There aresome updates from the source
            print(f"The datasource is last updated at {source_update_date}. Updates are available.")
            # Generate date filter to fetch only the new data
            date_filter = self.__generate_date_filter(table_update_date,source_update_date)
            print(date_filter)
            new_data = pd.DataFrame(cbsodata.get_data(catalog_url=data_cat_url,
                                        table_id=dataset_id,
                                        filters=date_filter,
                                        typed=True))
            
            return new_data
        else:
            print(f"No updates from the {table_name} dateset.")
            return None

    def __date_parser(self,date_string):
        """
        This method parses a date string in Dutch format and returns a formatted date string in the 'YYYY-MM' format.

        Args:
            date_string (str): The date string in the format 'YYYY month'.

        Returns:
            str: The formatted date string in the 'YYYY-MM' format.
        """
        # Split the date string into components
        date_components = date_string.split()
        year = int(date_components[0])
        month_name = date_components[1]

        month_dict = {
            'januari': 1, 'februari': 2, 'maart': 3, 'april': 4,
            'mei': 5, 'juni': 6, 'juli': 7, 'augustus': 8,
            'september': 9, 'oktober': 10, 'november': 11, 'december': 12
        }
        month = month_dict.get(month_name.lower())  
        # Format the date as 'YYYY-MM'
        date_object = datetime.date(year, month, 1).strftime('%Y-%m')
        return date_object

    def __get_region(self)->pd.DataFrame:
        """
        This method reads the region code data from a CSV file.

        Returns:
            pd.DataFrame: The DataFrame containing region codes.
        """
        region = pd.read_csv(f"{self.data_path}/region_codes.csv")
        return region

    def __clean_updated_crimes(self, new_data:pd.DataFrame, crimecodes:list)->pd.DataFrame:
        """
        This method cleans the updated crime data by dropping unnecessary columns, renaming columns,
        handling missing values, and filtering for specified crime codes.

        Args:
            new_data (pd.DataFrame): The new crime data to be cleaned.
            crimecodes (list): List of crime codes to filter.

        Returns:
            pd.DataFrame: The cleaned crime data.
        """ 
        new_data = new_data.drop(columns=['Aangiften_2','Internetaangiften_3','ID'])
        new_data = new_data.rename(columns={"SoortMisdrijf": "crime_code",
                                            "RegioS": "region",
                                            "Perioden":"period",
                                            "GeregistreerdeMisdrijven_1":"crime_counts"
                                        })
        new_data['crime_counts'] = new_data['crime_counts'].replace('       .',0)
        new_data["period"] = new_data["period"].apply(self._date_parser)
        new_data["crime_code"] = new_data["crime_code"].str.split(' ').str[0]
        data = new_data[new_data['crime_code'].isin(crimecodes)]
        data = data.dropna()
        return data

    def __clean_updated_population(self, new_data:pd.DataFrame)->pd.DataFrame:
        """
        This method cleans the updated population data by selecting relevant columns,
        renaming columns, applying a date parser, and dropping NaN values.

        Args:
            new_data (pd.DataFrame): The new population data to be cleaned.

        Returns:
            pd.DataFrame: The cleaned population data.
        """
        new_data = new_data.filter(items=["RegioS","Perioden","BevolkingAanHetEindeVanDePeriode_15"])
        data = new_data.rename(columns={"RegioS":"region",
                                    "Perioden":"period",
                                    "BevolkingAanHetEindeVanDePeriode_15":"population"})
        data["period"] = data["period"].apply(self.__date_parser)
        data = data.dropna()
        return data

    def __update_crimerates_dataset(self, crimes:pd.DataFrame, population:pd.DataFrame,
                                save_folder: Path, file_name: str)->pd.DataFrame:
        """
        This method updates the crimerates dataset by merging crime and population data,
        handling missing values, calculating crime rates, and saving the updated dataset.

        Args:
            crimes (pd.DataFrame): The cleaned crime data.
            population (pd.DataFrame): The cleaned population data.
            save_folder (Path): The folder to save the updated dataset.
            file_name (str): The name of the updated dataset file.

        Returns:
            pd.DataFrame: The updated crimerates dataset.
        """
        # Merge crime and population data
        data = pd.merge(crimes, population, on=["region","period"])
        # Merge with region codes
        region = self.__get_region()
        data = pd.merge(data, region,on="region")
        data = data.dropna()
        # Filter out specific region codes
        data = data[(data["region_code"] != "RE99")&(data["region_code"] != "GM0998")
                            & (data["region_code"] != "LD99")&(data["region_code"] != "PV99")
                            & (data["region_code"] != "GM0999")]
        # Calculate crime rates
        data["crime_counts"] = pd.to_numeric(data["crime_counts"], errors="coerce")
        data["population"] = pd.to_numeric(data["population"], errors="coerce")
        data = data.dropna()
        data["crime_rate"] = (data["crime_counts"].div(data["population"])*10000).round(5)
        # Read the existing crimerates dataset
        crimerates = pd.read_csv(f"{save_folder}/{file_name}.csv")
        # Concatenate new and existing datasets, drop duplicates, and save the updated dataset
        updated_crimerates = pd.concat([crimerates, data], axis=0, ignore_index=True)
        new_crimerates = updated_crimerates.drop_duplicates()
        self.__dataframe_to_csv(new_crimerates, save_folder, file_name)
        return new_crimerates


    def __check_and_update_local_csv(self):
        """
        Check for updates in data sources and update local CSV files accordingly.

        Returns:
            None
        """
        new_data= []
        both_updated = True
        # Check for updates in each dataset specified in the API configuration
        for dataset, details in self.api_url.items():
            url = details["url"]
            table_id = details["table_id"]
            update = self.__detect_updates(table_name=dataset, 
                                    data_cat_url=url,
                                    dataset_id=table_id)
            if update is None:
                both_updated =  both_updated & False
            else:
                new_data.append(update)
        # If both datasets are updated, proceed with cleaning and updating
        if both_updated :
            # Load crime codes from a JSON file
            with open(f"{self.data_path}/selected_crimes_codes.json",'r') as file:
                code = json.load(file)
            crimecodes=[]
            for item in code.items():
                crimecodes += list(item[1].keys())
            # Clean and update the crime and population datasets
            new_crimes = self.__clean_updated_crimes(new_data=new_data[0],crimecodes=crimecodes)
            new_populations = self.__clean_updated_population(new_data = new_data[1])

            # Log updates for each dataset specified in the API configuration
            log = pd.read_csv(f"{self.data_path}/update_log.csv")
            for dataset, details in self.api_url.items():
                url = details["url"]
                table_id = details["table_id"]
                log = self.__log_update(log, 
                        table_name = dataset,
                        source_url = url,
                        table_id = table_id)
            # Update the crimerates dataset
            self.__update_crimerates_dataset(crimes = new_crimes,
                                    population = new_populations,
                                    save_folder=f"{data_path}/crime_rates",
                                    file_name="crime_rates")
            log = self.__log_update(log_table=log,
                            table_name="crime_rates",
                            current_source_date= datetime.utcnow().date()
                            )
            # Save the updated log
            self.__dataframe_to_csv(log, save_folder=data_path, file_name="update_log")
            
        else:
            print("No updates needed for crime rate dataset.")
        return

    def run_data_pipeline(self):
        """
        Run the entire data pipeline, including fetching data from the API and updating local CSV files.

        Returns:
            None
        """
        try:
            logging.info("Starting data pipeline")
            # Fetch initial data from the API if there is no local data 
            self.__fetch_data_from_api()
            # Check for updates in data sources and update local CSV files
            self.__check_and_update_local_csv()
        except Exception as e:
            logging.error(f"Error occurred: {e}", exc_info=True)
            raise e
        return 


def check_and_create_folder(folder_path: Path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")


if __name__ == "__main__":
    check_and_create_folder(data_path)
    check_and_create_folder(f"{data_path}/crime_rates")

    data_loader: DataLoader = DataLoader()
    data_loader.run_data_pipeline()
