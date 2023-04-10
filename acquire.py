from env import protocol, user, password, host, db
import os
import pandas as pd



#### Get titanic data functions ####
def get_titanic_data():
    filename = "titanic.csv"
    mysqlcon=f"{protocol}://{user}:{password}@{host}/{db}"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df_titanic = pd.read_sql(df_titanic = pd.read_sql_query("select * from passengers", mysqlcon))
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df_titanic.to_csv(filename)
        # Return the dataframe to the calling code
        return df_titanic

def get_connection(db, user, host, password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic():
    my_query="SELECT * FROM passengers"
    df = pd.read_sql(my_query, get_connection('titanic_db'))
    return df


### get iris data functions
def get_iris_data():
    filenameiris = "iris.csv"
    mysqlcon=f"{protocol}://{user}:{password}@{host}/{db}"

    if os.path.isfile(filenameiris):
        return pd.read_csv(filenameiris)
    else:
        # read the SQL query into a dataframe
        df_iris = pd.read_sql_query('''select * from measurements
        left join species 
        using (species_id);''', mysqlcon)
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df_iris.to_csv(filenameiris)
        # Return the dataframe to the calling code
        return df_iris
    


    
def get_telco_data():
    filenametelco = "telco.csv"
    mysqlcon=f"{protocol}://{user}:{password}@{host}/{db}"

    if os.path.isfile(filenametelco):
        return pd.read_csv(filenametelco)
    else:
        # read the SQL query into a dataframe
        df_telco_churn = pd.read_sql_query(
'''select * from customers
left join contract_types
using(contract_type_id)
left join internet_service_types
using (internet_service_type_id)
left join payment_types
using (payment_type_id);''', mysqlcon)

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df_telco_churn.to_csv(filenametelco)

        # Return the dataframe to the calling code
        return df_telco_churn
