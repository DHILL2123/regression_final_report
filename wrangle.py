import os
import pandas as pd 
import numpy as np
from env import protocol, user, host, password, db
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, RFE, f_regression



mysqlcon = f"{protocol}://{user}:{password}@{host}/{db}"

######################## Zillow Data ###########
def get_connection(db, user, host, password, protocol):
    return f'{protocol}://{user}:{password}@{host}/{db}'
   

def get_zillow_data():
    filename = "zillow_exc_final_report.csv"
    mysqlcon=f"{protocol}://{user}:{password}@{host}/zillow"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql_query('''
                                
                                select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
                                taxamount, numberofstories, structuretaxvaluedollarcnt,
                                
                                propertyzoningdesc, transactiondate, yearbuilt, fips, propertylandusedesc, propertycountylandusecode,
                                
                                taxvaluedollarcnt

                                from properties_2017
                                left join predictions_2017
                                using(id)
                                left join typeconstructiontype
                                using (typeconstructiontypeid)
                                left join propertylandusetype
                                using (propertylandusetypeid)
                                left join storytype
                                using(storytypeid)
                                where propertylandusedesc like 'Single Family%%' and transactiondate between '2017-01-01%%' and '2017-12-31%%'
                                ''', mysqlcon)  
        

         # renaming column names to one's I like better
        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'year_built', 
                              'structuretaxvaluedollarcnt':"structure_taxvalue", "taxamount" : "tax_amount",
                              "propertyzoningdesc" : "zoning_desc", "transactiondate" : "trans_date", "propertylandusedesc":"prop_landuse_desc",
                              "propertycountylandusecode":"county_landuse_code"})

        # Write that dataframe to disk for later. Called "caching" the data for later.
    df.to_csv(filename)

        # Return the dataframe to the calling code
    return df


def wrangle_zillow(df):
    '''Wrangle zillow will clean the data and update missing values 
    for the df'''
   #Replace a whitespace sequence or empty with a NaN value and reassign this manipulation to df.
    df = df.replace(r'^\s*$', np.nan, regex=True)    
    #drop any duplicates in the df
    df = df.drop_duplicates()
    #filter column for single family homes
    df = df[df['propertylandusetypeid']==261.0]
    #drop redundant column
    df = df.drop(columns = 'propertylandusedesc')
    #filled nan values with 0 
    df = df.fillna(0)
    return df








###### Zillow Prepare ######################################
####### Split data #########################################
def train_test_split(df):
    x_train_and_validate, x_test = train_test_split(df, test_size=.2, random_state=123)
    
    x_train, x_validate = train_test_split(x_train_and_validate, test_size=.3, random_state=123)

    return x_train, x_validate, x_test, x_train_and_validate

################# Data Scalers ###########################################




def minmax_scaler(x_train, x_validate, x_test, numeric_cols):
    ######## Min Max Scaler (range calculations)
    scaler = sklearn.preprocessing.MinMaxScaler(copy=True)
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(x_train[numeric_cols])
    ### Apply to train, validate, and test
    x_train_scaled_array = scaler.transform(x_train[numeric_cols])
    x_validate_scaled_array = scaler.transform(x_validate[numeric_cols])
    x_test_scaled_array = scaler.transform(x_test[numeric_cols])

 # convert arrays to dataframes
    x_train_scaled = pd.DataFrame(x_train_scaled_array, 
                                  columns=numeric_cols).\
                                  set_index([x_train.index.values])

    x_validate_scaled = pd.DataFrame(x_validate_scaled_array, 
                                     columns=numeric_cols).\
                                     set_index([x_validate.index.values])

    x_test_scaled = pd.DataFrame(x_test_scaled_array, 
                                 columns=numeric_cols).\
                                 set_index([x_test.index.values])

    return x_train_scaled, x_validate_scaled, x_test_scaled

def standard_scaler(x_train,x_validate,x_test,numeric_cols):
    scaler = sklearn.preprocessing.StandardScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_validate_scaled = scaler.transform(x_validate)
    x_test_scaled = scaler.transform(x_test)
    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(x_train, bins=25, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(x_train_scaled, bins=25, ec='black')
    plt.title('Scaled')
    plt.show()
    return x_train_scaled, x_validate_scaled, x_test_scaled

def robust_scaler(x_train,x_validate,x_test,numeric_cols):
    scaler = sklearn.preprocessing.RobustScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_validate_scaled = scaler.transform(x_validate)
    x_test_scaled = scaler.transform(x_test)

    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(x_train, bins=25, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(x_train_scaled, bins=25, ec='black')
    plt.title('Scaled')
    plt.show()
    return x_train_scaled, x_validate_scaled, x_test_scaled


#############Student Data#####################################

def get_student_data():
    filename = "student_grades.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM student_grades', get_connection('school_sample'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df

def wrangle_grades(path):
    '''
    Read student_grades into a pandas DataFrame from mySQL,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    '''

    # Acquire data

    grades = get_student_data()

    # Replace white space values with NaN values.
    grades = grades.replace(r'^\s*$', np.nan, regex=True)

    # Drop all rows with NaN values.
    df = grades.dropna()

    # Convert all columns to int64 data types.
    df = df.astype('int')

    return df

'''Wrangles data from Zillow Database'''

##################################################Wrangle.py###################################################

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from env import user, password, host

#**************************************************Acquire*******************************************************

def acquire_zillow():
    ''' Acquire data from Zillow using env imports and rename columns'''
    
    url = f"mysql+pymysql://{user}:{password}@{host}/zillow"
    
    query = """
            
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017

    LEFT JOIN propertylandusetype USING(propertylandusetypeid)

    WHERE propertylandusedesc IN ("Single Family Residential",                       
                                  "Inferred Single Family Residential")"""

    # get dataframe of data
    df = pd.read_sql(query, url)
    
    
    # renaming column names to one's I like better
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'year_built',})
    return df

#**************************************************Remove Outliers*******************************************************

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#**************************************************Distributions*******************************************************

def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if col not in ['fips', 'year_built']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=7)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()
        
        
def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'taxamount']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()
        
#**************************************************Prepare*******************************************************

def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    # removing outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'area', 'tax_value', 'tax_amount','structure_taxvalue','numberofstories'])
    df['fips'] = df['fips'].replace([6037.0, 6059.0,6111.0], ['Los Angeles County, CA', 'Orange County, CA','Ventura County, CA'])

    # get distributions of numeric data
    #get_hist(df)
    #get_box(df)
    
    # converting column datatypes
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)

    df['zoning_desc']=LabelEncoder().fit_transform(df['zoning_desc'])
    df['trans_date']=LabelEncoder().fit_transform(df['trans_date'])
    df['year_built']=LabelEncoder().fit_transform(df['year_built'])
    df['prop_landuse_desc']=LabelEncoder().fit_transform(df['prop_landuse_desc'])
    df['county_landuse_code']=LabelEncoder().fit_transform(df['county_landuse_code'])
    df['fips']=LabelEncoder().fit_transform(df['fips'])


    
    # train/validate/test split
    train_validate, test = train_test_split(df,  test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using mode
    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['year_built','numberofstories','area','tax_amount','structure_taxvalue']])

    train[['year_built','numberofstories','area','tax_amount','structure_taxvalue']] = imputer.transform(train[['year_built','numberofstories','area','tax_amount','structure_taxvalue']])
    validate[['year_built','numberofstories','area','tax_amount','structure_taxvalue']] = imputer.transform(validate[['year_built','numberofstories','area','tax_amount','structure_taxvalue']])
    test[['year_built','numberofstories','area','tax_amount','structure_taxvalue']] = imputer.transform(test[['year_built','numberofstories','area','tax_amount','structure_taxvalue']])
    
    train['zoning_desc'] = train['zoning_desc'].fillna('Unknown')
    validate['zoning_desc'] = validate['zoning_desc'].fillna('Unknown')      
    test['zoning_desc'] = test['zoning_desc'].fillna('Unknown')
    train_validate['zoning_desc'] = train_validate['zoning_desc'].fillna('Unknown')
    

   
    return train, validate, test , train_validate  


#**************************************************Wrangle*******************************************************


def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(acquire_zillow())
    
    return train, validate, test

def select_kbest(X, y, k=2):
    '''
    X: dataframe of independent features
    y: single pandas Series (a target)
    k: kwarg, a number of k best features to select
    '''
    # make our kbest object:
    kbest = SelectKBest(f_regression, k=k)
    # fit it from x and y's relationships
    kbest.fit(X, y)
    # get the support values:
    mask = kbest.get_support()
    return X.columns[mask]