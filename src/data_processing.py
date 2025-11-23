import pandas as pd
import numpy as np


def process_brazil_data(input_filepath: str, output_filepath: str = None) -> pd.DataFrame:
    df = pd.read_csv(input_filepath)
    df_brazil = df[df['Country Name'] == 'Brazil'].groupby('Year')['No of Suicides'].sum().reset_index()
    
    if output_filepath:
        df_brazil.to_csv(output_filepath, index=False)
    
    return df_brazil

def process_japan_data(input_filepath: str, output_filepath: str = None) -> pd.DataFrame:
    df = pd.read_csv(input_filepath)
    df_japan = df[df['Country Name'] == 'Japan'].groupby('Year')['No of Suicides'].sum().reset_index()
    
    if output_filepath:
        df_japan.to_csv(output_filepath, index=False)
    
    return df_japan
