import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_processing import process_brazil_data, process_japan_data


def main():
    df_brazil = process_brazil_data(
        input_filepath='data/combined_processed_data.csv',
        output_filepath='data/brazil_aggregated.csv'
    )
    df_japan = process_japan_data(
        input_filepath='data/combined_processed_data.csv',
        output_filepath='data/japan_aggregated.csv'
    )
    print(df_brazil)
    print(df_japan)

if __name__ == "__main__":
    main()
