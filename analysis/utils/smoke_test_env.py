import os

import pandas as pd

from analysis.utils.config import (

    PROJECT_ROOT,

    DATA_RAW_DIR,

    ANALYSIS_SETTINGS,

    COUNTRIES,

)

from analysis.data.wrds_helper import save_sample_global_daily_prices

from analysis.data.yf_helper import download_monthly_prices





def run_env_smoke_test():

    print("===========================================")

    print("      ENVIRONMENT SMOKE TEST (CAPM)        ")

    print("===========================================\n")



    # ---------------------------------------------------

    # 1. Basic path & config test

    # ---------------------------------------------------

    print("Project root:", PROJECT_ROOT)

    print("Raw data directory:", DATA_RAW_DIR)

    print("Analysis start date:", ANALYSIS_SETTINGS.start_date)

    print("Analysis end date:", ANALYSIS_SETTINGS.end_date)

    print("Countries configured:", list(COUNTRIES.keys()))

    print("\n")



    # ---------------------------------------------------

    # 2. Test WRDS access & saving a sample file

    # ---------------------------------------------------

    print("Testing WRDS connection and sample fetch...")

    try:

        sample_path = save_sample_global_daily_prices(

            n_obs=3,

            filename="wrds_comp_global_daily_sample_env.csv",

        )

        print(f"WRDS SUCCESS — sample saved to {sample_path}\n")

    except Exception as e:

        print("WRDS FAILED — cannot retrieve WRDS sample.")

        print("Error:", str(e))

        print("\n")



    # ---------------------------------------------------

    # 3. Test Yahoo Finance API for DAX index (^GDAXI)

    # ---------------------------------------------------

    print("Testing yfinance download for ^GDAXI...")

    try:
        # download_monthly_prices requires start and end parameters
        start_date = ANALYSIS_SETTINGS.start_date
        end_date = ANALYSIS_SETTINGS.end_date
        df = download_monthly_prices("^GDAXI", start=start_date, end=end_date)

        print("Last rows of downloaded data:\n", df.tail())

        print("YF SUCCESS — data received.\n")

    except Exception as e:

        print("YF FAILED — cannot download from yfinance.")

        print("Error:", str(e))

        print("\n")



    print("===========================================")

    print("Smoke test complete.")

    print("===========================================\n")





if __name__ == "__main__":

    run_env_smoke_test()

