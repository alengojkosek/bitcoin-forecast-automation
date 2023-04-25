import pandas as pd 
from cryptocmd import CmcScraper

# initialise scraper without time interval for max historical data
scraper = CmcScraper("BTC")

df = scraper.get_dataframe()

df.to_csv('data/raw/raw_data.csv', index=False)