import os
from FeatureSelection import FeatureSelection
import pandas as pd


bench=FeatureSelection("C",5501)

for csv in os.listdir("C:/Users/Kostas/Desktop/testspace2/csvs"):
	data = pd.read_csv("C:/Users/Kostas/Desktop/testspace2/csvs/"+csv, encoding = 'iso-8859-1')
	bench.split_data(data)

bench.uniform('single')

