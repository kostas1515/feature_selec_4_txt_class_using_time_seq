import os
from FeatureSelection import FeatureSelection
import pandas as pd


bench=FeatureSelection("C",12852)

for csv in os.listdir("C:/Users/Kostas/Desktop/testspace2/csvs"):
	data = pd.read_csv("C:/Users/Kostas/Desktop/testspace2/csvs/"+csv, encoding = 'iso-8859-1')
	bench.split_data(data)

bench.rdf(12)
print(bench.rdf_rel_pool)

