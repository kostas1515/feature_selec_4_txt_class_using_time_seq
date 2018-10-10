import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
import csv

d = {'filename': None,'title': None, 'text': None, 'date': None, 'topic': None, 'industry': None, 'region': None}


for folder in os.listdir("C:/Users/Kostas/Desktop/School/Διπλωματική/rcv1"):
	if (folder.endswith(".py"))==False:
		with open("C:/Users/Kostas/Desktop/rcv1/csvs/"+folder+'.csv', 'w',newline='') as f:  # Just use 'w' mode in 3.x
			w = csv.DictWriter(f,d.keys())
			w.writeheader()
			for name in os.listdir(folder):
					if (name.endswith(".csv") or name.endswith(".py"))==False: 
						xml_data = ET.parse(folder+'/'+name)
						roots= xml_data.getroot()
						has_topic=0
						for codes in roots[-1].findall('codes'):
							if (codes.attrib['class']=='bip:topics:1.0'):
								has_topic=1
								break
						for metadata in roots[-1]:
							if (metadata.tag=='codes'):
								if (metadata.attrib['class']=='bip:topics:1.0'):
									topic=''
									text=''
									for child in metadata:
										topic=topic+ child.attrib['code'] +';'
									d['topic']=topic
									d['date']=roots.attrib['date']
									d['title']=roots[1].text
									d['filename']=roots.attrib['itemid']
									for child in roots.find('text'):
										text=text+child.text+' '
									d['text']=text
								if (has_topic==1):
									country=''
									industry=''
									if (metadata.attrib['class']=='bip:countries:1.0'):
										for child in metadata:
											country=country+ child.attrib['code'] +';'
										d['region']=country
									if (metadata.attrib['class']=='bip:industries:1.0'):
										for child in metadata:
											industry=industry+ child.attrib['code'] +';'
										d['industry']=industry
						if (has_topic==1):
							try :
								w.writerow(d)
							except UnicodeEncodeError:
								for k,v in d.items():
									d[k]=v.encode("utf-8")
								w.writerow(d)
								print (folder)
								print (name)
								print(d)	














