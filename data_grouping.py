import os
import tqdm
import pandas as pd
import  numpy as np

#data[(data['NACCID']==x[0]) | (data['MRIYR'] == x[1])]

def filter_mri(x,data):

    datas = data[(data['NACCID'] == x[0]) & (data['MRIYR'] == x[1])]
    if len(datas) !=0:
        return  list(x)

csv_path_mri="E:\\fed_work2\cobbinah11042020mri.csv"
csv_path_uds="E:\\fed_work2\cobbinah11042020.csv"
compare_path = "C:\\NACC"

read_mri = pd.read_csv(csv_path_mri)
read_uds = pd.read_csv(csv_path_uds)

mri_headers =["NACCID","NACCMRFI","MRIYR"]
uds_headers =["NACCID","VISITYR","DEMENTED","NACCMMSE","NACCUDSD"]

mri_data = read_mri[mri_headers]
uds_data = read_uds[uds_headers]



#slice_uds=uds_data.loc[uds_data['NACCID'].isin(mri_data['NACCID'])]

refined_data=uds_data.apply(lambda x: filter_mri(x,mri_data), axis=1 )

print(refined_data.dropna())



#print(slice_uds)

for  local_file in os.listdir(compare_path):
  pass



