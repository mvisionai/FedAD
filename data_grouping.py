import os
import tqdm
import pandas as pd
import  numpy as np

#data[(data['NACCID']==x[0]) | (data['MRIYR'] == x[1])]
csv_path_mri="E:\\fed_work2\cobbinah11042020mri.csv"
csv_path_uds="E:\\fed_work2\cobbinah11042020.csv"
path_save = "E:\\fed_work2\mri_p.npy"

compare_path = "C:\\NACC"

mri_headers =["NACCID","NACCMRFI","MRIYR"]
uds_headers =["NACCID","VISITYR","DEMENTED","NACCMMSE","NACCUDSD"]


def filter_mri(x,data):

    datas = data[(data['NACCID'] == x[0]) & (data['MRIYR'] == x[1])]
    if len(datas) !=0:
        return  list(x)+ datas['NACCMRFI']


def extracted_mri():
    # slice_uds=uds_data.loc[uds_data['NACCID'].isin(mri_data['NACCID'])]

    read_mri = pd.read_csv(csv_path_mri)
    read_uds = pd.read_csv(csv_path_uds)

    uds_data = read_uds[uds_headers]
    mri_data = read_mri[mri_headers]

    refined_data = uds_data.apply(lambda x: filter_mri(x, mri_data), axis=1)

    numpy_convert = np.asarray(refined_data.dropna())
    np.save(path_save,numpy_convert)

def read_mri_npy():

    return  np.load(path_save)


def move_mri(data):

    for dt in data:

        dt= np.asarray(dt)
        print(dt[5])

        exit(1)
        m_class = dt[4]
        m_mri = dt[6]
        class_label=""

        if m_class == 1:
            class_label = "NC"
        elif m_class == 3:
            class_label = "MCI"
        elif m_class == 4:
            class_label = "AD"



if __name__ == "__main__":
    #extracted_mri()
    load_data = read_mri_npy()
    move_mri(load_data)

#for  local_file in os.listdir(compare_path):




