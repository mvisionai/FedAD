import os
import tqdm
import pandas as pd
import  numpy as np
import shutil

#data[(data['NACCID']==x[0]) | (data['MRIYR'] == x[1])]
csv_path_mri="E:\\fed_work2\cobbinah11042020mri.csv"
csv_path_uds="E:\\fed_work2\cobbinah11042020.csv"
path_save = "E:\\fed_work2\mri_p.npy"
destination_base = "E:\\fed_work2\\NAC"
src_base = "E:\\fed_work2\\NACC"
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
        mri_split = str(dt[5]).split()


        m_class = dt[4]
        m_mri = mri_split[1]

        try:

         if m_class in [1,3,4]:

           move_util(m_class,m_mri)

        except Exception:
            raise



def move_util(m_class,src_file):

    if m_class == 1:
        class_label = "NC"

    elif m_class == 3:
        class_label = "MCI"

    elif m_class == 4:
        class_label = "AD"

    destine_path = os.path.join(destination_base,class_label,src_file)
    source_path = os.path.join(src_base,src_file)

    try:

     #shutil.move(source_path, destine_path)
     print(destine_path,source_path)

    except Exception:
        pass

if __name__ == "__main__":
    #extracted_mri()
    load_data = read_mri_npy()
    move_mri(load_data)

#for  local_file in os.listdir(compare_path):




