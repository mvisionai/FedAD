import os
import tqdm
import pandas as pd
import  numpy as np
import shutil

#data[(data['NACCID']==x[0]) | (data['MRIYR'] == x[1])]
csv_path_mri="/media/mvisionai/BACK_UP/fed_work2/cobbinah11042020mri.csv"
csv_path_uds="/media/mvisionai/BACK_UP/fed_work2/cobbinah11042020.csv"
path_save = "/media/mvisionai/BACK_UP/fed_work2/mri_p.npy"
destination_base = "/media/mvisionai/BACK_UP/fed_work2/NAC"
src_base = "/media/mvisionai/BACK_UP/fed_work2/NACC"
aibl_mri_path = "/media/mvisionai/BACK_UP/fed_work2/AIBL/AIBL-A/AIBL"
aibl_dia_csv = "/media/mvisionai/BACK_UP/fed_work2/AIBL/aibl_19Sep2019/Data_extract_3.3.0/aibl_pdxconv_01-Jun-2018.csv"
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

    return  np.load(path_save,allow_pickle=True)


def move_mri_nacc(data):

    error_c=0
    for dt in data:

        dt= np.asarray(dt)
        mri_split = str(dt[5]).split()


        m_class = dt[4]
        m_mri = mri_split[1]

        try:

         if m_class in [1,3,4]:

           move_util_nacc(m_class,m_mri)

        except Exception:

            error_c = error_c+1
            print("Error",error_c)
            continue



def move_mri_aibl():

    read_mri = pd.read_csv(aibl_dia_csv)
    read_mri = np.asarray(read_mri.loc[read_mri['VISCODE'] == 'bl'])

    for dt in read_mri:

        m_class = dt[3]
        data_id = dt[0]

        if m_class == 1:
            class_label = "NC"

        elif m_class == 2:
            class_label = "MCI"

        elif m_class == 3:
            class_label = "AD"


        destine_path = os.path.join(aibl_mri_path,class_label,str(data_id))
        source_path = os.path.join(aibl_mri_path,str(data_id))

        try:

           shutil.move(source_path, destine_path)
           #print(destine_path,source_path)
        except Exception:
            continue
def move_util_nacc(m_class,src_file):

    if m_class == 1:
        class_label = "NC"

    elif m_class == 3:
        class_label = "MCI"

    elif m_class == 4:
        class_label = "AD"

    src_file_split = str(src_file).split(".")

    rename_file = src_file_split[0]+"ni."+src_file_split[1]
    src_file = rename_file

    destine_path = os.path.join(destination_base,class_label,src_file)
    source_path = os.path.join(src_base,src_file)

    shutil.move(source_path, destine_path)
    #print(destine_path,source_path)



if __name__ == "__main__":
    #extracted_mri()
    #load_data = read_mri_npy()
    #move_mri_nacc(load_data)
    move_mri_aibl()

#for  local_file in os.listdir(compare_path):




