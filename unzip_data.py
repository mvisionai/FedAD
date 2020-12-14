import zipfile
import os


def unzip_data(path_to_zip_file,directory_to_extract_to=None):

    for zip_file in os.listdir(path_to_zip_file):

        zip_split = zip_file.split(".")[0]
        directory_path = os.path.join(path_to_zip_file,zip_split)
        zip_path = os.path.join(path_to_zip_file,zip_file)

        print(zip_file)

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(directory_path)
        exit(1)




if  __name__ == "__main__":
    unzip_data("E:\\fed_work2\\NACC")