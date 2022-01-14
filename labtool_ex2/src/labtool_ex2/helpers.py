import math

from pathlib import Path
from shutil import copy #info to setup files with templates
import xlsxwriter #info to create standard excel file for documentation while in the lab
from datetime import date
from shutil import rmtree #! only for testing remove later

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))

def init_folder(root_path, root_folder_name):
    """Init empty folders for a typical Latex-File with python scripts in it's own folder.
        - <Name>.latex/       ... main folder for latex 
            <Name>.latex/img/ ... folder for images
            <Name>.latex/out/ ... folder for pdf outputs  
        - <Name>.python/      ... folder for small python scripts

    Args:
        root_path ([string]): Path at which folder structure should be initialized 
        root_folder_name ([string]): Name of the root folder
    """
    path = Path(f"{root_path}/{root_folder_name}")
    if not path.exists():
        print(f"Project-Folder with Name {root_folder_name} doesn't exist!")
        Path(f"{root_path}").mkdir() #create root
        #create subfolders
        latex_p = Path(f"{path}.latex")
        Path(f"{path}.pyhton").mkdir()
        latex_p.mkdir()
        img_p=Path(f"{latex_p}/img")
        out_p=Path(f"{latex_p}/out")
        img_p.mkdir()
        out_p.mkdir()
        
        #create default files!
        workbook = xlsxwriter.Workbook(f"{path}_notes.xlsx")
        worksheet_overview = workbook.add_worksheet("overview")
        worksheet_overview.write(0,0,root_folder_name)
        cur_date = date.today().strftime("%d.%m.%Y")
        worksheet_overview.write(0,2,f"Date: {cur_date}")
        workbook.add_worksheet("data")
        workbook.add_worksheet("notes")
        workbook.close()

        #fix for file copy installation path of package is needed. 
        copy("""labtool_ex2/data/templates/template.bib""",latex_p)
        copy("""labtool_ex2/data/templates/template.tex""",latex_p)
        Path(f"{latex_p}/input").mkdir()
        copy("""labtool_ex2/data/templates/input/shared_preamble.tex""",latex_p)
        copy("""labtool_ex2/data/templates/input/tabularray-environments.tex""",latex_p)


        return
    print(f"Project-Folder with Name {root_folder_name} already existing!")
    return

if __name__ == "__main__": #! only for testing delete later
    TEST_PATH = """/home/elias/Code/Lab-Tool/pathtest_folder""" #"""/home/etschgi1/CODE/UNI/Lab-Tool/pathtest_folder"""
    TEST_NAME = "hello_name"
    init_folder(TEST_PATH,TEST_NAME)
    #info cleanup -- just for testing
    try:
        p = Path(f"{TEST_PATH}")
        if p.exists:
            input()
            rmtree(TEST_PATH)
        else:
            raise("!!! TEST failed ROOT not created!")
    except Exception:
        pass
