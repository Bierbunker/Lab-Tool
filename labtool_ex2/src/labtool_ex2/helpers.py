import math

from pathlib import Path
from shutil import rmtree  #! only for testing remove later


def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier


def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))


def init_folder(root_path, root_folder_name):
    """Init empty folders for a typical Latex-File with python scripts in it's own folder.

    Args:
        root_path ([string]): Path at which folder structure should be initialized
        root_folder_name ([string]): Name of the root folder
    """
    path = Path(f"{root_path}/{root_folder_name}")
    if not path.exists():
        print(f"Project-Folder with Name {root_folder_name} doesn't exist!")
        Path(f"{root_path}").mkdir()  # create root
        # create subfolders
        latex_f = Path(f"{path}.latex")
        python_f = Path(f"{path}.pyhton")
        latex_f.mkdir()
        python_f.mkdir()
        return
    print(f"Project-Folder with Name {root_folder_name} already existing!")
    return


def _args(*_nargs, **_kwargs):
    """Collect call parameters from a call"""
    return _nargs, _kwargs


def split_into_args_kwargs(s: str) -> tuple[list[str], dict[str, str]]:
    _nargs = list()
    _kwargs = dict()
    for param in s.split(","):
        if "=" in param:
            key, val = param.split("=")
            key = key.strip()
            val = val.strip()
            _kwargs[key] = val
            continue
        if param:
            _nargs.append(param)
    return _nargs, _kwargs


if __name__ == "__main__":  #! only for testing delete later
    TEST_PATH = """/home/etschgi1/CODE/UNI/Lab-Tool/pathtest_folder"""
    TEST_NAME = "hello"
    init_folder(TEST_PATH, TEST_NAME)
    # info cleanup -- just for testing
    try:
        p = Path(f"{TEST_PATH}")
        if p.exists:
            rmtree(TEST_PATH)
        else:
            raise Exception("!!! TEST failed ROOT not created!")
    except Exception:
        pass
