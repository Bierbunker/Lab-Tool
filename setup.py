from setuptools import setup
import site
import sys

site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

if __name__ == "__main__":
    setup_args = {
    'packages': ['labtool_ex2_data','labtool_ex2_data.input'],
    'package_dir': {
        'labtool_ex2_data': 'labtool_ex2_data',
        'labtool_ex2_data.input':'labtool_ex2_data/input'
    },
    'package_data':{'labtool_ex2_data':['labtool_ex2_data/*.bib','labtool_ex2_data/*.tex'],'labtool_ex2_data.input':['labtool_ex2_data/input/*tex']},
    'include_package_data': True
    }
    setup(**setup_args)
