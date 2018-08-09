from dotenv import dotenv_values, set_key, find_dotenv, get_key
from getpass import getpass
import os
import json

#
# if dotenv_path == '':
#     dotenv_path = '.env'
#     with open(dotenv_path, 'a'):
#         os.utime(dotenv_path)


def get_password(dotenv_path):
    if 'PASSWORD' not in dotenv_values(dotenv_path=dotenv_path):
        password = getpass('Please enter password to use for the cluster')
        _=set_key(dotenv_path, 'PASSWORD', password)


def write_json_to_file(json_dict, filename, mode='w'):
    with open(filename, mode) as outfile:
        json.dump(json_dict, outfile, indent=4, sort_keys=True)
        outfile.write('\n\n')