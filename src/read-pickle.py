import sys
import os
import _pickle as pickle
from pickle import UnpicklingError
import json


def convert_dict_to_json(file_path):
    # with open(file_path, 'rb') as fpkl:
    #     try:
    #         data = pickle.load(fpkl)
    #         print(data)
    #     except UnpicklingError as e:
    #         print(e)
    with open(file_path, 'rb') as fpkl, open('%s.json' % file_path, 'w', encoding='UTF-8') as fjson:
        data = pickle.load(fpkl)
        json.dump(data, fjson, ensure_ascii=False, sort_keys=True, indent=4)


def main():
    file_path = '../data/result/user.pickle'
    print("Processing %s ..." % file_path)
    convert_dict_to_json(file_path)


if __name__ == '__main__':
    main()
