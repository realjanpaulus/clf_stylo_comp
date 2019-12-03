import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize
import re

from typing import Dict, List, Tuple
import argparse
import logging
from pathlib import Path

### texts_to_csv logging handler ###
logging.basicConfig(level=logging.INFO, filename="../logs/texts_to_csv.log", filemode="w")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s: %(message)s")
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


def get_file_list(path: str) -> List[str]:
    return [x.stem for x in Path(path).glob("**/*.txt") if x.is_file()]

def hasNumbers(inputString: str) -> bool:
    return any(char.isdigit() for char in inputString)

def get_metadata(filename: str, file: str) -> List:
    
    author = ""
    title = ""
    year = ""

    filename_split = filename.split("-", 1)
    author = filename_split[0].replace("_", " ")[:-1]
    
    #filename_elements = re.findall(r"\w+", "".join(filename_split[1]))
    filename_elements = re.findall(r"\w+(?:-\w+)+|\w+", "".join(filename_split[1]))
    
    if len(filename_elements) > 2:
        if hasNumbers(filename_elements[-1]):
            year = filename_elements[-1]
            joined_title = "".join(filename_elements[:-1])
            title = joined_title.replace("_", " ")[1:-1]
        else:
            year = "0"
            joined_title = "".join(filename_elements)
            title = joined_title.replace("_", " ")[1:-1]
    elif len(filename_elements) == 2:
        if hasNumbers(filename_elements[1]):
            year = filename_elements[1]
            title = filename_elements[0].replace("_", " ")[1:-1]
        else:
            year = "0"
            joined_title = "".join(filename_elements)
            title = joined_title.replace("_", " ")[1:-1]
    elif len(filename_elements) == 1:
        if hasNumbers(filename_elements[0]):
            year = filename_elements[0]
            title = "no_title"
        else:
            year = "0"
            title = filename_elements[0].replace("_", " ")[1:-1]
    elif len(filename_elements) == 0:
        year = "0"
        title = "no_title"
        

    factor = (len(author) + len(title) + 4)*3
    metalist = [author, title, year]
    
    #split txt-file in two parts by a factor and replace
    #author-, title- and year-informations
    file_p1 = file[:factor]
    file_p2 = file[factor:]
    
    for entry in metalist:
        if entry in file_p1:
            file_p1 = file_p1.replace(entry, "")

    file = file_p1 + file_p2
    file = file.replace("━", "") #replace stroke
    
    tok_text = word_tokenize(file)
    textlength = len(tok_text)
    text = " ".join(tok_text)
    return metalist + [textlength, text]


def texts_to_df(path: str) -> str:
    d = {}
    file_list = get_file_list(path)

    # pare file_list and add meta data with text to dict
    for c, filename in enumerate(file_list):
        logging.info(f"Text {c+1} von {len(file_list)}.")

        file_dir = path + "/" + filename + ".txt"
        with open(file_dir, encoding="utf-8") as f:
            tmp_file = f.read()
            d[filename] = get_metadata(filename, tmp_file)


    # dict to dataframe
    df = pd.DataFrame.from_dict(d, orient="index").reset_index()
    df.columns = ["filename", "author", "title", 
                  "year", "textlength", "text"]
    return df

def main():

    df = texts_to_df(args.path)
    df.to_csv("../data/corpus.csv", index=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog="texts_to_csv", description="Stores txt-files in a DataFrame.")
    parser.add_argument("path", type=str, help="Path to the txt-files directory.")
    args = parser.parse_args()

    main()