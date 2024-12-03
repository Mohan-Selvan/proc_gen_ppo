import numpy as np
import json

def write_dictionary_to_file(data):
    file_path = "data.json"
    print(f"Writing data to file : {file_path}")
    with open(file_path, "w") as write_file:
        json.dump(data, write_file)
    
    print("File written successfully!")