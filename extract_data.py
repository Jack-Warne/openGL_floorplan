import os
from glob import glob
import pandas as pd
from functools import reduce
from xml.etree import ElementTree as et

# Load all xml files in the folder
xml_list = glob('./pascal data/*.xml')
# Data cleaning: replace '\\' with '/'
xmlfiles = list(map(lambda x: x.replace('\\', '/'), xml_list))

def extract_text(filename):
    # Read xml file; we need to extract: filename, size (width, height), object (name, xmin, xmax, ymin, ymax)
    tree = et.parse(filename)
    root = tree.getroot()

    image_name = root.find('filename').text
    width = root.find('size').find('width').text
    height = root.find('size').find('height').text

    objs = root.findall('object')
    parser = []

    for obj in objs:
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        xmax = bndbox.find('xmax').text
        ymin = bndbox.find('ymin').text
        ymax = bndbox.find('ymax').text
        parser.append([image_name, width, height, name, xmin, xmax, ymin, ymax])
    return parser

parser_all = list(map(extract_text, xmlfiles))

# Convert the parsed data into a DataFrame
df = pd.DataFrame(reduce(lambda x, y: x + y, parser_all), columns=['image_name', 'width', 'height', 'name', 'xmin', 'xmax', 'ymin', 'ymax'])

print(df)
print(df.shape)
print(df['name'].value_counts())