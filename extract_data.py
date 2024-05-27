import os
from glob import glob
import pandas as pd
from functools import reduce
from xml.etree import ElementTree as et
from shutil import move

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
df = pd.DataFrame(reduce(lambda x, y: x + y, parser_all), columns=['filename', 'width', 'height', 'name', 'xmin', 'xmax', 'ymin', 'ymax'])

#print(df)
#print(df.shape)
#print(df['name'].value_counts())

cols = ['width', 'height', 'xmin', 'xmax','ymin','ymax']
df[cols] = df[cols].astype(int)
#print(df.info())

# Define centre x and centre y
df['centre_x'] = ((df['xmin'] + df['xmax']) / 2) / df['width']
df['centre_y'] = ((df['ymin'] + df['ymax']) / 2) / df['height']

# Define width and height
df['w'] = (df['xmax'] - df['xmin']) / df['width']
df['h'] = (df['ymax'] - df['ymin']) / df['height']
#print(df.head())

# Split the data
images = df['filename'].unique()
img_df = pd.DataFrame(images, columns=['filename'])
img_train = tuple(img_df.sample(frac=0.9)['filename']) # shuffle and pick 90% of the images in the dataset
img_test = tuple(img_df.query('filename not in @img_train')['filename']) # rest of data set for testing

train_df = df.query('filename in @img_train')
test_df = df.query('filename in @img_test')

# Can't train text in model so need to convert the obj into ID
# Label encoding is needed

def labelEncoding(x):
    labels={'door': 0,
            'window': 1, 
            'toilet': 2,
            'basin': 3,
            'bathtub': 4,
            'shower': 5,
            'sink': 6,
            'stairs': 7,
            'cupboard-door': 8,
            'cupboard door': 9
            }
    return labels[x]

train_df.loc[:, 'id'] = train_df['name'].apply(labelEncoding)
test_df.loc[:, 'id'] = test_df['name'].apply(labelEncoding)

train_folder = 'pascal data/train'
test_folder = 'pascal data/test'

# Save images and labels in text
if not os.path.exists(train_folder):
    os.mkdir(train_folder)
if not os.path.exists(test_folder):
    os.mkdir(test_folder)

cols = ['filename', 'id', 'centre_x', 'centre_y', 'w', 'h']
groupby_obj_train = train_df[cols].groupby('filename')
groupby_obj_test = test_df[cols].groupby('filename')

def saveData(filename, folder_path, group_obj):
    # Move image
    src = os.path.join('pascal data', filename)
    dst = os.path.join(folder_path, filename)
    move(src, dst)

    # Save labels
    text_filename = os.path.join(folder_path, os.path.splitext(filename)[0] + '.txt')
    group_obj.get_group(filename).set_index('filename').to_csv(text_filename, sep=' ', index=False, header=False)

filename_series = pd.Series(groupby_obj_train.groups.keys())
filename_series.apply(saveData, args=(train_folder, groupby_obj_train))

filename_series_test = pd.Series(groupby_obj_test.groups.keys())
filename_series_test.apply(saveData, args=(test_folder, groupby_obj_test))
