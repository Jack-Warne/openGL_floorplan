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
df = pd.DataFrame(reduce(lambda x, y: x + y, parser_all), columns=['image_name', 'width', 'height', 'name', 'xmin', 'xmax', 'ymin', 'ymax'])

#print(df)
#print(df.shape)
#print(df['name'].value_counts())

cols = ['width', 'height', 'xmin', 'xmax','ymin','ymax']
df[cols] = df[cols].astype(int)
#print(df.info)

# define centre x and centre y
df['centre_x'] = ((df['xmin']+ df['xmax'])/2)/df['width']
df['centre_y'] = ((df['ymin']+ df['ymax'])/2)/df['height']

# define width and height
df['w'] = (df['xmax']- df['xmin'])/df['width']
df['h'] = (df['ymax']- df['ymin'])/df['height']
#print(df.head())

# split the data 
images = df['filename'].unique()
img_df = pd.DataFrame(images, columns=['filename'])
img_train = tuple(img_df.sample(frac=0.9)['filename']) # shuffle and pick 90% of the images in the dataset
img_test = tuple(img_df.query(f'filname not in {img_train}')['filename']) # rest of data set for testing

train_df = df.query(f'filname in {img_train}')
test_df = df.query(f'filname in {img_test}')

# can't train text in model so need to convert the obj into ID
# label encoding is needed

def labelEncoding(x):
    labels={'door':0,
            'window' : 1, 
            'toilet' : 2,
            'basin' : 3,
            'bathtub': 4,
            'shower': 5,
            'sink' : 6,
            'stairs': 7,
            'cupboard-door': 8,
            'cupboard door': 9
            }
    return labels[x]

train_df = train_df['name'].apply(labelEncoding)
test_df =  test_df['name'].apply(labelEncoding)


train_folder = 'pascal data/train'
test_folder = 'pascal data/test'

# save images and labels in text
if os.path.exists('pascal data/train') == False:    
    os.mkdir(train_folder)
if os.path.exists('pascal data/test') == False:    
    os.mkdir(test_folder)


