import pandas as pd
import ast
import os
import json

def convert_str_to_floats(s):
    """Convert string "['x', 'y']" to [x, y]."""
    lst = ast.literal_eval(s)
    return [float(x) for x in lst]

def fix_format(df):
    """Fix dataframe format in-place.
    
    The type of fix depends on the dataset:
        - Fill in missing values of column ImagerPixelSpacing with 
          values from column PixelSpacing.
    """
    df['ImagerPixelSpacing'] = df['ImagerPixelSpacing'].fillna(df['PixelSpacing'])

def extract_metadata_file(filepath, folder_name="N/A"):
    """Extract metadata from a single file for ETT project.
    
    Args:
        filepath: path to metadata file.
    
    Returns:
        metadata: a pandas dataframe containing metadata.
    """
    raw_file = pd.read_csv(filepath)
    fix_format(raw_file)
    metadata = pd.DataFrame()

    # Load relevant metadata.
    metadata['image'] = raw_file['png_name']
    metadata['pixel_spacing_x'], metadata['pixel_spacing_y'] = zip(*raw_file['ImagerPixelSpacing'].apply(convert_str_to_floats))
    metadata['image_source'] = folder_name
    metadata['have_metadata'] = True
    metadata['original_width'] = raw_file['Rows']
    metadata['original_height'] = raw_file['Columns']
    metadata['image_id'] = raw_file['filename']
    return metadata

def extract_metadata_dir(
    dirpath, 
    annotation_file_path,
    skip_folder=[]
):
    """Extract metadata from all subfolders in a directory for ETT project.
    
    Go through all subfolder in dirpath, except for the subfolders specified by skip_folder.

    Args:
        dirpath: path to metadata.
        annotation_file_path: path to annotation file.
        skip_folder: name of the subfolders to be skipped.
    
    Returns:
        metadata: a pandas dataframe containing metadata of all subfolders.
    """
    # Go through all subfolders in this path and extract metadata.
    metadata = pd.DataFrame()
    for folder in os.listdir(dirpath):
        subfolder_path = os.path.join(dirpath, folder)
        if folder not in skip_folder and os.path.isdir(subfolder_path):
            print("=============", folder)
            metadata = metadata.append(extract_metadata_file(os.path.join(subfolder_path, 'extracted_metadata.csv'), folder))
    
    # Load annotation file.
    annotation = json.load(open(annotation_file_path, 'r'))
    name_to_id = {image['file_name']: image['id'] for image in annotation['images']}
    for i in range(len(metadata)):
        image_name = metadata['image'].iloc[i]
        if image_name+".png" in name_to_id:
            metadata['image_id'].iloc[i] = name_to_id[image_name+".png"]
        else:
            print(image_name, "not in annotation file")
    return metadata

if __name__ == "__main__":
    dirpath = "/n/data1/hms/dbmi/rajpurkar/lab/hospital_data/Metadata/"
    annotation_file_path = "/n/data1/hms/dbmi/rajpurkar/lab/hospital_data/annotations.json"
    skip_folder = ['.DS_Store', 'Austral']
    metadata = extract_metadata_dir(
        dirpath,
        annotation_file_path,
        skip_folder,
        )
    metadata.to_csv("./pixel_spacing_10_hospitals.csv", index=False)
    print(metadata)