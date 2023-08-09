import pandas as pd
import ast
import os

def convert_str_to_floats(s):
    lst = ast.literal_eval(s)
    return [float(x) for x in lst]

def extract_metadata_file(filepath, folder_name="N/A"):
    """Extract metadata from a single file for ETT project.
    
    Args:
        filepath: path to metadata file.
    
    Returns:
        metadata: a pandas dataframe containing metadata.
    """
    raw_file = pd.read_csv(filepath)
    metadata = pd.DataFrame()

    # Load relevant metadata.
    metadata['image'] = raw_file['png_name']
    metadata['pixel_spacing_x'], metadata['pixel_spacing_y'] = zip(*raw_file['ImagerPixelSpacing'].apply(convert_str_to_floats))
    metadata['image_source'] = folder_name
    metadata['original_width'] = raw_file['Rows']
    metadata['original_height'] = raw_file['Columns']
    metadata['image_id'] = raw_file['filename']
    return metadata

def extract_metadata_dir(dirpath):
    # Go through all subfolders in this path and extract metadata.
    metadata = pd.DataFrame()
    for folder in os.listdir(dirpath):
        subfolder_path = os.path.join(dirpath, folder)
        if os.path.isdir(subfolder_path):
            metadata = metadata.append(extract_metadata_file(os.path.join(subfolder_path, 'extracted_metadata.csv'), folder))
    return metadata

if __name__ == "__main__":
    filepath = "./extracted_metadata.csv"
    metadata = extract_metadata_file(filepath)
    # save metadata to csv
    metadata.to_csv("./metadata.csv", index=False)
    print(metadata)