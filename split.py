import os
import shutil
import random
import math
import json

def split_anno(anno_file, out_dir, data_split_dict):
    f = open(anno_file)
    data = json.load(f)
    id_to_file = {}
    train_anno = {
        'images': [],
        'annotations': [],
        'info': data['info'],
        'categories': data['categories']
    }
    val_anno = {
        'images': [],
        'annotations': [],
        'info': data['info'],
        'categories': data['categories']
    }
    test_anno = {
        'images': [],
        'annotations': [],
        'info': data['info'],
        'categories': data['categories']
    }

    for img in data['images']:
        file = img['file_name']
        id_to_file[img['id']] = file
        if file in data_split_dict['train']:
            train_anno['images'].append(img)
        elif file in data_split_dict['val']:
            val_anno['images'].append(img)
        elif file in data_split_dict['test']:
            test_anno['images'].append(img)
        else:
            raise ValueError("file not found in split")

    for anno in data['annotations']:
        # if 'image_id' not in id_to_file.keys():
        #     print('image_id')
        #     continue
        file = id_to_file[anno['image_id']]
        if file in data_split_dict['train']:
            train_anno['annotations'].append(anno)
        elif file in data_split_dict['val']:
            val_anno['annotations'].append(anno)
        elif file in data_split_dict['test']:
            test_anno['annotations'].append(anno)
        else:
            raise ValueError("file not found in split")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'train_annotations.json'), 'w') as outfile:
        json.dump(train_anno, outfile)
    with open(os.path.join(out_dir, 'val_annotations.json'), 'w') as outfile:
        json.dump(val_anno, outfile)
    with open(os.path.join(out_dir, 'test_annotations.json'), 'w') as outfile:
        json.dump(test_anno, outfile)



def split_images(data_dir, out_dir, train_ratio, test_ratio, val_ratio):
    train_dir = os.path.join(out_dir, 'train')
    test_dir = os.path.join(out_dir, 'test')
    val_dir = os.path.join(out_dir, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    all_images = os.listdir(data_dir)
    random.shuffle(all_images)

    total_images = len(all_images)
    train_split = math.floor(total_images * train_ratio)
    test_split = math.floor(total_images * test_ratio)

    train_images = all_images[:train_split]
    test_images = all_images[train_split:train_split+test_split]
    val_images = all_images[train_split+test_split:]

    data_split_dict = {
        'train': train_images,
        'test': test_images,
        'val': val_images
    }

    for image in train_images:
        src_path = os.path.join(data_dir, image)
        dest_path = os.path.join(train_dir, image)
        shutil.copy(src_path, dest_path)

    for image in test_images:
        src_path = os.path.join(data_dir, image)
        dest_path = os.path.join(test_dir, image)
        shutil.copy(src_path, dest_path)

    for image in val_images:
        src_path = os.path.join(data_dir, image)
        dest_path = os.path.join(val_dir, image)
        shutil.copy(src_path, dest_path)
    return data_split_dict




if __name__ == "__main__":
    data_directory = "/n/data1/hms/dbmi/rajpurkar/lab/ett/hospital_downsized/Chiang_Mai_University"
    train_ratio = 0.7
    test_ratio = 0.2
    val_ratio = 0.1

    random.seed(123)

    data_split_dict = split_images(
        data_dir = os.path.join(data_directory, 'images'),
        out_dir = os.path.join(data_directory, 'split/images'),
        train_ratio = train_ratio,
        test_ratio = test_ratio,
        val_ratio = val_ratio)

    split_anno(
        anno_file = os.path.join(data_directory, 'annotations/annotations.json'),
        out_dir = os.path.join(data_directory, 'split/annotations'),
        data_split_dict = data_split_dict)

    print("Completed.")



