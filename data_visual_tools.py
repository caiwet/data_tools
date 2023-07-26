import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFile

def view_gt_bbox(
    root="/home/ec2-user/segmenter/MAIDA/MIMIC_ETT_annotations",
    annotation_file='annotations.json',
    image_dir='PNGImages', target_dir='bbox3046'):
    # load annotations.json
    f = open(os.path.join(root, annotation_file))
    data = json.load(f)

    # image to id
    img_to_id = {}
    for img in data['images']:
        file_name = img['file_name'].replace('.dcm', '.png')
        img_to_id[file_name] = img['id']

    # id to ann
    id_to_ann = {}
    for i, ann in enumerate(data['annotations']):
        if ann['image_id'] not in id_to_ann:
            id_to_ann[ann['image_id']] = []
        id_to_ann[ann['image_id']].append(i)
    # print(id_to_ann)

    # draw bbox
    for file_path in os.listdir(os.path.join(root, image_dir)):
        image_path = os.path.join(root, image_dir, file_path)
        image = Image.open(image_path).convert("RGB")
        if file_path not in img_to_id.keys():
            continue
        if img_to_id[file_path] not in id_to_ann.keys():
            print(img_to_id[file_path])
            continue
        ann_idx = id_to_ann[img_to_id[file_path]]

        for idx in ann_idx:
            ann = data['annotations'][idx]
            if ann['category_id'] == 3048:
                continue
            xmin, ymin, w, h = ann["bbox"]
            xmax = xmin + w
            ymax = ymin + h
            bbox = [xmin, ymin, xmax, ymax]
            image = draw_single_bbox(image, bbox, ann['category_id'])

        image.save(os.path.join(root, target_dir, file_path))

def draw_single_bbox(image, bbox, id=None):
    labelled_img = ImageDraw.Draw(image)
    shapes = [bbox]
    if not id or id == 3046:
        col = 'red'
    elif id == 3047:
        col = 'blue'
    elif id == 3048:
        col = 'purple'
    elif id == 3049:
        col = 'orange'
    labelled_img.rectangle(shapes[0], outline=col, width = 5)
    return image

def view_labels(
    root="/home/ec2-user/segmenter/MAIDA/MIMIC_ETT_annotations",
    label_dir='enlarged10_1280/labels/train',
    image_dir='enlarged10_1280/images/train', target_dir='view_label',
    annotation_file='annotations_enlarged_10.json'):

    f = open(os.path.join(root, annotation_file))
    data = json.load(f)

    img_to_id = {}
    for img in data['images']:
        file_name = img['file_name'].replace('.dcm', '.jpg')
        img_to_id[file_name] = img['id']

    id_to_width = {}
    id_to_height = {}
    for img in data['images']:
        id_to_width[img['id']] = img['width']
        id_to_height[img['id']] = img['height']

    # draw bbox
    for file_path in os.listdir(os.path.join(root, image_dir)):
        label_path = file_path.replace('.jpg', '.txt')
        image_path = os.path.join(root, image_dir, file_path)
        image = Image.open(image_path).convert("RGB")
        label_path = os.path.join(root, label_dir, label_path)
        id = img_to_id[file_path]

        with open(label_path) as f:
            lines = f.readlines()
        for line in lines:
            line = line.split()
            xmid, ymid, w, h = float(line[1]), float(line[2]), float(line[3]), float(line[4])
            w *= id_to_width[id]
            h *= id_to_height[id]
            xmid *= id_to_width[id]
            ymid *= id_to_height[id]
            xmin = xmid - w/2
            ymin = ymid - h/2
            xmax = xmin + w
            ymax = ymin + h
            bbox = [xmin, ymin, xmax, ymax]
            colmap = {0: 3046, 1: 3047, 2: 3048}
            image = draw_single_bbox(image, bbox, id=colmap[int(line[0])])
            image.save(os.path.join(root, target_dir, file_path))
