import pydicom as dicom
import matplotlib.pylab as plt
import os
from PIL import Image
import torchvision.transforms.functional as F
import cv2
import numpy as np
import json


def get_img_path(path="/home/ec2-user/efs/MAIDA/MIMIC-981/data/"):
    image_paths = []
    for dir in os.listdir(path=path):
        image_path = os.path.join(path, dir)
        image_paths.append(image_path)
    return image_paths


def convert_dcm(image_paths, output_path="MAIDA/data", PNG=True):
    for n, image in enumerate(image_paths):
        ds = dicom.dcmread(image)
        pixel_array_numpy = ds.pixel_array.astype(float)
        scaled_image = (
            np.maximum(pixel_array_numpy, 0) / pixel_array_numpy.max()
        ) * 255.0
        scaled_image = np.uint8(scaled_image)
        if PNG == False:
            image = image.replace(".dcm", ".jpg")
        else:
            image = image.replace(".dcm", ".png")
        img = Image.fromarray(scaled_image)
        img.save(os.path.join(output_path, image.split("/")[-1]))
        if n % 50 == 0:
            print("{} image converted".format(n))


def downsize_one_image(image_path, resized_dim):
    # breakpoint()
    image = Image.open(image_path)
    img = np.array(image)
    if img.max() > 255:
        img = (img / 256).astype("uint8")
    image = Image.fromarray(img).convert("RGB")
    w, h = image.size

    # crop
    if w > h:
        image = image.crop(
            ((w - h) / 2, 0, (w + h) / 2, h)
        )  # cut evenly from left and right
    else:
        image = image.crop(
            (0, (h - w) * 0.2, w, h - (h - w) * 0.8)
        )  # cut more from bottom and less from top

    # resize
    image = image.resize((resized_dim, resized_dim))
    return image


def downsize_images(image_dir, target_dir, resized_dim):
    os.makedirs(target_dir, exist_ok=True)
    # downsize images
    for file_path in os.listdir(image_dir):
        image_path = os.path.join(image_dir, file_path)
        image = downsize_one_image(image_path, resized_dim)
        image.save(os.path.join(target_dir, file_path))


def downsize_anno(ori_ann_file, output_file, resized_dim):
    f = open(ori_ann_file)
    data = json.load(f)
    # downsize gt bbox
    for i, ann in enumerate(data["annotations"]):
        # if ann['image_id'] == 2946144: # corrupted image
        #     continue
        if "bbox" not in ann:
            print(ann)
            continue
        for j in range(len(data["images"])):
            if data["images"][j]["id"] == ann["image_id"]:
                h = data["images"][j]["height"]
                w = data["images"][j]["width"]
                break
        curr_dim = min(w, h)
        if w > h:
            ann["bbox"][0] = ann["bbox"][0] - (w - h) / 2
        else:
            ann["bbox"][1] = ann["bbox"][1] - (h - w) * 0.2
        ann["bbox"][0] = ann["bbox"][0] * resized_dim / curr_dim
        ann["bbox"][1] = ann["bbox"][1] * resized_dim / curr_dim
        ann["bbox"][2] = ann["bbox"][2] * resized_dim / curr_dim
        ann["bbox"][3] = ann["bbox"][3] * resized_dim / curr_dim

    for i in range(len(data["images"])):
        data["images"][i]["width"] = resized_dim
        data["images"][i]["height"] = resized_dim
    # save target.json
    directory_path = os.path.dirname(output_file)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(output_file, "w+") as outfile:
        json.dump(data, outfile)


def get_stats(
    root="/home/ec2-user/segmenter/MAIDA/MIMIC_ETT_annotations",
    image_dir="ETTAnnotated",
):
    # find mean and std of images in folder
    mean = [0, 0, 0]
    std = [0, 0, 0]
    for file_path in os.listdir(os.path.join(root, image_dir)):
        image_path = os.path.join(root, image_dir, file_path)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        # image = image / 255
        mean += image.mean(axis=(0, 1))
        std += image.std(axis=(0, 1))
    mean /= len(os.listdir(os.path.join(root, image_dir)))
    std /= len(os.listdir(os.path.join(root, image_dir)))
    print("Mean of the dataset: ", mean)
    print("Std of the dataset: ", std)
    return mean, std


def get_reviewed_anno(
    ori_ann_file="/home/ec2-user/efs/MAIDA/MIMIC-1105-annotations.json",
    target_dir="MAIDA/data1000/annotations1105.json",
):
    f = open(ori_ann_file)
    data = json.load(f)
    repeated = []
    m = {}
    for ann in data["annotations"]:
        if ann["category_id"] == 3046 or ann["category_id"] == 3047:
            if ann["image_id"] in m.keys():
                repeated.append(ann["image_id"])
                # print(ann['image_id'])
                # print(m[ann['image_id']]['assignee'])
                # print(ann['assignee'])
            else:
                m[ann["image_id"]] = [0, 0, 0]

    final_ann = []
    for ann in data["annotations"]:
        if ann["image_id"] not in repeated:
            final_ann.append(ann)
        elif ann["assignee"] == "reviewer1@vinbrain.net":
            if ann["category_id"] == 3046 and m[ann["image_id"]][0] == 0:
                final_ann.append(ann)
                m[ann["image_id"]][0] += 1
            elif ann["category_id"] == 3047 and m[ann["image_id"]][1] == 0:
                final_ann.append(ann)
                m[ann["image_id"]][1] += 1
            elif ann["category_id"] == 3048 and m[ann["image_id"]][2] < 2:
                final_ann.append(ann)
                m[ann["image_id"]][2] += 1

    data["annotations"] = final_ann
    with open(target_dir, "w") as outfile:
        json.dump(data, outfile)


def enlarge_bbox(
    annotation_file="annotations.json",
    output_file="annotations_enlarged.json",
    factor=2,
):
    f = open(annotation_file)
    data = json.load(f)
    for i, ann in enumerate(data["annotations"]):
        if ann["image_id"] == 2946144:  # corrupted image
            continue
        if "bbox" not in ann:
            continue

        ann["bbox"][0] = ann["bbox"][0] - (ann["bbox"][2] / 2) * (factor - 1)
        ann["bbox"][1] = ann["bbox"][1] - (ann["bbox"][3] / 2) * (factor - 1)
        ann["bbox"][2] = ann["bbox"][2] * factor
        ann["bbox"][3] = ann["bbox"][3] * factor

    with open(output_file, "w") as outfile:
        json.dump(data, outfile)


def generate_gt_label(  # xmid, ymid, w, h
    root="/home/ec2-user/segmenter/MAIDA/MIMIC_ETT_annotations",
    annotation_file="annotations.json",
    image_dir="yolo/images",
    out_dir="yolo/labels",
):
    # load annotations.json
    f = open(os.path.join(root, annotation_file))
    data = json.load(f)

    # image to id
    img_to_id = {}
    for img in data["images"]:
        file_name = img["file_name"].replace(".dcm", ".png")
        img_to_id[file_name] = img["id"]

    # id to width, height
    id_to_width = {}
    id_to_height = {}
    for img in data["images"]:
        id_to_width[img["id"]] = img["width"]
        id_to_height[img["id"]] = img["height"]
    # id to ann
    id_to_ann = {}
    for i, ann in enumerate(data["annotations"]):
        if ann["image_id"] not in id_to_ann:
            id_to_ann[ann["image_id"]] = []
        id_to_ann[ann["image_id"]].append(i)

    id_to_class = {
        3046: 0,
        3047: 1,
        3048: 2,
        3049: 3,
        0: 0,
        1: 1,
        2: 2,
        3: 3,
    }
    problem_id = []
    for file_path in os.listdir(os.path.join(root, image_dir)):
        if file_path not in img_to_id.keys():
            print(f"no {file_path}")
            continue
        out_file = open(
            os.path.join(root, out_dir, file_path).replace(".png", ".txt"), "w+"
        )
        id = img_to_id[file_path]
        if id not in id_to_ann.keys():
            problem_id.append(id)
            continue

        ann_idx = id_to_ann[id]
        bbox = None
        for idx in ann_idx:
            ann = data["annotations"][idx]
            if "bbox" not in ann.keys():
                print(ann)
                continue
            xmin, ymin, w, h = ann["bbox"]
            # print(id_to_height[id])
            bbox = [
                (xmin + w / 2) / id_to_width[id],
                (ymin + h / 2) / id_to_height[id],
                w / id_to_width[id],
                h / id_to_height[id],
            ]
            out_file.write(
                " ".join([str(a) for a in (id_to_class[ann["category_id"]], *bbox)])
                + "\n"
            )

    print("Problem ID: ", problem_id)


def make_bbox_nonzero(anno_dir="MIMIC-2k-annotations.json"):
    f = open(anno_dir)
    data = json.load(f)
    cnt = 0
    # print(data['annotations'])
    for ann in data["annotations"]:
        if "bbox" not in ann:
            data["annotations"].remove(ann)
            continue
        x, y, w, h = ann["bbox"]
        if w == 0.0:
            w = 1.0
            cnt += 1
        if h == 0.0:
            h = 1.0
            cnt += 1
        ann["bbox"] = [x, y, w, h]
    print("Zero boxes:", cnt)
    with open(anno_dir, "w") as f:
        json.dump(data, f)


def create_area(file="/home/ec2-user/segmenter/MAIDA/data1000/val_annotations.json"):
    f = open(file)
    data = json.load(f)
    for ann in data["annotations"]:
        if "bbox" not in ann:
            data["annotations"].remove(ann)
            continue
        x, y, w, h = ann["bbox"]
        ann["area"] = w * h
    with open(file, "w") as f:
        json.dump(data, f)


def make4classes(
    file="/home/ec2-user/segmenter/MAIDA/data1000/train_annotations.json",
    outfile="/home/ec2-user/segmenter/MAIDA/data1000/train_annotations_4cls.json",
):
    f = open(file)
    data = json.load(f)
    for i in data["categories"]:
        if i["id"] == 3048:
            data["categories"].remove(i)

    data["categories"].append(
        {"id": 3048, "name": "left clavicle", "supercategory": ""}
    )
    data["categories"].append(
        {"id": 3049, "name": "right clavicle", "supercategory": ""}
    )
    id_ann_map = {}
    for ann in data["annotations"]:
        if ann["category_id"] != 3048:
            continue
        if ann["image_id"] not in id_ann_map.keys():
            id_ann_map[ann["image_id"]] = ann
            continue
        prev = id_ann_map[ann["image_id"]]
        if prev["bbox"][0] < ann["bbox"][0]:
            ann["category_id"] = 3049
        else:
            prev["category_id"] = 3049
    with open(outfile, "w") as f:
        json.dump(data, f)


def train_val_test_split(
    root="/home/ec2-user/segmenter/MAIDA/data1000/downsized",
    image_dir="images",
    label_dir="labels",
    target_dir="downsized/split",
):
    src = os.path.join(root, image_dir)

    for file in os.listdir(src):
        name = file[:-4]
        if name in list(train_info["imageID"]):
            shutil.copy(
                os.path.join(src, file),
                os.path.join(root, target_dir, image_dir, "train"),
            )
        elif name in list(val_info["imageID"]):
            shutil.copy(
                os.path.join(src, file),
                os.path.join(root, target_dir, image_dir, "val"),
            )
        elif name in list(test_info["imageID"]):
            shutil.copy(
                os.path.join(src, file),
                os.path.join(root, target_dir, image_dir, "test"),
            )
        else:
            print(name)
    src = os.path.join(root, label_dir)
    for file in os.listdir(src):
        name = file[:-4]
        if name in list(train_info["imageID"]):
            shutil.copy(
                os.path.join(src, file),
                os.path.join(root, target_dir, label_dir, "train"),
            )
        elif name in list(val_info["imageID"]):
            shutil.copy(
                os.path.join(src, file),
                os.path.join(root, target_dir, label_dir, "val"),
            )
        elif name in list(test_info["imageID"]):
            shutil.copy(
                os.path.join(src, file),
                os.path.join(root, target_dir, label_dir, "test"),
            )
        else:
            print(name)


def get_test_anno(
    anno_file="/home/ec2-user/segmenter/MAIDA/data1000/annotations_downsized1105.json",
    out_file="/home/ec2-user/segmenter/MAIDA/data1000/annotations1105.json",
    split_info="/home/ec2-user/segmenter/ETT_Evaluation/data_split/train_mimic_only.csv",
    mimic=False,
    ranzcr=False,
    split="test",
):
    # column = 'imageID'
    column = "FileName"
    f = open(anno_file)
    data = json.load(f)
    test_images = []
    info = pd.read_csv(split_info)

    if mimic:
        test_info = info[(info["Split"] == split) & (info["Source"] == "MIMIC")]
    elif ranzcr:
        test_info = info[(info["Split"] == split) & (info["Source"] == "RANZCR")]
    else:
        raise ValueError("No dataset specified.")
    for image in data["images"]:
        image["file_name"] = image["file_name"].replace(".dcm", ".png")
        if image["file_name"][:-4] in list(test_info[column]):
            test_images.append(image)

    test_coco = {
        "info": data["info"],
        "categories": data["categories"],
        "images": test_images,
        "annotations": data["annotations"],
    }

    with open(out_file, "w") as f:
        json.dump(test_coco, f)


def move_ranzcr_images(
    src_dir="/home/ec2-user/efs/RANZCR/data",
    csv_file="/home/ec2-user/segmenter/ETT_Evaluation/data_split/all_data_split.csv",
    output_dir="/home/ec2-user/segmenter/ETTDATA/Test/RANZCR",
    split="test",
    source="RANZCR",
):
    os.makedirs(output_dir, exist_ok=True)

    # Loop through the csv file
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check if the image is in one of the subdirectories
            for subdir_name in os.listdir(src_dir):
                subdir_path = os.path.join(src_dir, subdir_name)
                # print(subdir_path)
                if not os.path.isdir(subdir_path):
                    continue
                img_path = os.path.join(subdir_path, row["FileName"])
                img_path += ".png"
                if os.path.isfile(img_path):
                    # If the image is in the csv file and meets the criteria, copy it to the new directory
                    if row["Source"] == source and row["Split"] == split:
                        shutil.copy(img_path, output_dir)


def move_mimic_images(
    src_dir="/home/ec2-user/segmenter/MAIDA/all_images",
    csv_file="/home/ec2-user/segmenter/ETT_Evaluation/data_split/all_data_split.csv",
    output_dir="/home/ec2-user/segmenter/ETTDATA/Test/MIMIC",
    split="test",
    source="MIMIC",
):
    os.makedirs(output_dir, exist_ok=True)
    info = pd.read_csv(csv_file)

    target_imgs = list(
        info[(info["Source"] == source) & (info["Split"] == split)]["FileName"]
    )

    for img_path in os.listdir(src_dir):
        if img_path.replace(".png", "") in target_imgs:
            image = Image.open(os.path.join(src_dir, img_path))
            # img_path = img_path.replace('.png', '.jpg')
            image.save(os.path.join(output_dir, img_path))


def combine_anno(anno1, anno2, output_anno):
    with open(anno1) as f1:
        data1 = json.load(f1)
    with open(anno2) as f2:
        data2 = json.load(f2)

    data = {}
    data["categories"] = data1["categories"]  # same for mimic and ranzcr
    data["images"] = data1["images"]
    data["images"].extend(data2["images"])

    data["annotations"] = data1["annotations"]
    data["annotations"].extend(data2["annotations"])

    with open(output_anno, "w") as f:
        json.dump(data, f)
