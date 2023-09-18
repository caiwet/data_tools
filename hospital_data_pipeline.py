from constants import (
    ANNO_ANNOTATION_FIELD,
    ANNO_CATEGORIES_FIELD,
    ANNO_FILE_NAME_FIELD,
    ANNO_ID_FIELD,
    ANNO_IMAGE_FIELD,
    ANNO_IMAGE_ID_FIELD,
    ANNO_INFO_FIELD,
    MAPPING_INSTITUITION_COL,
    MAPPING_NEW_NAME_COL,
)
from data_preprocess import *
import pandas as pd
import yaml
import os
import shutil

from utils import remove_file_extension


class DataPipeline:
    def __init__(
        self,
        input_image_dir,
        input_anno_dir,
        output_image_dir,
        downsized_anno_dir,
        enlarged_anno_dir,
        resized_dim,
        enlarge_ratio,
    ):
        self.input_image_dir = input_image_dir
        self.input_anno_dir = input_anno_dir
        self.output_image_dir = output_image_dir
        self.downsized_anno_dir = downsized_anno_dir
        self.enlarged_anno_dir = enlarged_anno_dir
        self.resized_dim = resized_dim
        self.enlarge_ratio = enlarge_ratio

    def data_process(self):
        downsize_images(
            image_dir=self.input_image_dir,
            target_dir=self.output_image_dir,
            resized_dim=self.resized_dim,
        )

        downsize_anno(
            ori_ann_file=self.input_anno_dir,
            output_file=self.downsized_anno_dir,
            resized_dim=self.resized_dim,
        )

        make_bbox_nonzero(anno_dir=self.downsized_anno_dir)

        enlarge_bbox(
            annotation_file=self.downsized_anno_dir,
            output_file=self.enlarged_anno_dir,
            factor=self.enlarge_ratio,
        )

        create_area(file=self.enlarged_anno_dir)

    def data_split_by_instituitions(
        self,
        annotation_root,
        mapping,
    ):
        mapping = pd.read_csv(mapping)
        for image_name in os.listdir(self.output_image_dir):
            image_id = remove_file_extension(image_name)
            institution = mapping[mapping[MAPPING_NEW_NAME_COL] == image_id][
                MAPPING_INSTITUITION_COL
            ].values[0]
            institution = institution.replace(" ", "_")

            instituition_out_dir = annotation_root + institution + "/images/"
            if not os.path.exists(instituition_out_dir):
                os.makedirs(instituition_out_dir)

            # Copy the image to the corresponding institution folder
            shutil.copy(
                os.path.join(self.output_image_dir, image_name),
                os.path.join(instituition_out_dir, image_name),
            )

    def anno_split_by_instituitions(
        self,
        annotation_root,
        mapping,
    ):
        mapping = pd.read_csv(mapping)
        f = open(self.enlarged_anno_dir)
        data = json.load(f)
        id_to_ins = {}

        ins_to_anno = {}

        for img in data[ANNO_IMAGE_FIELD]:
            image_name = remove_file_extension(img[ANNO_FILE_NAME_FIELD])
            institution = mapping[mapping[MAPPING_NEW_NAME_COL] == image_name][
                MAPPING_INSTITUITION_COL
            ].values[0]
            institution = institution.replace(" ", "_")
            id_to_ins[img[ANNO_ID_FIELD]] = institution

            if institution not in ins_to_anno.keys():
                ins_to_anno[institution] = {
                    "info": data[ANNO_INFO_FIELD],
                    "images": [img],
                    "categories": data[ANNO_CATEGORIES_FIELD],
                    "annotations": [],
                }
            else:
                ins_to_anno[institution]["images"].append(img)

        for anno in data[ANNO_ANNOTATION_FIELD]:
            institution = id_to_ins[anno[ANNO_IMAGE_ID_FIELD]]
            ins_to_anno[institution]["annotations"].append(anno)

        for ins, anno in ins_to_anno.items():
            print(ins)
            os.makedirs(annotation_root + ins + "/annotations", exist_ok=True)
            anno_path = annotation_root + ins + "/annotations/annotations.json"
            with open(anno_path, "w") as outfile:
                json.dump(anno, outfile)

    def get_name_to_id(self):
        f = open(self.output_anno_dir + self.enlarged_anno)
        data = json.load(f)
        name_to_id = {}
        for img in data[ANNO_IMAGE_FIELD]:
            name_to_id[remove_file_extension(img[ANNO_FILE_NAME_FIELD])] = img["id"]
        return name_to_id

    # backup function, not necessary
    def make_image_brighter(self, data_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for image_path in os.listdir(data_dir):
            image = Image.open(os.path.join(data_dir, image_path))
            img = np.array(image)
            img *= 2
            image = Image.fromarray(img)
            image.save(os.path.join(out_dir, image_path))


def main():
    # read config yaml file into root
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    root = config["root"]

    # output paths
    output_dir = root + config["output_dir"]
    output_image_dir = output_dir + config["output_image_dir"]
    downsized_anno_dir = output_dir + config["downsized_anno_dir"]
    enlarged_anno_dir = output_dir + config["enlarged_anno_dir"]

    # input paths
    input_anno_dir = root + config["input_anno_dir"]
    input_image_dir = root + config["input_image_dir"]
    anno_image_mapping = root + config["anno_image_mapping"]

    # create data pipeline object
    pipe = DataPipeline(
        resized_dim=config["resized_dim"],
        enlarge_ratio=config["enlarge_ratio"],
        output_image_dir=output_image_dir,
        downsized_anno_dir=downsized_anno_dir,
        enlarged_anno_dir=enlarged_anno_dir,
        input_anno_dir=input_anno_dir,
        input_image_dir=input_image_dir,
    )

    # process data
    pipe.data_process()
    # split data
    pipe.data_split_by_instituitions(
        annotation_root=config["annotation_root"], mapping=anno_image_mapping
    )
    # split annotation
    pipe.anno_split_by_instituitions(
        annotation_root=config["annotation_root"], mapping=anno_image_mapping
    )


if __name__ == "__main__":
    main()
