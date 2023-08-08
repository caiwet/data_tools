from data_preprocess import *
from data_preprocess import *
from data_visual_tools import *
import pandas as pd
import os
import csv
import shutil

class DataPipeline:
    def __init__(self, root, output_data_dir, output_anno_dir, downsized_anno,
                 enlarged_anno, ori_anno, ori_data_dir):
        self.root = root
        self.output_data_dir = output_data_dir
        self.output_anno_dir = output_anno_dir
        self.downsized_anno = downsized_anno
        self.enlarged_anno = enlarged_anno
        self.ori_anno = ori_anno
        self.ori_data_dir = ori_data_dir

    def data_process(self):
        downsize_images(image_dir=self.ori_data_dir,
                        target_dir=self.output_data_dir,
                        resized_dim=1280)

        downsize_anno(ori_ann_file=self.ori_anno,
            output_file=self.output_anno_dir + self.downsized_anno,
            resized_dim=1280)

        make_bbox_nonzero(anno_dir=self.output_anno_dir + self.downsized_anno)

        enlarge_bbox(root=self.output_anno_dir,
                     annotation_file=self.downsized_anno,
                     out_file=self.enlarged_anno, factor=5)

        create_area(file=self.output_anno_dir+self.enlarged_anno)

    def data_split(self, mapping='/n/data1/hms/dbmi/rajpurkar/lab/hospital_data/annotation_image_name_mapping.csv'):
        mapping = pd.read_csv(mapping)
        for image_name in os.listdir(self.output_data_dir):
            image = Image.open(os.path.join(self.output_data_dir, image_name))

            image_name = image_name[:-4]
            institution = mapping[mapping['new_name']==image_name]['institution'].values[0]
            institution = institution.replace(' ', '_')

            out_dir = self.root+institution+'/images/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            image.save(os.path.join(out_dir, image_name+'.jpg'))


    def anno_split(self, mapping='/n/data1/hms/dbmi/rajpurkar/lab/hospital_data/annotation_image_name_mapping.csv'):
        mapping = pd.read_csv(mapping)
        f = open(self.output_anno_dir+self.enlarged_anno)
        data = json.load(f)
        id_to_ins = {}

        ins_to_anno = {}

        for img in data['images']:
            img['file_name'] = img['file_name'].replace('.png', '.jpg')
            image_name = img['file_name'][:-4]
            institution = mapping[mapping['new_name']==image_name]['institution'].values[0]
            institution = institution.replace(' ', '_')
            id_to_ins[img['id']] = institution

            anno_path = self.root+institution+'/annotations/annotations.json'
            if institution not in ins_to_anno.keys():
                ins_to_anno[institution] = {'info': data['info'], 'images': [img],
                                            'categories': data['categories'],
                                            'annotations': []}
            else:
                ins_to_anno[institution]['images'].append(img)

        for anno in data['annotations']:
            institution = id_to_ins[anno['image_id']]
            ins_to_anno[institution]['annotations'].append(anno)

        for ins, anno in ins_to_anno.items():
            print(ins)
            os.makedirs(self.root+ins+'/annotations', exist_ok=True)
            anno_path = self.root+ins+'/annotations/annotations.json'
            with open(anno_path, 'w') as outfile:
                json.dump(anno, outfile)

    def get_name_to_id(self):
        f = open(self.output_anno_dir+self.enlarged_anno)
        data = json.load(f)
        name_to_id = {}
        for img in data['images']:
            name_to_id[img['file_name'][:-4]] = img['id']
        return name_to_id

    def make_image_brighter(self, data_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for image_path in os.listdir(data_dir):
            image = Image.open(os.path.join(data_dir, image_path))
            img = np.array(image)
            img *= 2
            image = Image.fromarray(img)
            image.save(os.path.join(out_dir, image_path))


    def read_metadata(self, meta_dir = '/n/data1/hms/dbmi/rajpurkar/lab/hospital_data/Metadata'):
        name_to_id = self.get_name_to_id()
        for ins in os.listdir(meta_dir):
            meta_data = pd.read_csv(os.path.join(meta_dir, ins, 'extracted_metadata.csv'))
            ps_df = meta_data[['png_name', 'Rows',
                                 'Columns', 'PixelSpacing']]
            ps_df['pixel_spacing_x'] = meta_data['PixelSpacing'][:,0]
            ps_df['pixel_spacing_x'] = meta_data['PixelSpacing'][:,1]
            ps_df = ps_df.rename(columns={'png_name': 'image',
                                    'Columns': 'original_width',
                                    'Rows': 'original_height'})
            ps_df['have_metadata'] = 'True'
            ps_df['image_source'] = ins
            ps_df.loc[ps_df['pixel_spacing_x'].isna(), 'have_metadata'] = 'False'
            print(ins)
            print(ps_df['have_metadata'].value_counts())
            for index, row in ps_df.iterrows():
                ps_df.at[index, 'image_id'] = name_to_id[row['image']]
            # display(ps_df)


if __name__ == "__main__":
#     Sanity Check
#     f = open('/n/data1/hms/dbmi/rajpurkar/lab/ett/hospital_downsized/Austral/annotations/annotations.json')
#     data = json.load(f)
#     print(data.keys())
#     print(len(data['images']))
#     print(len(data['annotations']))
    root = '/n/data1/hms/dbmi/rajpurkar/lab/ett/hospital_downsized/'
    pipe = DataPipeline(root = root,
        output_data_dir = root + 'images/',
        output_anno_dir = root + 'annotations/',
        downsized_anno = 'hospital_downsized.json',
        enlarged_anno = 'hospital_downsized_enlarged5.json',
        ori_anno = '/n/data1/hms/dbmi/rajpurkar/lab/hospital_data/annotations.json',
        ori_data_dir = '/n/data1/hms/dbmi/rajpurkar/lab/hospital_data/Images')
    # pipe.data_process()
    # pipe.data_split()
    # pipe.anno_split()
    pipe.make_image_brighter(
        data_dir='/n/data1/hms/dbmi/rajpurkar/lab/ett/hospital_downsized/Newark_Beth_Israel_Medical_Center/images',
        out_dir='/n/data1/hms/dbmi/rajpurkar/lab/ett/hospital_downsized/Newark_Beth_Israel_Medical_Center/images_bright')
    # pipe.read_metadata()
