import os
import argparse
import json

import numpy as np
import glob
import PIL.Image
from PIL import ImageDraw

class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path="./coco.json"):
        """
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json
        """
        self.labelme_json = labelme_json
        self.dir_name = os.path.split(labelme_json[0])[0]
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, "r") as fp:
                data = json.load(fp)
                self.images.append(self.image(data, num))
                for shapes in data["shapes"]:
                    label = shapes["label"]
                    if label not in self.label:
                        self.label.append(label)
                    points = shapes["points"]
                    shape_type = shapes['shape_type']
                    self.annotations.append(self.annotation(points, label, num, shape_type))
                    self.annID += 1

        # Sort all text labels so they are in the same order across data splits.
        self.label.sort()
        for label in self.label:
            self.categories.append(self.category(label))
        for annotation in self.annotations:
            annotation["category_id"] = self.getcatid(annotation["category_id"])

    def image(self, data, num):
        image = {}
        height, width = data['imageHeight'], data['imageWidth']
        image["height"] = height
        image["width"] = width
        image["id"] = num
        image["file_name"] = os.path.join(self.dir_name, data["imagePath"])

        self.height = height
        self.width = width

        return image

    def category(self, label):
        category = {}
        category["supercategory"] = label
        category["id"] = len(self.categories)+1
        category["name"] = label
        return category

    def annotation(self, points, label, num, shape_type):
        annotation = {}
        contour = np.array(points)
        x = contour[:, 0]
        y = contour[:, 1]
        area = self.height * self.width
        annotation["bbox"] = list(map(float, self.getbbox(points, shape_type)))
        x = annotation['bbox'][0]
        y = annotation['bbox'][1]
        w = annotation['bbox'][2]
        h = annotation['bbox'][3]
        if shape_type == 'rectangle':
            annotation['segmentation'] = [[x, y, x+w, y, x+w, y+h, x, y+h]] # at least 6 points
        elif shape_type == 'polygon':
            points = [np.asarray(points).flatten().tolist()]
            annotation['segmentation'] = points
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = num

        annotation["category_id"] = label  # self.getcatid(label)
        annotation["id"] = self.annID
        return annotation

    def getcatid(self, label):
        for category in self.categories:
            if label == category["name"]:
                return category["id"]
        print("label: {} not in categories: {}.".format(label, self.categories))
        exit()
        return -1

    def getbbox(self, points, shape_type):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):

        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco["images"] = self.images
        data_coco["categories"] = self.categories
        data_coco["annotations"] = self.annotations
        return data_coco

    def save_json(self):
        print("save coco json")
        self.data_transfer()
        self.data_coco = self.data2coco()

        print(self.save_json_path)
        os.makedirs(
            os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
        )
        json.dump(self.data_coco, open(self.save_json_path, "w"), indent=4)
        cat_path = os.path.join(os.path.split(self.save_json_path)[0], 'classes.txt')
        with open(cat_path, 'wb') as f:
            f.writelines(map(lambda x: x + '\n' + x, self.label))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="labelme annotation to coco data json file.",
        epilog = """
        Example:
            python labelme2coco.py path/to/labelme/dir --output train.json
        """
    )
    parser.add_argument(
        "labelme_directory",
        help="Directory to labelme images and annotation json files.",
        type=str
    )
    parser.add_argument(
        "--output", help="Output json file path.", default="trainval.json"
    )

    args = parser.parse_args()
    labelme_json = glob.glob(os.path.join(args.labelme_directory, "*.json"))
    labelme2coco(labelme_json, args.output)
