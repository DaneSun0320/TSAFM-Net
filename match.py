# 加载图像对
import glob
import json
import os
import cv2
import xml.etree.ElementTree as ET
from torchvision import transforms
from tqdm import tqdm

execp_list = []

def parse_json(json_path):
    with open(json_path, 'r') as fid:
        data = json.load(fid)
    return data


def xml_reader(filename):
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def get_all_components(img, objects, classes):
    components = []
    labels = []
    bar = tqdm(objects, desc="Extracting components")
    for obj in bar:
        name = obj['name']
        box = obj['bbox']
        p1 = (box[0], box[1])
        p2 = (box[2], box[3])
        p3 = (max(box[0], 15), max(box[1], 15))
        if "text" in name: continue
        name = name.replace('\t', ' ').replace('"', '').split(" ")[0]
        if name not in classes.keys():
            print(f"Warning: {name} not in classes")
            continue
        component = img[p1[1]:p2[1], p1[0]:p2[0]]
        components.append(component)
        labels.append(list(classes.keys()).index(name))
    return components, labels

def load_pairs(test_path, size, classes):
    for pair_path in os.listdir(test_path):
        images = glob.glob(os.path.join(test_path, pair_path, f'*.jpg'))
        images = [image for image in images if "pair" not in image.split("/")[-1]]
        xml = [image.replace(".jpg", ".xml") for image in images]
        try:
            t_image = cv2.imread(images[0])
            t_xml = xml_reader(xml[0])
            print(images)
            d_image = cv2.imread(images[1])
            d_xml = xml_reader(xml[1])
        except IndexError:
            execp_list.append(pair_path)

        t_components, t_labels = get_all_components(t_image, t_xml, classes)
        d_components, d_labels = get_all_components(d_image, d_xml, classes)

        if len(t_labels) != len(d_labels):
            print(f"Warning: {pair_path} has different number of components")
            continue

        yield t_components,d_components, t_labels,  d_labels
import torch
import torch.nn.functional as F

def batch_distance(x1, x2):
    cosine_similarity = F.cosine_similarity(x1.unsqueeze(1), x2.unsqueeze(0), dim=2)
    cosine_distance = 1 - cosine_similarity
    return cosine_distance

def match(vector_model, data_path, c_size, class_json, device):
    vector_model.eval()
    classes = parse_json(class_json)
    transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(c_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    for t_components, d_components, t_labels, d_labels in load_pairs(data_path, c_size, classes):

        t_components = [transform(component) for component in t_components]
        d_components = [transform(component) for component in d_components]

        t_components = torch.stack(t_components).to(device)
        d_components = torch.stack(d_components).to(device)

        t_labels = torch.tensor(t_labels, dtype=torch.uint8).to(device)
        d_labels = torch.tensor(d_labels, dtype=torch.uint8).to(device)

        match_acc = []
        cls_acc = []
        with torch.no_grad():
            features, _ = vector_model(t_components, d_components)
            t_features, d_features = features

            distance = batch_distance(t_features, d_features)

            min_distance = torch.min(distance, dim=1)[1]
            matches = min_distance == torch.arange(len(min_distance)).to(device)

            gt_matches = t_labels == d_labels
            min_indices = torch.argmin(distance, dim=0)

            d_label_pred = torch.zeros_like(d_labels)
            d_label_pred[min_indices] = t_labels
            accuracy = torch.sum(matches == gt_matches) / len(matches)
            match_acc.append(accuracy)
            print(f"match Accuracy: {accuracy}")

            wrong_idx = torch.where(matches == False)[0]

            wrong_distance = batch_distance(d_features[wrong_idx], t_features)
            wrong_min_distance = torch.min(wrong_distance, dim=1)[1]
            wrong_labels = t_labels[wrong_min_distance]
            d_label_pred[wrong_idx] = wrong_labels
            cls_accuracy = torch.sum(d_label_pred == d_labels) / len(d_labels)
            cls_acc.append(cls_accuracy)

    print(f"Match Accuracy: {sum(match_acc) / len(match_acc)}"
            f"Label Accuracy: {sum(cls_acc) / len(cls_acc)}")