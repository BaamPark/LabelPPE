import xml.etree.ElementTree as ET
import os

def main():

    xml_dir = "/home/beomseok/ppe_data/dataset/Vanishing Pen #3/Labels with person"

    with open('Vanishing Pen #3 complete label.txt', 'r') as file:
        for line in file.readlines():
            # print(line)
            parts = line.strip().split(', ')
            xml_name = parts[0].replace('.png', '.xml') # Assuming XML file has the same name as image but with .xml extension

            values = parts[1].strip().split()
            class_id = int(values[0]) # Convert label to integer
            coords = tuple(map(float, values[1:])) # Convert coordinates to float

            xml_path = os.path.join(xml_dir, xml_name)
            if class_id == 0:  # Assuming class ID for 'person' is 0
                update_xml(xml_path, "person", coords)

    for file in os.listdir(xml_dir):
        if file.endswith(".xml"):
            inspect_xml(os.path.join(xml_dir, file))

def convert_to_absolute(image_width, image_height, x_center, y_center, width, height):
    abs_width = width * image_width
    abs_height = height * image_height
    abs_x_center = x_center * image_width
    abs_y_center = y_center * image_height

    xmin = int(abs_x_center - (abs_width / 2))
    xmax = int(abs_x_center + (abs_width / 2))
    ymin = int(abs_y_center - (abs_height / 2))
    ymax = int(abs_y_center + (abs_height / 2))

    return xmin, ymin, xmax, ymax


def update_xml(xml_file, class_name, coords):
    if not os.path.exists(xml_file):
        print(f"XML file {xml_file} does not exist. Skipping...")
        return
    
    tree = ET.parse(xml_file)
    root = tree.getroot()

    width = int(root.find(".//width").text)
    height = int(root.find(".//height").text)
    xmin, ymin, xmax, ymax = convert_to_absolute(width, height, *coords)

    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = class_name
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"
    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(xmin)
    ET.SubElement(bndbox, "ymin").text = str(ymin)
    ET.SubElement(bndbox, "xmax").text = str(xmax)
    ET.SubElement(bndbox, "ymax").text = str(ymax)
    print("finish writing")
    tree.write(xml_file)


def get_bounding_box(obj):
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    return {"label": obj.find('name').text, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def overlap(box1, box2):
    inter_xmin = max(box1['xmin'], box2['xmin'])
    inter_ymin = max(box1['ymin'], box2['ymin'])
    inter_xmax = min(box1['xmax'], box2['xmax'])
    inter_ymax = min(box1['ymax'], box2['ymax'])
    
    if inter_xmin < inter_xmax and inter_ymin < inter_ymax:
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box2_area = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
        return inter_area / box2_area
    return 0


def inspect_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    persons = [obj for obj in root.findall(".//object[name='person']")]
    gas = [obj for obj in root.findall(".//object[name='ga']")]
    gis = [obj for obj in root.findall(".//object[name='gi']")]
    gas_and_gis = gas + gis
    
    for person_obj in persons:
        person_bbox = get_bounding_box(person_obj)
        renamed = False

        for ga_gi_obj in gas_and_gis:
            ga_gi_bbox = get_bounding_box(ga_gi_obj)
            if overlap(person_bbox, ga_gi_bbox) > 0.9:
                if ga_gi_bbox['label'] == 'ga':
                    person_obj.find('name').text = "person_ga"
                else:
                    person_obj.find('name').text = "person_gi"
                renamed = True
                break

        if not renamed:
            person_obj.find('name').text = "person_gc"

    tree.write(xml_file)



if __name__ == "__main__":
    main()