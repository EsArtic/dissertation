import sys
import os
import os.path
import xml.etree.ElementTree as ET

def parse(filename):

    tree = ET.ElementTree(file = filename)
    root = tree.getroot()

    objects = {}
    index = 1
    for obj in root.iter(tag='object'):
        for elem in obj.iter():
            if elem.tag == 'name':
                if elem.text == 'head' or elem.text == 'hand' or elem.text == 'foot':
                    continue
                objects[index] = {'type': elem.text}

            if elem.tag == 'xmin' or elem.tag == 'ymin' or elem.tag == 'xmax' or elem.tag == 'ymax':
                if len(elem.text.split('.')) > 1:
                    objects[index][elem.tag] = int(elem.text.split('.')[0])
                else:
                    objects[index][elem.tag] = int(elem.text)

            if index in objects.keys():
                if len(objects[index].keys()) == 5:
                    break
        index += 1

    return objects
