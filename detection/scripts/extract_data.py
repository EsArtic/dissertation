import sys
import os
import os.path
import cv2
import argparse
import xml_parser

label_map = {
    'sheep':        0, 'horse':   1, 'bicycle':       2, 'bottle':     3,
    'cow':          4, 'car':     5, 'dog':           6, 'bus':        7,
    'cat':          9, 'person':  9, 'train':        10, 'boat':      11,
    'aeroplane':   12, 'sofa':   13, 'pottedplant':  14, 'tvmonitor': 15,
    'chair':       16, 'bird':   17, 'diningtable':  18, 'motorbike': 19,
    'dinnertable': 18
}

def extract_multi_label(objects):
    rect = [0] * 20
    for key, obj in objects.items():
        rect[label_map[obj['type']]] = 1
    return rect

def write_multi_label(text_file, img_name, label):
    text_file.write(img_name)
    for value in label:
        text_file.write(' ' + str(value))
    text_file.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--image_dir', type=str, required=True,
                        help='The path of the folder that stores source images')
    parser.add_argument('-a', '--annotation_dir', type=str, required=True,
                        help='The path of the folder that stores annotations')
    parser.add_argument('-t', '--testset', action='store_true',
                        help='Operate on test set. If unspecified then train set.')
    args = parser.parse_args()

    IMAGE_DIR = args.image_dir
    LABEL_DIR = args.annotation_dir
    CLASSIFICATION_DIR = './classification'
    LOCALIZATION_DIR = './localization'
    CLASSIFICATION_TARGET = ''
    LOCALIZATION_TARGET = ''
    CLASSIFICATION_TEXT = ''
    LOCALIZATION_TEXT = ''

    if args.testset:
        CLASSIFICATION_TARGET = os.path.join(CLASSIFICATION_DIR, 'test')
        LOCALIZATION_TARGET = os.path.join(LOCALIZATION_DIR, 'test')
        CLASSIFICATION_TEXT = os.path.join(CLASSIFICATION_DIR, 'test.txt')
        LOCALIZATION_TEXT = os.path.join(LOCALIZATION_DIR, 'test.txt')
    else:
        CLASSIFICATION_TARGET = os.path.join(CLASSIFICATION_DIR, 'train')
        LOCALIZATION_TARGET = os.path.join(LOCALIZATION_DIR, 'train')
        CLASSIFICATION_TEXT = os.path.join(CLASSIFICATION_DIR, 'train.txt')
        LOCALIZATION_TEXT = os.path.join(LOCALIZATION_DIR, 'train.txt')

    classification_text = open(CLASSIFICATION_TEXT, 'w')
    localization_text = open(LOCALIZATION_TEXT, 'w')
    for parent, dirnames, filenames in os.walk(LABEL_DIR):
        total = len(filenames)
        interval = total / 50
        count = 1
        bar = ''
        for xml_file in filenames:

            objects = xml_parser.parse(os.path.join(LABEL_DIR, xml_file))
            label_vector = extract_multi_label(objects)

            img_file = xml_file.split('.')[0] + '.jpg'
            write_multi_label(classification_text, img_file, label_vector)

            img = cv2.imread(os.path.join(IMAGE_DIR, img_file))

            for key, obj in objects.items():
                curr_img = img[obj['ymin'] : obj['ymax'], obj['xmin'] : obj['xmax']]
                output_img_name = xml_file.split('.')[0] + '_' + str(key) + '.jpg'

                curr_img = cv2.resize(curr_img, (256, 256))
                cv2.imwrite(os.path.join(LOCALIZATION_TARGET, output_img_name), curr_img)

                write_multi_label(localization_text, output_img_name, [label_map[obj['type']]])

            if not args.testset:
                curr_img = cv2.resize(img, (256, 256))
                cv2.imwrite(os.path.join(CLASSIFICATION_TARGET, img_file), curr_img)

            curr = int(float(count) / total * 100)
            remain = 50 - len(bar) - 1

            if count < total:
                sys.stdout.write(str(curr) + '% [' + bar + '>' + remain * ' ' + ']\r')
                sys.stdout.flush()
            else:
                sys.stdout.write(str(curr) + '% [' + bar + ']\r')
                sys.stdout.flush()

            count += 1
            if count % interval == 0:
                bar += '='

    print
    classification_text.close()
    localization_text.close()

if __name__ == '__main__':
    main()
