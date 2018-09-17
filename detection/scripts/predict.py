import argparse
import time
import os
import sys
import cv2
import skimage
import random

sys.path.append('../caffe/python/')
import caffe
import selectivesearch

mapper = {0: 'sheep', 1: 'horse', 2: 'bicycle', 3: 'bottle',
          4: 'cow', 5: 'car', 6: 'dog', 7: 'bus',
          8: 'cat', 9: 'person', 10: 'train', 11: 'boat',
          12: 'aeroplane', 13: 'sofa', 14: 'pottedplant', 15: 'tvmonitor',
          16: 'chair', 17: 'bird', 18: 'dinnertable', 19: 'motorbike',
          20: 'background'}

def removeFileInFirstDir(targetDir):
    for f in os.listdir(targetDir):
        targetFile = os.path.join(targetDir, f)
        if os.path.isfile(targetFile):
            os.remove(targetFile)

def get_color():
    return [random.randrange(0, 255, 50), random.randrange(0, 255, 50), random.randrange(0, 255, 50)]

def get_str(prob):
    ans = ''
    for i in xrange(20):
        if prob[i] == 1:
            ans += ' ' + mapper[i]
    return ans

def is_cover(r1, r2, h, w):
    # r1_size = float((r1[2] - r1[0]) * (r1[3] - r1[1]))
    # r2_size = float((r2[2] - r2[0]) * (r2[3] - r2[1]))

    cover_x = 0
    x_list = [0] * w
    for i in xrange(r1[0], r1[2]):
        x_list[i] = 1
    for i in xrange(r2[0], r2[2]):
        if x_list[i] > 0:
            cover_x += 1

    cover_y = 0
    y_list = [0] * h
    for i in xrange(r1[1], r1[3]):
        y_list[i] = 1
    for i in xrange(r2[1], r2[3]):
        if y_list[i] > 0:
            cover_y += 1

    # cover_area = float(cover_x * cover_y)
    # if ((cover_area / r1[5]) > 0.16) or ((cover_area / r2[5]) > 0.16):
    #     return True
    if cover_x > 1 and cover_y > 1:
        return True
    return False

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, required=True,
                        help='Path of the source image')
    parser.add_argument('-c', '--classification_modelname', type=str, required=True,
                        help='Name of the model file for classification')
    parser.add_argument('-l', '--localization_modelname', type=str, required=True,
                        help='Name of the model file for localization')
    parser.add_argument('-d', '--destination', type=str, required=True,
                        help='Target path to store the predict image')
    args = parser.parse_args()

    DIR = '../models'
    CLASSIFICATION_NET = '../prototxt/classification_GoogleNet/deploy.prototxt'
    LOCALIZATION_NET = '../prototxt/localization_GoogleNet/deploy.prototxt'
    CLASSIFICATION_MODEL = os.path.join(DIR, args.classification_modelname)
    LOCALIZATION_MODEL = os.path.join(DIR, args.localization_modelname)

    SOURCE = args.source
    TARGET = args.destination

    caffe.set_mode_gpu()

    classification_net = caffe.Net(CLASSIFICATION_NET, CLASSIFICATION_MODEL, caffe.TEST)
    localization_net = caffe.Net(LOCALIZATION_NET, LOCALIZATION_MODEL, caffe.TEST)

    begin = time.time()
    caffe_img = caffe.io.load_image(SOURCE)
    transformer = caffe.io.Transformer({'data': classification_net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0 ,1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    classification_net.blobs['data'].reshape(1, 3, 224, 224)
    classification_net.blobs['data'].data[...] = transformer.preprocess('data', caffe_img)
    
    classification_net.blobs['data'].data[0][0] -= 104
    classification_net.blobs['data'].data[0][1] -= 117
    classification_net.blobs['data'].data[0][2] -= 123

    classification_net.forward()
    result = classification_net.blobs['prob'].data[0]

    prob_predict = [0] * 20
    max_point = result[0]
    max_index = 0
    count = 0
    for i in xrange(len(result)):
        if result[i] > max_point:
            max_point = result[i]
            max_index = i
        if result[i] > 0.5:
            count += 1
            prob_predict[i] = 1
    if count == 0:
        prob_predict[max_index] = 1

    print '[Message] classification stage output:', get_str(prob_predict)

    color_map = {}
    for i in xrange(len(prob_predict)):
        if prob_predict[i] == 1:
            color_map[i] = get_color()

    sk_img = skimage.io.imread(SOURCE)

    search_area = [(500, 0.9, 10), (1000, 0.6, 10)]
    regions = []
    for v1, v2, v3 in search_area:
        img_lbl, r = selectivesearch.selective_search(sk_img, scale=v1, sigma=v2, min_size=v3)
        regions += r

    candidates = set()
    for r in regions:
        if r['rect'] in candidates:
            continue

        if r['size'] < 1600:
            continue

        x, y, w, h = r['rect']
        candidates.add(r['rect'])

    localization_net.blobs['data'].reshape(1, 3, 224, 224)
    final = {}

    for x, y, w, h in candidates:
        crop = caffe_img[y : y + h, x : x + w]
        localization_net.blobs['data'].data[...] = transformer.preprocess('data', crop)

        localization_net.blobs['data'].data[0][0] -= 104
        localization_net.blobs['data'].data[0][1] -= 117
        localization_net.blobs['data'].data[0][2] -= 123

        out = localization_net.forward()
        predicts = out['prob']

        type_id = predicts.argmax()
        if prob_predict[type_id] != 1:
            continue

        if not final.has_key(type_id):
            final[type_id] = []

        curr_region = (x, y, x + w, y + h, predicts[0][type_id], w * h)
        final[type_id].append(curr_region)

    detection_result = []
    last_type = 0
    img_size = float(sk_img.shape[0] * sk_img.shape[1])
    for label, regions in final.items():

        # Prefer the smaller boundingboxes
        cand_regions = sorted(regions, key=lambda x: x[4] * 0.995 + (1.0 - x[5] / img_size) * 0.005)

        # Prefer the larger boundingboxes
        # cand_regions = sorted(regions, key=lambda x: x[4] * 0.9949 + float(x[5]) / img_size * 0.0051)

        # No preference
        # cand_regions = sorted(regions, key=lambda x: x[4])

        index = len(cand_regions) - 1
        while cand_regions[index][4] > 0.8 and index > -1:
            permit = True
            for i in xrange(last_type, len(detection_result)):
                if is_cover(cand_regions[index], detection_result[i][1],
                            sk_img.shape[0], sk_img.shape[1]):
                    permit = False
                    break

            if permit:
                detection_result.append((label, cand_regions[index]))
            index -= 1
        last_type = len(detection_result)

    cv2_img = cv2.imread(SOURCE)
    output_text = open(os.path.join(TARGET, SOURCE.split('/')[-1].split('.')[0] + '_predict.txt'), 'w')

    for label, region in detection_result:
        print '[Message]', mapper[label], region
        cv2.rectangle(cv2_img, (region[0], region[1]), (region[2], region[3]), color_map[label], 3)
        output_text.write('type: ' + mapper[label] + '\n')
        output_text.write('min_x: ' + str(region[0]) + '\n')
        output_text.write('min_y: ' + str(region[1]) + '\n')
        output_text.write('max_x: ' + str(region[2]) + '\n')
        output_text.write('max_y: ' + str(region[3]) + '\n')
        output_text.write('prob: ' + str(region[4]) + '\n\n')
    end = time.time()
    print '[Message] Time Cost:', end - begin, 'seconds'

    # cv2.imshow('result_img', cv2_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(TARGET, SOURCE.split('/')[-1].split('.')[0] + '_predict.jpg'), cv2_img)

if __name__ == '__main__':
    main()

