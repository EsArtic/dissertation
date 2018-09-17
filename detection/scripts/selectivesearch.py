import skimage
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np

##
# Generate the initial regions by the algorithm of Felzenswalb and Huttenlocher
##
def segmentation(source_img, scale, sigma, min_size):

    img_mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(source_img), scale=scale, sigma=sigma, min_size=min_size)

    source_img = np.append(
        source_img, np.zeros(source_img.shape[:2])[:, :, np.newaxis], axis=2)

    source_img[:, :, 3] = img_mask

    return source_img

##
# Caculate the similarity of color
##
def similarity_color(r1, r2):

    return sum([min(a, b) for a, b in zip(r1['hist_c'], r2['hist_c'])])

##
# Calculate the similarity of texture
##
def similarity_texture(r1, r2):

    return sum([min(a, b) for a, b in zip(r1['hist_t'], r2['hist_t'])])

##
# Calculate the similarity of size
##
def similarity_size(r1, r2, size):

    return 1.0 - (r1['size'] + r2['size']) / size

##
# Calculate the similarity for the relations of whole and part
##
def similarity_fill(r1, r2, size):

    bnd_box_size = (
        (max(r1['max_x'], r2['max_x']) - min(r1['min_x'], r2['min_x'])) 
        * (max(r1['max_y'], r2['max_y']) - min(r1['min_y'], r2['min_y']))
    )
    return 1.0 - (bnd_box_size - r1['size'] - r2['size']) / size

##
# Calculate the whole similarity
##
def similarity(r1, r2, size):

    return (similarity_color(r1, r2) + similarity_texture(r1, r2) 
            + similarity_size(r1, r2, size) + similarity_fill(r1, r2, size))

##
# Calculate the color histogram for each region
##
def color_histogram(img):

    BINS = 25
    histogram = np.array([])

    for channel in (0, 1, 2):
        curr = img[:, channel]
        histogram = np.concatenate([histogram] + [np.histogram(curr, BINS, (0.0, 255.0))[0]])

    # L1 normalize
    histogram = histogram / len(img)
    return histogram

##
# Calculate texture gradient for the entire image
##
def texture_gradient(img):

    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))

    for channel in (0, 1, 2):
        ret[:, :, channel] = skimage.feature.local_binary_pattern(img[:, :, channel], 8, 1.0)

    return ret

##
# Calculate the texture histogram for each region
##
def texture_histogram(img):

    BINS = 10
    histogram = np.array([])
    for channel in (0, 1, 2):
        curr = img[:, channel]
        histogram = np.concatenate([histogram] + [np.histogram(curr, BINS, (0.0, 1.0))[0]])

    histogram = histogram / len(img)
    return histogram

def extract_regions(img):

    R = {}
    hsv = skimage.color.rgb2hsv(img[:, :, :3])

    for y, col in enumerate(img):
        for x, (r, g, b, label) in enumerate(col):

            # initialize a new region
            if label not in R:
                R[label] = {'min_x': 2000, 'min_y': 2000, 'max_x': 0, 'max_y': 0, 'labels': [label]}

            # find the bounding box position
            if R[label]['min_x'] > x:
                R[label]['min_x'] = x
            if R[label]['min_y'] > y:
                R[label]['min_y'] = y
            if R[label]['max_x'] < x:
                R[label]['max_x'] = x
            if R[label]['max_y'] < y:
                R[label]['max_y'] = y

    text_grad = texture_gradient(img)

    for k in R.keys():

        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        R[k]['size'] = len(masked_pixels / 4)
        R[k]['hist_c'] = color_histogram(masked_pixels)
        R[k]['hist_t'] = texture_histogram(text_grad[:, :][img[:, :, 3] == k])

    return R

def adjacent(r1, r2):

    if (r1['min_x'] <= r2['min_x'] < r1['max_x'] and r1['min_y'] <= r2['min_y'] < r1['max_y']) or (
        r1['min_x'] < r2['max_x'] <= r1['max_x'] and r1['min_y'] < r2['max_y'] <= r1['max_y']) or (
        r1['min_x'] <= r2['min_x'] < r1['max_x'] and r1['min_y'] < r2['max_y'] <= r1['max_y']) or (
        r1['min_x'] < r2['max_x'] <= r1['max_x'] and r1['min_y'] <= r2['min_y'] < r1['max_y']):
        return True
    return False

def extract_neighbours(regions):

    R = regions.items()
    neighbours = []
    for curr, r1 in enumerate(R[:-1]):
        for r2 in R[curr + 1:]:
            if adjacent(r1[1], r2[1]):
                neighbours.append((r1, r2))

    return neighbours

def merge(r1, r2):

    new_size = r1['size'] + r2['size']
    new_region = {
        'min_x': min(r1['min_x'], r2['min_x']),
        'min_y': min(r1['min_y'], r2['min_y']),
        'max_x': max(r1['max_x'], r2['max_x']),
        'max_y': max(r1['max_y'], r2['max_y']),
        'size': new_size,
        'hist_c': (r1['hist_c'] * r1['size'] + r2['hist_c'] * r2['size']) / new_size,
        'hist_t': (r1['hist_t'] * r1['size'] + r2['hist_t'] * r2['size']) / new_size,
        'labels': r1['labels'] + r2['labels']
    }
    return new_region

def selective_search(source_img, scale=1.0, sigma=0.8, min_size=50):

    img = segmentation(source_img, scale, sigma, min_size)

    if img is None:
        return None, {}

    img_size = img.shape[0] * img.shape[1]
    R = extract_regions(img)
    neighbours = extract_neighbours(R)

    S = {}
    for (k1, r1), (k2, r2) in neighbours:
        S[(k1, k2)] = similarity(r1, r2, img_size)

    #     Merge the regions until there are no neighbour relations,
    # which means the whole image is built again.
    while S != {}:

        # Find two regions with highest similarity
        k1, k2 = sorted(S.items(), key=lambda x: x[1])[-1][0]

        # Merge the selected regions
        new_key = max(R.keys()) + 1.0
        R[new_key] = merge(R[k1], R[k2])

        # Update the neighbour relations and similarities
        key_to_delete = []
        for k, region in S.items():
            if (k1 in k) or (k2 in k):
                key_to_delete.append(k)

        for k in key_to_delete:
            del S[k]

        for k in filter(lambda x: x != (k1, k2), key_to_delete):
            n = k[1] if k[0] in (k1, k2) else k[0]
            S[(new_key, n)] = similarity(R[new_key], R[n], img_size)

    regions = []
    for k, r in R.items():
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })
    return img, regions

