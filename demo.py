from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np
from PIL import Image
import cv2

from functools import partial
import sys
import json

import pdb

# load a pretrained model, such a model already has a global pooling at the end
# model_id: 1 - SqueezeNet, 2 - ResNet, 3 - DenseNet
def load_model(model_id):
    if model_id == 1:
        model = models.squeezenet1_1(pretrained = True)
        final_conv_layer = 'classifier.1'
    elif model_id == 2:
        model = models.resnet101(pretrained = True)
        final_conv_layer = 'layer4'
    elif model_id == 3:
        model = models.densenet161(pretrained = True)
        final_conv_layer = 'features'
    else:
        sys.exit('No such model!')

    return model, final_conv_layer

# a hook to a given layer
def hook(module, input, output, feature_blob):
    feature_blob.append(output.data.numpy())

# load and preprocess an image
def load_image(filename = './test.jpb'):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    image = Image.open(filename)
    image = preprocess(image)

    return Variable(image.unsqueeze(0))

# read in labels, original file url: https://s3.amazonaws.com/outcome-blog/imagenet/labels.json
def get_labels(filename = '.labels.json'):
    with open(filename) as f:
        content = json.load(f)

    labels = {int(k) : v for (k, v) in content.items()}

    return labels

# compute class activation map
def compute_cam(activation, softmax_weight, class_ids):
    b, c, h, w = activation.shape
    cams = []
    for idx in class_ids:
        activation = activation.reshape(c, h * w)
        cam = softmax_weight[idx].dot(activation)
        cam = cam.reshape(h, w)
        # normalize to [0, 1]
        cam =  (cam - cam.min()) / (cam.max() - cam.min())
        # conver to [0, 255]
        cam = np.uint8(255 * cam)
        pdb.set_trace()
        # reshape to (224, 224)
        cams.append(cv2.resize(cam, (224, 224)))

    return cams

if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.exit('Wrong number of arguments!')

    try:
        model_id = int(sys.argv[1])
    except:
        sys.exit('Wrong second argument')

    # load a pretrained model
    model, final_conv_layer = load_model(model_id)    # model_id: 1 - SqueezeNet, 2 - ResNet, 3 - DenseNet
    model.eval()

    # add a hook to a given layer
    feature_blob = []
    model._modules.get(final_conv_layer).register_forward_hook(partial(hook, feature_blob = feature_blob))

    # get the softmax (last fc layer) weight
    params = list(model.parameters())
    softmax_weight = np.squeeze(params[-2].data.numpy())

    input = load_image('./test.jpg')

    output = model(input)   # scores

    labels = get_labels('./labels.json')

    probs = F.softmax(output).data.squeeze()
    probs, idx = probs.sort(0, descending = True)

    print(labels)

    # output the top-5 prediction
    for i in range(5):
        print('{:.3f} -> {}'.format(probs[i], labels[idx[i]]))

    # generate class activation map for the top-1 prediction
    cams = compute_cam(feature_blob[0], softmax_weight, [idx[0]])

    # render cam and original image
    print('output cam.jpg for the top1 prediction: %s' % labels[idx[0]])

    img = cv2.imread('./test.jpg')
    h, w, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(cams[0], (w, h)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('cam.jpg', result)