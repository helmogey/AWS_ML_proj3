
import logging
from torch import nn
import torch
import torchvision.models as models
import os
from torchvision import transforms
from PIL import Image
import json
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("s3fs")


import s3fs
fs = s3fs.S3FileSystem()


logger = logging.getLogger(__name__)

def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading the model.')
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    logger.info('Done loading model')
    return model





def input_fn(request_body, content_type='application/json'):
    logger.info('Deserializing the input data.')
    if content_type == 'application/json':
        input_data = json.loads(request_body)

        for i,url in enumerate(input_data):
#             url = input_data["url"]
            logger.info(url)
            with fs.open(url) as f:
                image_data = Image.open(f)
                image_transform = transforms.Compose([
                    transforms.CenterCrop(size=224),
                    transforms.ToTensor()
                ])
                img = image_transform(image_data).view(1, 3, 224, 224)
                if i == 0:
                    imgs = img
                else:
                    imgs = torch.cat((imgs, img), 0)

        return imgs

    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')





def predict_fn(input_data, model):
    logger.info('Generating prediction based on input parameters.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Using ',device)
    input_data = input_data.to(device)
    model.to(device)
    with torch.no_grad():
        model.eval()
        out = model(input_data)
        ps = torch.exp(out)
    return ps


def output_fn(prediction_output, accept='application/json'):
    logger.info('Serializing the generated output.')
    classes = {0: 'error', 1: 'no_error'}
    _, preds = torch.max(prediction_output, 1)
    results = []
    for pred in preds:
        results.append(classes[int(pred)])

    if accept == 'application/json':
        return json.dumps(results), accept
    raise Exception(f'Requested unsupported ContentType in Accept:{accept}')








