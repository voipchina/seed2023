#!/usr/bin/env python
# coding: utf-8
import os
import io
from PIL import Image, ImageFilter
from fastapi import FastAPI, UploadFile, File,Response,HTTPException
import uvicorn
import numpy as np
import random
import torch
import torchvision.transforms.functional as F
from torchvision import transforms

from unet import UNet

app = FastAPI()
net = UNet(n_channels=1, n_classes=1)
current_path = os.path.dirname(os.path.realpath(__file__))
checkpoint = torch.load(f'{current_path}/segment.pth', map_location='cpu')
net.load_state_dict(checkpoint['state_dict'])
net.eval()

def gen_black_mask(width, height):
    return np.zeros((height, width), dtype=np.uint8)

def gen_white_mask(width, height):
    return np.ones((height, width), dtype=np.uint8) * 255

image_size = (224, 224)

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')

# now use it as the replacement of transforms.Pad class
basic_transform=transforms.Compose([
    SquarePad(),
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

def img_to_mask_np_sanity(img_file):
    img = Image.open(img_file)
    width, height = img.size
    img = np.array(img)
    mask_np = gen_white_mask(width, height)
    mask_np = mask_np.flatten()
    mask_np[random.sample(range(0, height*width), 50)] = 0
    mask_np = mask_np.reshape(height, width)
    return mask_np

def img_to_mask_np_sanity(img_file):
    img = Image.open(img_file)
    width, height = img.size
    img = np.array(img)
    mask_np = gen_white_mask(width, height)
    mask_np = mask_np.flatten()
    mask_np[random.sample(range(0, height*width), 50)] = 0
    mask_np = mask_np.reshape(height, width)
    return mask_np

def post_process(image):
    image = image.filter(ImageFilter.ModeFilter(size=13))
    return image

@torch.no_grad()
def img_to_mask_np(img_file):
    THRESHOLD_VALUE = 127
    image = Image.open(img_file).convert("L")
    width, height = image.size
    max_wh = max(width, height)
    left = int((max_wh - width) / 2)
    upper = int((max_wh - height) / 2)
    input = basic_transform(image)
    input = input.unsqueeze(0)
    pred = net(input)
    pred = np.array(pred.data.cpu()[0])[0]
    bin_pred = (pred > 0.5).astype(np.float32)
    im = Image.fromarray(np.uint8(bin_pred*255), 'L')
    im = im.resize((max_wh, max_wh))
    im = im.crop((left, upper, left+width, upper+height))
    im = post_process(im)
    mask_np = np.asarray(im)
    mask_np = (mask_np > THRESHOLD_VALUE)
    return np.uint8(mask_np)

def test():
     from glob import glob
     from tqdm import tqdm
     test_file_list = sorted(glob(f'test/input/*.png'))
     pbar = tqdm(test_file_list)
     for img_file in pbar:
        background = Image.open(img_file)
        mask_np = img_to_mask_np(img_file) * 255
        mask = Image.fromarray(mask_np, 'L').convert('RGB')
        mask_edge = mask.filter(ImageFilter.FIND_EDGES)
        blended = Image.blend(background, mask_edge, alpha=0.5)
        blended.save(f'test/output/{os.path.basename(img_file)}')


def img_to_mask_png(img_file):
    outpu_np = img_to_mask_np(img_file) * 255
    img_byte_arr = io.BytesIO()
    Image.fromarray(outpu_np).save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        image = await file.read()
        img_file = io.BytesIO(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail="read file error") from e

    response_np = img_to_mask_np(img_file)
    response_bytes = response_np.tobytes()
    response = Response(content=response_bytes, media_type="application/octet-stream")
    return response

@app.post("/mask")
async def segment_image(file: UploadFile = File(...)):
    try:
        image = await file.read()
        img_file = io.BytesIO(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail="read file error") from e

    response_bytes = img_to_mask_png(img_file)
    response = Response(content=response_bytes, media_type="image/png")
    return response

if __name__ == "__main__":
     uvicorn.run(app, host="0.0.0.0", port=8000)


