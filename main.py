#!/usr/bin/env python
# coding: utf-8

import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File,Response,HTTPException
import uvicorn
import numpy as np
import random
app = FastAPI()

def gen_black_mask(width, height):
    return np.zeros((height, width), dtype=np.uint8)

def gen_white_mask(width, height):
    return np.ones((height, width), dtype=np.uint8) * 255

def img_to_mask_np(img_file):
    img = Image.open(img_file)
    width, height = img.size
    img = np.array(img)
    mask_np = gen_white_mask(width, height)
    mask_np = mask_np.flatten()
    mask_np[random.sample(range(0, height*width), 50)] = 0
    mask_np = mask_np.reshape(height, width)
    return mask_np

def img_to_mask_png(img_file):
    outpu_np = img_to_mask_np(img_file)
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

