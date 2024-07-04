import os
import hashlib
import time

import cv2
import numpy as np
import requests
from PIL import Image
from werkzeug.utils import secure_filename

from .constants import UPLOAD_FOLDER, VIDEO_FOLDER, DETECTION_FOLDER, CSV_FOLDER, SEGMENTATION_FOLDER, IMAGE_ALLOWED_EXTENSIONS, VIDEO_ALLOWED_EXTENSIONS
from .modules import get_prediction


def allowed_file_image(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in IMAGE_ALLOWED_EXTENSIONS


def allowed_file_video(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in VIDEO_ALLOWED_EXTENSIONS


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def file_type(path):
    filename = path.split('/')[-1]
    if allowed_file_image(filename):
        filetype = 'image'
    elif allowed_file_video(filename):
        filetype = 'video'
    else:
        filetype = 'invalid'
    return filetype


def download(url):
    """
    Handle input url from client 
    """

    make_dir(UPLOAD_FOLDER)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2)',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
               'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
               'Accept-Encoding': 'none',
               'Accept-Language': 'en-US,en;q=0.8',
               'Connection': 'keep-alive'}
    r = requests.get(url, stream=True, headers=headers)
    print('Image Url')

    # Get cache name by hashing image
    data = r.content
    ori_filename = url.split('/')[-1]
    _, ext = os.path.splitext(ori_filename)
    filename = hashlib.sha256(data).hexdigest() + f'{ext}'

    path = os.path.join(UPLOAD_FOLDER, filename)

    with open(path, "wb") as file:
        file.write(r.content)

    return filename, path


def save_upload(file):
    """
    Save uploaded image and video if its format is allowed
    """
    filename = secure_filename(file.filename)
    if allowed_file_image(filename):
        make_dir(UPLOAD_FOLDER)
        path = os.path.join(UPLOAD_FOLDER, filename)

    elif allowed_file_video(filename):
        make_dir(VIDEO_FOLDER)
        path = os.path.join(VIDEO_FOLDER, filename)

    file.save(path)

    return path

def process_webcam_capture(request):
    f = request.files['blob-file']
    ori_file_name = secure_filename(f.filename)
    filetype = file_type(ori_file_name)

    filename = time.strftime("%Y%m%d-%H%M%S") + '.png'
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # save file to /static/uploads
    img = Image.open(f.stream)
    img.save(filepath)

    return filename, filepath, filetype

def process_url_input(request):
    """
    This function handles the process of getting an image or video from an input URL.

    Parameters:
    request (flask.Request): The request object containing the form data.
        The form data should contain a 'url_link' field which contains the URL of the image or video.

    Returns:
    tuple: A tuple containing three elements:
        1. filename (str): The name of the downloaded file.
        2. filepath (str): The path of the downloaded file.
        3. filetype (str): The type of the file ('image' or 'video').
    """
    # Extract the URL from the form data
    url = request.form['url_link']

    # Download the image or video from the URL
    filename, filepath = download(url)

    # Determine the type of the file
    filetype = file_type(filename)

    # Return the filename, filepath, and filetype
    return filename, filepath, filetype

def process_image_file(filename, filepath, model_types, tta, ensemble, min_conf, min_iou, enhanced, segmentation):
    """
    This function processes an image file by applying object detection or semantic segmentation.

    Parameters:
    filename (str): The name of the input image file.
    filepath (str): The path of the input image file.
    model_types (str or list): The type(s) of model(s) to use for prediction.
    tta (bool): Whether to apply test time augmentation (TTA).
    ensemble (bool): Whether to use ensemble models.
    min_conf (float): The minimum confidence threshold for detections.
    min_iou (float): The minimum intersection over union (IoU) threshold for non-maximum suppression.
    enhanced (bool): Whether to enhance the labels of detected objects.
    segmentation (bool): Whether to perform semantic segmentation instead of object detection.

    Returns:
    tuple: A tuple containing three elements:
        1. out_name (str): The name of the output file.
        2. output_path (str): The path of the output file.
        3. output_type (str): The type of the output ('image' or 'segmentation').
    """
    # Get filename of detected image
    out_name = "Image Result"
    output_path = os.path.join(
        DETECTION_FOLDER, filename) if not segmentation else os.path.join(
        SEGMENTATION_FOLDER, filename)

    output_path, output_type = get_prediction(
        filepath,
        output_path,
        model_name=model_types,
        tta=tta,
        ensemble=ensemble,
        min_conf=min_conf,
        min_iou=min_iou,
        enhance_labels=enhanced,
        segmentation=segmentation)
    
    return out_name, output_path, output_type

def process_output_file(output_path):
    filename = os.path.basename(output_path)
    csv_name, _ = os.path.splitext(filename)

    csv_name1 = os.path.join(
        CSV_FOLDER, csv_name + '_info.csv')
    csv_name2 = os.path.join(
        CSV_FOLDER, csv_name + '_info2.csv')
    
    return filename, csv_name1, csv_name2

def process_upload_file(request):
    # Get uploaded file
    f = request.files['file']
    ori_file_name = secure_filename(f.filename)
    _, ext = os.path.splitext(ori_file_name)

    filetype = file_type(ori_file_name)

    if filetype == 'image':
        # Get cache name by hashing image
        data = f.read()
        filename = hashlib.sha256(data).hexdigest() + f'{ext}'

        # Save file to /static/uploads
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        np_img = np.fromstring(data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        cv2.imwrite(filepath, img)
    
    return filename, filepath, filetype
