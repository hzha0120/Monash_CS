from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.db.models import Q
from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.http import HttpResponse

from .camera import MaskDetect
from .models import EWasteSite, Clothing
from django.core import serializers
import base64
import os
import cv2
from .model.garbageDetection import modelThread
import tensorflow
import numpy as np

tensorflow.keras.backend.clear_session()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model = modelThread()
model.__int__("./base/model/")
model.model.summary
img = cv2.imread("./static/images/coffee.png")
model.predict(img)

tensorflow.keras.backend.clear_session()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model = modelThread()
model.__int__("./base/model/")
model.model.summary
img = cv2.imread("./static/images/coffee.png")
model.predict(img)

def home(request):
    return render(request, 'base/home.html')


def index(request):
    return render(request, 'base/index.html')


def about_us(request):
    return render(request, 'base/team.html')


def e_waste_classification(request):
    site = str(request.GET.get('ESite')) if request.GET.get('ESite') != None else ''
    print(site)
    ewastes = EWasteSite.objects.all()
    context = {'ewastes': ewastes}
    return render(request, 'base/e-waste-classification.html', context)


def e_waste(request):
    site = str(request.GET.get('ESite')) if request.GET.get('ESite') != None else ''
    print(site)
    ewastes = EWasteSite.objects.all()
    context = {'ewastes': ewastes}

    return render(request, 'base/e-waste-classification.html', context)


def clothing(request):
    site = str(request.GET.get('CSite')) if request.GET.get('CSite') != None else ''
    print(site)
    clothings = Clothing.objects.all()
    context = {'clothings': clothings}

    return render(request, 'base/clothing.html', context)


def garbage(request):
    return render(request, 'base/garbage.html', )


def garbage_video(request):
    return render(request, 'base/garbage-webcam.html', )


def getbase64byndarray(pic_img):
    retval, buffer = cv2.imencode('.jpg', pic_img)
    pic_str = base64.b64encode(buffer)
    return pic_str.decode()


def garbageClassification(request):
    image = request.FILES['img']
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    res, idx = model.predict(img)
    res = base64.b64encode(res)
    print("predict classes index", idx)
    content = {"image": res}
    return render(request, 'base/garbage.html', content)


def gen(camera):
    while True:
        frame = camera.get_frame(model)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video(request):
    return StreamingHttpResponse(gen(MaskDetect(model)),content_type='multipart/x-mixed-replace; boundary=frame')
