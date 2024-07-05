import cv2
import imutils
import numpy as np
import pytesseract
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask

from flask import render_template


app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread('car.jpg')
det = cv2.text.TextDetectorCNN_create("textbox.prototxt", "TextBoxes_icdar13.caffemodel")

print(img.shape)
rects, probs = det.detect(img)

THR = 0.3
minimumx = 99999
minimumy = 99999
maximumx = 0
maximumy = 0
for i, r in enumerate(rects):
    if probs[i] > THR:
        #cv2.rectangle(img, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0, 255, 0), 3)
        
        if (r[0] < minimumx):
            minimumx = r[0]
            
        if (r[1] < minimumy):
            minimumy = r[1]

        if (r[0]+r[2] > maximumx):
            maximumx = r[0]+r[2]
            
        if (r[1]+r[3] > maximumy):
            maximumy = r[1]+r[3]
        
#print (minimumx)

#print (minimumy)

#print (maximumx)

#print (maximumy)

border = 2
x = minimumx - 4*border
y = minimumy - border
w = maximumx - minimumx + 8*border
h = maximumy - minimumy + 2*border



gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)

#Cropped = binary_img[x:x+w, y:y+h]
Cropped = binary_img[y:y+h,x:x+w]

#print (w)
#print (h)
#text = pytesseract.image_to_string(Cropped, config='--psm 11 --oem 3 -c tessedit_char_whitelist=AB0123456789')

text = pytesseract.image_to_string(Cropped, config='--psm 11')


cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

print(text)


plt.figure(figsize=(10,8))
plt.axis('off')
plt.imshow(img[:,:,[2,1,0]])
plt.tight_layout()
plt.show()

cv2.imshow('1', img)
cv2.imshow('2', gray_img)
cv2.imshow('3', binary_img)

cv2.imshow('4', Cropped)
cv2.waitKey()
cv2.destroyAllWindows()


@app.route('/')
@app.route('/<text>')
def index(text=text):

    return render_template('index.html',text = text)






if __name__ == '__main__':
    app.run()

