import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps

x = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
n_classes = len(classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 500, train_size = 3500, random_state = 9)
x_train_scalled = x_train / 255
x_test_scalled = x_test / 255

classifier = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train_scalled, y_train)

y_predict = classifier.predict(x_test_scalled)
accuracy = accuracy_score(y_test, y_predict)
print('Accuracy:', accuracy * 100 , '%')

def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((22, 30), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized - min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled) / max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,660)
    test_predict = classifier.predict(test_sample)
    return test_predict[0]