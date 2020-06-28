import cv2
import numpy as np

def crop(image_path):
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread(image_path)
    #print(image_path)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (25,25), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Perform morph operations, first open to remove noise, then close to combine
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=3)

    # Find enclosing boundingbox and crop ROI
    coords = cv2.findNonZero(close)
    x,y,w,h = cv2.boundingRect(coords)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    crop = original[y:y+h, x:x+w]

    #cv2.imshow('thresh', thresh)
    #cv2.imshow('close', close)
    #cv2.imshow('image', image)
    #cv2.imshow('crop', crop)
    #cv2.waitKey()
    return crop

def add_white(crop):
    height=crop.shape[0]
    height=height*2
    width=crop.shape[1]
    width=width*2
    blank = np.zeros((height, width, 3))
    blank += 255 #←全ゼロデータに255を足してホワイトにする
    #cv2.imshow('white',blank)
    #cv2.waitKey()
    return blank

def new_image(crop, blank):
    x_offset=blank.shape[1]//4
    y_offset=blank.shape[0]//4
    blank[y_offset:y_offset+crop.shape[0], x_offset:x_offset+crop.shape[1]] = crop
    img_new=blank
    #cv2.imshow('new',blank)
    #cv2.waitKey()
    #print(img_new.shape)
    return img_new


