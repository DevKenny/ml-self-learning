# ANPG/ALPE Plate Recognition

# Steps:
    # 1. Detect and localize a license plate in and output image/frame
    # 2. Extract the characters from license plate
    # 3. Apply some form of Optical Character Recognition (OCR) to recognize

from anpr import anpr
from imutils import paths
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt


def cleanup_text(text):
    return "".join([c if ord(c) < 128 else '' for c in text]).strip()


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='Path to input directory of images')
ap.add_argument('-c', '--clear-border', type=int, default=1, help="Whether or clear border of pixels before OCR'ing")
ap.add_argument('-p', '--psm', type=int, default=7, help='Default PSM mode for OCR license plates')
ap.add_argument('-d', '--debug', type=int, default=1, help='Whether or not to show additional visualizations')
args = vars(ap.parse_args())

anpr = anpr.ANPR(debug=args['debug'] > 0)

imagePaths = sorted(list(paths.list_images(args['input'])))
print('Paths: ', imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    plt.imshow(image)
    plt.show()

    (lpText, lpCnt) = anpr.find_and_ocr(image, psm=args['psm'], clearBorder=args['clear_border'] > 0)
    print('Text', lpText)
    print('CNT:', lpCnt)

    if lpText is not None and lpCnt is not None:
        box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
        box = box.astype('int')
        cv2.drawContours(image, [box], -1,  (0, 255, 0), 2)

        (x, y, h, w) = cv2.boundingRect(lpCnt)
        cv2.putText(image, cleanup_text(lpText), (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        print('[INFO] {}'.format(lpText))
        cv2.imshow('Output ANPR', image)
        cv2.waitKey(0)