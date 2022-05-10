import pytesseract
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import clear_border


class ANPR:
    def __init__(self, minAR=1, maxAR=6, debug=False):
        self.minAR = minAR
        self.maxAR = maxAR
        self.debug = debug

    def debug_imshow(self, image):
        if self.debug:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()

    @staticmethod
    def locate_license_plate_candidate(gray, keep=5):
        gray = cv2.blur(gray, (4, 5))
        canny = cv2.Canny(gray, 150, 250)
        canny = cv2.dilate(canny, None, iterations=2)

        cnts, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(canny, cnts, 0, (0, 255, 0), 2)
        cv2.imshow('Canny', canny)
        cv2.waitKey(0)

        return cnts

    @staticmethod
    def locate_license_plate(gray, candidates, clearBorder=False):
        lp = None
        lpCnt = None

        for c in candidates:
            area = cv2.contourArea(c)
            (x, y, w, h) = cv2.boundingRect(c)
            epsilon = 0.09 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)

            if len(approx) == 4 and (5000 < area < 5500):
                print('Approx -> {}'.format(len(approx)))
                aspect_ratio = float(w) / h
                print('Plate aspect ratio: ', aspect_ratio)
                if aspect_ratio > 1.2:
                    # cv2.drawContours(gray, [c], -1, (0, 255, 0), 2)
                    lp = gray[y:y + h, x:x + w]
                    lpCnt = c

        return lp, lpCnt

    @staticmethod
    def build_tesseract_options(psm=7):
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        # set the PSM mode
        options += " --psm {}".format(psm)

        return options

    def find_and_ocr(self, image, psm=7, clearBorder=False):
        lpText = None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.locate_license_plate_candidate(gray)
        (license_plate, lpCnt) = self.locate_license_plate(image, candidates, clearBorder=clearBorder)
        # only OCR the license plate if the license plate ROI is not
        # empty
        if license_plate is not None:
            options = self.build_tesseract_options(psm=psm)
            lpText = pytesseract.image_to_string(license_plate, config=options)

        # return a 2-tuple of the OCR'd license plate text along with
        # the contour associated with the license plate region
        print('Plate -> ', lpText)
        print('Plate Contour -> ', lpCnt)
        return lpText, lpCnt
