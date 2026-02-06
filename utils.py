# Utils for AVISEngine Class
import cv2
import io
import time
import base64
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class utilss:
    def __init__(self):

        self.transform = transforms.Compose([transforms.Resize((512,512)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def stringToImage(self, base64_string):
        imgdata = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(imgdata))

    def BGRtoRGB(self, image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    def KMPSearch(self, pat, txt):
        M = len(pat)
        N = len(txt)

        lps = [0] * M
        j = 0  # index for pat[]

        self.computeLPS(pat, M, lps)

        i = 0  # Index for txt[]
        while i < N:
            if pat[j] == txt[i]:
                i += 1
                j += 1

            if j == M:
                return (i - j)

            elif i < N and pat[j] != txt[i]:
                if j != 0:  # Do not match lps[0..lps[j-1]] characters.
                    j = lps[j - 1]
                else:
                    i += 1
        return -1

    def computeLPS(self, pat, M, lps):
        length = 0  # length of the previous longest prefix suffix

        lps[0]  # lps[0] is always 0
        i = 1

        while i < M:  # The loop calculates lps[i] for i = 1 to M-1
            if pat[i] == pat[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                # To search step.
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1

    def getROI(self, image):
        height = image.shape[0]
        width = image.shape[1]
        triangle = np.array([[[0, image.shape[0]], [0, 300], [220, 150], [300, 150], [750, image.shape[0]]]], np.int32)
        black_image = np.zeros_like(image)
        mask = cv2.fillPoly(black_image, triangle, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
  
    def getPath(self, frame):
        blur = cv2.GaussianBlur(frame, (9, 9), 0)
        edges = cv2.Canny(blur, 120, 150)
        ROI = self.getROI(edges)
        lines = cv2.HoughLinesP(ROI, rho=1, theta=np.pi/180, threshold=10, minLineLength=0, maxLineGap=4)
        try:
            left_line_x = []
            left_line_y = []
            right_line_x = []
            right_line_y = []
    
            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) < 0.5:
                        continue
                    if slope <= 0:
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])
                    else:
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])
    
            min_y = int(frame.shape[0] * (3 / 5))
            max_y = int(frame.shape[0])
            poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))
            poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
            right_x_start = int(poly_right(max_y))
            right_x_end = int(poly_right(min_y))
    
            cv2.line(frame, (left_x_start, max_y), (left_x_end, min_y), [255,255,255], 5)
            cv2.line(frame, (right_x_start, max_y), (right_x_end, min_y), [255,255,255], 5)
            cv2.line(frame, (int((left_x_start+right_x_start)/2),max_y), (int((left_x_end+right_x_end)/2),min_y), [0,255,255], 5)
            current = (left_x_end+right_x_end)/2
    
        except:
            current = 256
        return current

    def rain(self, image, prob):
        rain = np.zeros(image.shape, np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn > (1 - (prob) / 1000):
                    for k in range(15):
                        rain[i - k][j] = 255
                else:
                    rain[i][j] = image[i][j]
        return rain

    def snow(self, image, num_points):
        height, width, _ = image.shape
        for _ in range(num_points * 400 + 1):
            i = random.randint(0, width - 1)
            j = random.randint(0, height - 1)
            k, l = 10, 10
            thickness = random.choice([1, 2])
            snow = cv2.circle(image, (i - k, j - l), 2, (255, 255, 255), thickness)
        return snow

    def haze(self, image, alpha):
        white_image = np.ones_like(image) * 255
        haze = cv2.addWeighted(image, 1 - (alpha) / 10, white_image, (alpha) / 10, 0)
        return haze

    def turn_direction(self, steer, t, car, current, steerings, speeds, x, references, x_errors, y, j, m, first):

        if steer>0:
            print('turning right')
        else:
            print('turning left')

        time_i = time.time()

        while ((time.time()-time_i)<t):
            car.getData()
            car.setSteering(steer)
            car.setSpeed(10)

            steerings.append(steer)
            speeds.append(car.getSpeed()*3)
            x.append((current[i]/512)*j + 0.05*random.randint(-1,1))
            references.append((current[i]/512)*j)
            x_errors.append((current[i]/512)*j-((current[i]/512)*j + 0.05*random.randint(-1,1)))
            y.append(y[-1]+0.21*car.getSpeed()*3)


        return steerings, speeds, x, references, x_errors, y
    
    def unnorm(self, img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(s)

        return img
    
    def transform(self, ):
        return self.transform
