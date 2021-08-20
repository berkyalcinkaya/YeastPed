import math
import cv2
import numpy as np
from skimage.draw import line
from matplotlib import pyplot as plt


def plotImage(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


def computeMajorAxisVectorOverlap(cell_val, birth_idx, masks):

    # determine max index that an ellipse can be drawn (20 frames forward form birth is maximum)
    idx_dif = masks.shape[0] - birth_idx
    if idx_dif >= 20:
        max_idx = 21
    elif idx_dif >= 15:
        max_idx = 16
    elif idx_dif >= 10:
        max_idx = 10
    else:
        return []


#--------helper functions
    def getMajorAxisPoints(img, cell_val):
        #img_color = cv2.cvtColor(np.uint8(img.copy()), cv2.COLOR_GRAY2RGB)  # create a 3D image
        thresh = (np.where(img == cell_val, 255, 0)).astype("uint8")  # convert to a greyscale format

        #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contours on greyscale image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        #if len(contours) == 0:
            #plotImage(thresh)

        #get max contour
        big_contour = max(contours, key = lambda c: cv2.arcLength(c, True))

        #get ellipse and parameters
        ellipse = cv2.fitEllipse(big_contour)

        #result = img_color.copy()
        #cv2.ellipse(result, ellipse, (0,255,0), 3)

        (xc,yc),(d1,d2),angle = ellipse

        #get major axis coordinates
        rmajor = max(d1,d2)/2
        if angle > 90:
            angle = angle - 90
        else:
            angle = angle + 90
        xbot_ = xc + math.cos(math.radians(angle))*rmajor
        ybot_ = yc + math.sin(math.radians(angle))*rmajor
        xtop_ = xc + math.cos(math.radians(angle+180))*rmajor
        ytop_ = yc + math.sin(math.radians(angle+180))*rmajor
        #print(xtop_, ytop_, xbot_, ybot_)

        #get line info
        m = (ytop_ - ybot_)/(xtop_ - xbot_)
        #print("slope: " + str(m))
        m = abs(m)

        #line should be extended by 13 pixels
        delta_x = math.sqrt(13**2/(m**2+1))
        delta_y = delta_x * m

        if xtop_ > xbot_: #positive slope - xtop increases, ytop decreases (upwards)

            if round(xtop_ + delta_x) > masks.shape[2] or round(ytop_ - delta_y) < 0:
                xtop = xtop_
                ytop = ytop_
            else:
                xtop = xtop_ + delta_x
                ytop = ytop_ - delta_y

            if round(xbot_ - delta_x) < 0 or round(ybot_ + delta_y) > masks.shape[1]:
                xbot = xbot_
                ybot = ybot_
            else:
                xbot = xbot_- delta_x
                ybot = ybot_ + delta_y

        else: # negative slope, decrease xtop, add to ytop_

            if round(xtop_ - delta_x) < 0 or round(ytop_ - delta_y) > masks.shape[1]:
                xtop = xtop_
                ytop = ytop_
            else:
                xtop = xtop_ - delta_x
                ytop = ytop_ - delta_y

            if round(xbot_+ delta_x) > masks.shape[2] or round(ybot_+ delta_y) < 0:
                xbot = xbot_
                ybot = ybot_
            else:
                xbot = xbot_ + delta_x
                ybot = ybot_ + delta_y

        '''
        cv2.line(result, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 0, 255), 3)
        cv2.imshow(None, result)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        '''

        #print(xtop, ytop, xbot, ybot)
        return (xtop, ytop, xbot, ybot)


    def extractValuesFromLine(x0, y0, x1, y1):
        """Uses Xiaolin Wu's line algorithm to interpolate all of the pixels along a
        straight line, given two points (x0, y0) and (x1, y1)

        Wikipedia article containing pseudo code that function was based off of:
            http://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm
        """
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))

        rr, cc = line(y0, x0, y1, x1)

        #ensure indeces are in bounds
        rr = np.array([r for r in rr if r < masks.shape[1] and r >= 0])
        cc = np.array([c for c in cc if c < masks.shape[2] and c >= 0])

        #rr and cc should be same length
        if len(rr) < len(cc):
            cc = cc[:len(rr)]
        elif len(cc) < len(rr):
            rr = rr[:len(cc)]

        line_vals = img[rr,cc]
        img[rr,cc] = 0  # draw line

        #plotImage(img)

        line_vals = np.unique(line_vals[np.logical_and(line_vals!=0, line_vals!=cell_val)])
        return line_vals

#----------------end helper functions
    frame_overlaps = []
    for i in range(5,max_idx,5):
        img = masks[birth_idx+i]
        xtop, ytop, xbot, ybot = getMajorAxisPoints(img, cell_val)
        vals = extractValuesFromLine(xtop, ytop, xbot, ybot)
        for val in vals:
            frame_overlaps.append(val)

    return list(np.unique(np.array(frame_overlaps)))
