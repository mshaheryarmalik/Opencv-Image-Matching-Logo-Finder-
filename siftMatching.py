import cv2
from matplotlib import pyplot as plt
import glob

THRESH_HOLD_1 = 3


def validateGoodMatches(left_points, right_points):
    if len(left_points) < 45:
        return False
    maxCount = 0
    for i in range(len(left_points)):
        currCount = 0
        for x in range(len(left_points)):
            if (right_points[i] - THRESH_HOLD_1 < right_points[x] < right_points[i] + THRESH_HOLD_1) and (
                    left_points[i] - THRESH_HOLD_1 < left_points[x] < left_points[i] + THRESH_HOLD_1):
                currCount += 1
        if maxCount < currCount:
            maxCount = currCount
    if maxCount >= 0.3 * len(left_points):
        return False
    return True


def siftMatcher(image, logoToSearch):
    MIN_MATCH_COUNT = 45
    img1 = logoToSearch
    img2 = image
    try:
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=30)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        right_points = []
        left_points = []
        for m, n in matches:
            if m.distance < 0.65 * n.distance:
                good.append(m)
                # Store image co-ordinate of train image
                pt = kp2[m.trainIdx].pt
                right_points.append(abs(pt[0]))  # Store x co-ordinate
                left_points.append(abs(pt[1]))  # Store y co-ordinate
        # Check if good matches are really accurate or not
        foundReallyGood = validateGoodMatches(left_points, right_points)
        if foundReallyGood is True and len(good) > MIN_MATCH_COUNT:
            return len(good)
    except:
        print('No Matching Found!')
        return 0
    return 0


# Main
path = glob.glob("images/test/*.jpeg")
cv_imgs = []
count = 0
goodMatchesFound = 0
final_image = 0
maxMatches = 0
for img in path:
    n = cv2.imread(img)
    cv_imgs.append(n)
    count = count + 1

# Find logo in image
logoToSearch = cv2.imread("images/tags/test_image_04.jpeg", 0)  # Mobile Phone logo
print("Loading Images....")
for i in range(count):
    print("Searching Images....")
    maxMatches = siftMatcher(cv_imgs[i], logoToSearch)
    if maxMatches > goodMatchesFound:
        goodMatchesFound = maxMatches
        final_image = cv_imgs[i]
try:
    plt.imshow(final_image, 'gray'), plt.show()
except:
    print("Oops! No image found that matches logo.")
