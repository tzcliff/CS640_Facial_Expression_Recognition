import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sys import maxsize

RAW_DATA_DIR = "presidential_videos/"
DATA_DIR = "dataset/"
CSV_FILE = "Labels.csv"
OUTPUT_CSV_FILE = "input.csv"

CLASS_NAMES = ["Positive", "Negative", "Neutral"]
CATEGORY_NAMES = ["Training", "PublicTest", "PrivateTest"]
OUTPUT_CSV_HEADERS = ["emotion", "pixels", "Usage"]


def cropImage(image):
    gray = image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    w, h = 0, 0
    for i in range(3, 6):
        faceCascadeProfile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        faceCascadeFront = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        facesProfile = faceCascadeProfile.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=i,
            minSize=(30, 30)
        )

        facesFront = faceCascadeFront.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=i,
            minSize=(30, 30)
        )

        if len(facesProfile) != 0:
            x, y, w, h = facesProfile[0]
            break
        elif len(facesFront) != 0:
            x, y, w, h = facesFront[0]
            break

    if w == 0 or h == 0:
        return None

    grayFace = gray[y:y + h, x:x + w]
    grayFace = cv2.resize(grayFace, (48, 48))

    return grayFace


def videoToImage(videoFile, desDir = ""):
    videoName = os.path.splitext(videoFile)[0]
    vidcap = cv2.VideoCapture(RAW_DATA_DIR + videoFile)
    success = True
    croppedImage = None

    while success:
        success, image = vidcap.read()

        if not success:
            break

        croppedImage = cropImage(image)

        if croppedImage is None:
            continue
        break

    if croppedImage is None:
        raise ValueError("Face not found.")

    if desDir != "":
        cv2.imwrite(os.path.join(desDir, "%s.jpg" % videoName), image)
        print("Converted %s to %s successfully!" % (videoName + ".mp4", videoName + ".jpg"))

    return croppedImage


def loadData(csvFile):
    df = pd.read_csv(csvFile)

    X = df[df.columns[0]].to_numpy()
    y = df[df.columns[1]].to_numpy()
    # split data into 3 parts (64%, 16%, 20%)
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=1)
    XTrain, XValidation, yTrain, yValidation = train_test_split(XTrain, yTrain, test_size=0.2, random_state=1)

    trainData = np.c_[XTrain, yTrain]
    validData = np.c_[XValidation, yValidation]
    testData = np.c_[XTest, yTest]

    dataList = [trainData, testData, validData]

    return dataList


def refactorData(csvFile):
    dataList = loadData(csvFile)
    outputDf = pd.DataFrame(columns = OUTPUT_CSV_HEADERS)
    for i in range(len(dataList)):
        print("%s Set\n" % CATEGORY_NAMES[i])
        X = dataList[i][:, 0]
        pixels = []

        np.set_printoptions(threshold=maxsize, linewidth=maxsize)

        for j in range(X.shape[0]):
            # get image
            print("Converting image %d" % j)
            pixel = videoToImage(X[j]).ravel()
            # convert numpy array to string
            pixelStr = np.array2string(pixel)[2:-1]
            pixels.append(pixelStr)

        # 0-Negative, 1-Neutral, 2-Positive
        y = dataList[i][:, 1]
        yOneHot = LabelEncoder().fit_transform(y)
        df = pd.DataFrame([yOneHot, pixels]).T
        df.columns = OUTPUT_CSV_HEADERS[:2]
        df[OUTPUT_CSV_HEADERS[-1]] = CATEGORY_NAMES[i]

        outputDf = pd.concat([outputDf, df], ignore_index=True)

    outputDf.to_csv(OUTPUT_CSV_FILE, index=False)


def refactorFolder(dataDir, csvFile):
    dataDir = os.path.join(dataDir)
    trainDir = dataDir + "train"
    validationDir = dataDir + "validation"
    testDir = dataDir + "test"
    dataDirs = [trainDir, validationDir, testDir]

    if not os.path.exists(dataDir):
        os.mkdir(dataDir)

    for directory in dataDirs:
        if not os.path.exists(directory):
            os.mkdir(directory)

    dataList = loadData(csvFile)

    count = 0
    for i in range(len(dataList)):
        desDirs = []
        # create new folder by class
        for j in range(len(CLASS_NAMES)):
            desDir = dataDirs[i] + "/" + CLASS_NAMES[j]
            desDirs.append(desDir)
            if not os.path.exists(desDir):
                os.mkdir(desDir)

        data = dataList[i]
        for j in range(data.shape[0]):
            file = data[j, 0]
            idx = CLASS_NAMES.index(data[j, 1])
            videoToImage(file, desDirs[idx])
            count += 1

    print("Total images: %s" % count)


# test cropImage function
# image = cv2.imread("joe.7wfrtnGV27k.00.jpg")
# cropImage(image)

refactorData(CSV_FILE)