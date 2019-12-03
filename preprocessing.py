import os
import sys
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


def videoToImage(videoFile, desDir="", crop=True, label=None, category=None):
    videoName = os.path.splitext(videoFile)[0]
    vidcap = cv2.VideoCapture(RAW_DATA_DIR + videoFile)

    frameCnt = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = 10

    # capture frames by step
    desiredFrames = [frame for frame in range(frameCnt) if (frame % step == 0)]

    images = []
    cnt = 0  # number of captured frames

    for dFrame in desiredFrames:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, dFrame)
        _, image = vidcap.read()

        if image is None:
            continue

        # convert int64 to uint8
        image = image.astype(np.uint8)

        if crop:
            image = cropImage(image)
            if image is None:
                continue

        if desDir != "":
            imageFile = "%s.%d.jpg" % (videoName, cnt)
            cv2.imwrite(os.path.join(desDir, imageFile), image)
            print("%s - frame %d ---> %s..." % (videoFile, cnt + 1, imageFile))
        else:
            # convert numpy array to string
            imageStr = " ".join(map(str, image.ravel().tolist()))
            images.append(imageStr)
        cnt += 1

    if cnt == 0:
        raise ValueError("Face not found.")

    if (label is None) or (category is None):
        return

    labels = np.repeat(label, cnt).astype(np.uint8)
    cates = np.repeat(category, cnt)

    newDf = pd.DataFrame({0: labels,
                         1: images,
                         2: cates})
    newDf[0] = newDf[0].astype(np.uint8)
    return cnt, newDf


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
    outputDf = pd.DataFrame()
    for i in range(len(dataList)):
        print("%s Set\n" % CATEGORY_NAMES[i])
        X = dataList[i][:, 0]

        # 0-Negative, 1-Neutral, 2-Positive
        y = dataList[i][:, 1]
        yLabel = LabelEncoder().fit_transform(y)

        for j in range(X.shape[0]):
            print("video - %s" % X[j], end=" ----> ")
            sys.stdout.flush()  # flush print buffer
            cnt, newDf = videoToImage(X[j], label=yLabel[j], category=CATEGORY_NAMES[i])
            outputDf = pd.concat([outputDf, newDf], axis=0, ignore_index=True)
            print("%d images" % cnt)

    outputDf.columns = OUTPUT_CSV_HEADERS
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
            videoToImage(file, desDirs[idx], crop=False)
            count += 1

    print("Total images: %s" % count)


choice = int(sys.argv[1])

if choice == 1:
    refactorData(CSV_FILE)
elif choice == 2:
    refactorFolder(dataDir=DATA_DIR, csvFile=CSV_FILE)
else:
    raise ValueError("the parameter must be 1 or 2!")
