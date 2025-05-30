import cv2
import numpy as np
import glob
import json
import matplotlib.pyplot as plt
import os
import random

random.seed(2022484)
np.random.seed(2022484)
cv2.setRNGSeed(2022484)

def CalibCam(folder, BoardSize, SqSize):
    objP = np.zeros((BoardSize[0] * BoardSize[1], 3), np.float32)
    objP[:, :2] = np.mgrid[0:BoardSize[0], 0:BoardSize[1]].T.reshape(-1, 2)
    objP *= SqSize
    objPts = []
    imgPts = []
    imgs = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    count = 0
    grayShape = None
    for f in imgs:
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayShape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, BoardSize, None)
        if ret:
            count += 1
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            objPts.append(objP)
            imgPts.append(corners)
    ret, camMat, dist, rVec, tVec = cv2.calibrateCamera(objPts, imgPts, grayShape, None, None)
    return ret, camMat, dist, rVec, tVec, objPts, imgPts, grayShape

def CalcReprojErr(objPts, imgPts, rVec, tVec, camMat, dist):
    errs = []
    for i in range(len(objPts)):
        projPts, _ = cv2.projectPoints(objPts[i], rVec[i], tVec[i], camMat, dist)
        err = cv2.norm(imgPts[i], projPts, cv2.NORM_L2) / len(projPts)
        errs.append(err)
    return errs

def CalcBoardNorm(rVec):
    norms = []
    for rv in rVec:
        R, _ = cv2.Rodrigues(rv)
        norm = R @ np.array([0, 0, 1], dtype=float)
        norms.append(norm)
    return norms

def SaveCalibJson(outPath, camMat, dist, rVec, tVec, reprojErrs):
    fx = camMat[0, 0]
    fy = camMat[1, 1]
    skew = camMat[0, 1]
    cx = camMat[0, 2]
    cy = camMat[1, 2]
    k1 = dist[0, 0]
    k2 = dist[0, 1]
    k3 = dist[0, 4] if dist.shape[1] >= 5 else 0.0
    meanErr = float(np.mean(reprojErrs))
    stdErr = float(np.std(reprojErrs))
    extr = []
    for i in range(min(2, len(rVec))):
        R, _ = cv2.Rodrigues(rVec[i])
        extr.append({
            "image_id": i + 1,
            "rotation_matrix": R.tolist(),
            "translation_vector": tVec[i].ravel().tolist()
        })
    data = {
        "intrinsic_parameters": {
            "focal_length": [float(fx), float(fy)],
            "skew": float(skew),
            "principal_point": [float(cx), float(cy)]
        },
        "extrinsic_parameters": extr,
        "radial_distortion_coefficients": [float(k1), float(k2), float(k3)],
        "reprojection_errors": {
            "mean_error": meanErr,
            "std_dev": stdErr
        }
    }
    with open(outPath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Calibration results saved to {outPath}")

def UndistortImgs(folder, camMat, dist, num=5):
    outDir = os.path.join(folder, "Undistort")
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    imgs = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    count = min(num, len(imgs))
    for i in range(count):
        path = imgs[i]
        img = cv2.imread(path)
        undist = cv2.undistort(img, camMat, dist)
        outPath = os.path.join(outDir, f"undistorted {i+1}.jpg")
        cv2.imwrite(outPath, undist)

def OverlayCorners(folder, objPts, imgPts, rVec, tVec, camMat, dist):
    outDir = os.path.join(folder, "Overlay")
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    imgs = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    for i, f in enumerate(imgs):
        img = cv2.imread(f)
        det = imgPts[i].reshape(-1, 2).astype(int)
        projPts, _ = cv2.projectPoints(objPts[i], rVec[i], tVec[i], camMat, dist)
        proj = projPts.reshape(-1, 2).astype(int)
        over = img.copy()
        for p in det:
            cv2.circle(over, (p[0], p[1]), 4, (0, 255, 0), -1)
        for p in proj:
            cv2.circle(over, (p[0], p[1]), 4, (0, 0, 255), -1)
        outPath = os.path.join(outDir, f"corners overlay {i+1}.jpg")
        cv2.imwrite(outPath, over)

def SaveNormals(outPath, norms):
    with open(outPath, "w") as f:
        for i, norm in enumerate(norms):
            f.write(f"Image {i+1}: {norm.tolist()}\n")
    print(f"Normals saved to {outPath}")

# Parameters for online dataset: 9 squares x 7 squares => 8x6 inner corners, 30 mm squares.
OnBoard = (8, 6)
OnSquare = 30.0
OnFolder = r"X:\Projects\CV\Assignment 2\dataset\online"

# Online Dataset
print("Calibrating ONLINE dataset")
retOn, camMatOn, distOn, rvecOn, tvecOn, objOn, imgOn, shapeOn = CalibCam(OnFolder, OnBoard, OnSquare)
errOn = CalcReprojErr(objOn, imgOn, rvecOn, tvecOn, camMatOn, distOn)
SaveCalibJson("X:/Projects/CV/Assignment 2/2022484_Parameters_Online.json", camMatOn, distOn, rvecOn, tvecOn, errOn)

UndistortImgs(OnFolder, camMatOn, distOn, num=5)
plt.figure()
plt.bar(range(1, len(errOn)+1), errOn)
plt.title("Re-projection Errors (ONLINE)")
plt.xlabel("Image")
plt.ylabel("Error (pixels)")
plt.savefig("X:/Projects/CV/Assignment 2/2022484_Reproj Error_Online.png")
plt.close()

OverlayCorners(OnFolder, objOn, imgOn, rvecOn, tvecOn, camMatOn, distOn)
normOn = CalcBoardNorm(rvecOn)
SaveNormals("X:/Projects/CV/Assignment 2/2022484_Normals_Online.txt", normOn)
print("ONLINE calibration complete.\n")

# Parameters for offline dataset: 11 squares x 8 squares => 10x7 inner corners, 25 mm squares.
OffBoard = (10, 7)
OffSquare = 25.0
OffFolder = r"X:\Projects\CV\Assignment 2\dataset\offline"

# Offline Dataset
print("Calibrating OFFLINE dataset")
retOff, camMatOff, distOff, rvecOff, tvecOff, objOff, imgOff, shapeOff = CalibCam(OffFolder, OffBoard, OffSquare)
errOff = CalcReprojErr(objOff, imgOff, rvecOff, tvecOff, camMatOff, distOff)
SaveCalibJson("X:/Projects/CV/Assignment 2/2022484_Parameters_Offline.json", camMatOff, distOff, rvecOff, tvecOff, errOff)

UndistortImgs(OffFolder, camMatOff, distOff, num=5)
plt.figure()
plt.bar(range(1, len(errOff)+1), errOff)
plt.title("Re-projection Errors (OFFLINE)")
plt.xlabel("Image")
plt.ylabel("Error (pixels)")
plt.savefig("X:/Projects/CV/Assignment 2/2022484_Reproj Error_Offline.png")
plt.close()

OverlayCorners(OffFolder, objOff, imgOff, rvecOff, tvecOff, camMatOff, distOff)
normOff = CalcBoardNorm(rvecOff)
SaveNormals("X:/Projects/CV/Assignment 2/2022484_Normals_Offline.txt", normOff)
print("OFFLINE calibration complete.")
print("\nCalibration steps complete for Both sets.")