import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

random.seed(2022484)
np.random.seed(2022484)
cv2.setRNGSeed(2022484)

# Clustering functions
def ColorHist(img, bins=(8,8,8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def ClusterColor(dirPath, k=3, bins=(8,8,8)):
    paths = []
    for f in os.listdir(dirPath):
        if f.lower().endswith(('.jpg','.png','.jpeg')):
            paths.append(os.path.join(dirPath, f))
    paths.sort()
    feats = []
    for p in paths:
        im = cv2.imread(p)
        feats.append(ColorHist(im, bins=bins))
    feats = np.array(feats)
    km = KMeans(n_clusters=k, random_state=2022484)
    km.fit(feats)
    labs = km.labels_
    clust = [[] for _ in range(k)]
    for i, lab in enumerate(labs):
        clust[lab].append(paths[i])
    return clust

# SIFT functions
def SiftKeyDesc(img):
    # Convert image to grayscale before SIFT computation.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, desc = cv2.SIFT_create().detectAndCompute(gray, None)
    return kp, desc

# Matching functions
def MatchBF(dA, dB, thresh):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(dA, dB)
    good = [m for m in matches if m.distance < thresh]
    return sorted(good, key=lambda x: x.distance)

def MatchFLANN(dA, dB, ratio):
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(dA, dB, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

# Draw both BF and FLANN matches
def DrawCombinedMatches(imgA, kpA, imgB, kpB, bfMs, flMs):
    bfMv = cv2.drawMatches(imgA, kpA, imgB, kpB, bfMs, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    flannMv = cv2.drawMatches(imgA, kpA, imgB, kpB, flMs, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.title("Brute Force Matches")
    plt.imshow(cv2.cvtColor(bfMv, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.title("FLANN Based Matches")
    plt.imshow(cv2.cvtColor(flannMv, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Homography functions
def RansacHomog(kpA, kpB, ms):
    sA = np.float32([kpA[m.queryIdx].pt for m in ms]).reshape(-1,1,2)
    sB = np.float32([kpB[m.trainIdx].pt for m in ms]).reshape(-1,1,2)
    H, mask = cv2.findHomography(sA, sB, cv2.RANSAC, 2.0)
    return H, mask

def SaveHomog(H, fname):
    np.savetxt(fname, H, delimiter=',')
    return 0

# Warp and Crop functions
def WarpImg(img, H, size=(800,600)):
    return cv2.warpPerspective(img, H, size)

def CropBlack(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(g, 1, 255, cv2.THRESH_BINARY)[1]
    cd = cv2.findNonZero(th)
    x, y, w, h = cv2.boundingRect(cd)
    return img[y:y+h, x:x+w]

# Stitching functions
def Stitch2Img(iA, iB, H):
    hA, wA = iA.shape[:2]
    hB, wB = iB.shape[:2]
    cA = np.float32([[0,0],[wA,0],[wA,hA],[0,hA]]).reshape(-1,1,2)
    cAT = cv2.perspectiveTransform(cA, H)
    cB = np.float32([[0,0],[wB,0],[wB,hB],[0,hB]]).reshape(-1,1,2)
    ac = np.concatenate((cAT, cB), axis=0)
    xmin, ymin = np.int32(ac.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(ac.max(axis=0).ravel() + 0.5)
    T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float32)
    rw = xmax - xmin
    rh = ymax - ymin
    PanoHard = cv2.warpPerspective(iA, T @ H, (rw, rh))
    r0 = -ymin
    r1 = hB - ymin
    c0 = -xmin
    c1 = wB - xmin
    PanoRaw = PanoHard.copy()
    PanoRaw[r0:r1, c0:c1] = iB
    RegionA = PanoHard[r0:r1, c0:c1].astype(np.float32)
    RegionB = iB.astype(np.float32)
    blend = ((RegionA + RegionB) / 2).astype(np.uint8)
    PanoBlend = PanoHard.copy()
    PanoBlend[r0:r1, c0:c1] = blend
    return PanoRaw, PanoBlend

def StitchList(paths):
    base = cv2.imread(paths[0])
    for idx in range(1, len(paths)):
        nxt = cv2.imread(paths[idx])
        kA, dA = SiftKeyDesc(base)
        kN, dN = SiftKeyDesc(nxt)
        ms = MatchBF(dA, dN, thresh=float('inf'))
        H, _ = RansacHomog(kA, kN, ms)
        base, _ = Stitch2Img(base, nxt, H)
    return base

# Plot images in each cluster
def PlotClustersImages(clusters):
    rows = len(clusters)
    max_cols = max(len(c) for c in clusters)
    plt.figure(figsize=(15, 4 * rows))
    for i, cluster in enumerate(clusters):
        for j, path in enumerate(cluster):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(rows, max_cols, i * max_cols + j + 1)
            plt.imshow(img)
            plt.title(f"C{i+1}-{j+1}")
            plt.axis("off")
    plt.tight_layout()
    plt.show()

# Combine raw and blended panorama images
def PlotCombinedPanorama(rawImg, blendImg):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Raw Panorama")
    plt.imshow(cv2.cvtColor(rawImg, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.title("Blended & Cropped Panorama")
    plt.imshow(cv2.cvtColor(blendImg, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Combine cluster panorama images into one plot
def PlotClusterPanoramas(clusters):
    clusterPanos = []
    for cl in clusters:
        pano = CropBlack(StitchList(cl))
        clusterPanos.append(pano)
    cols = len(clusterPanos)
    plt.figure(figsize=(5 * cols, 5))
    for i, pano in enumerate(clusterPanos):
        plt.subplot(1, cols, i+1)
        plt.imshow(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))
        plt.title(f"Cluster {i+1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Main code
imgDir = "X:/Projects/CV/Assignment 2/Dataset/Panorama"
k = 3
clust = ClusterColor(imgDir, k=k, bins=(8,8,8))

pA = os.path.join(imgDir, "image1.png")
pB = os.path.join(imgDir, "image2.png")
imgA = cv2.imread(pA)
imgB = cv2.imread(pB)

kpA, descA = SiftKeyDesc(imgA)
kpB, descB = SiftKeyDesc(imgB)
print("Keypoints in image1:", len(kpA), "| Keypoints in image2:", len(kpB))
iA_kp = cv2.drawKeypoints(imgA, kpA, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
iB_kp = cv2.drawKeypoints(imgB, kpB, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(iA_kp, cv2.COLOR_BGR2RGB))
plt.title("Image1 Keypoints")
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(iB_kp, cv2.COLOR_BGR2RGB))
plt.title("Image2 Keypoints")
plt.show()

bfMatches = MatchBF(descA, descB, thresh=220)
print("\nBrute Force Feature Matches:", len(bfMatches))
flannMatches = MatchFLANN(descA, descB, ratio=0.65)
print("FLANN Based Feature Matches:", len(flannMatches))
DrawCombinedMatches(imgA, kpA, imgB, kpB, bfMatches, flannMatches)

H, _ = RansacHomog(kpA, kpB, flannMatches)
print("\nHomography:\n", H)
SaveHomog(H, fname="X:/Projects/CV/Assignment 2/2022484_Homography.csv")
print("\nHomography Matrix Saved as a .csv")

wA = WarpImg(imgA, H, (800,600))
iden = np.eye(3, dtype=np.float32)
wB = WarpImg(imgB, iden, (800,600))
sb = np.hstack((wA, wB))
plt.figure()
plt.title("Warping side-by-side")
plt.imshow(cv2.cvtColor(sb, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

panoRaw, panoBlend = Stitch2Img(imgA, imgB, H)
panoBlendCrop = CropBlack(panoBlend)
PlotCombinedPanorama(panoRaw, panoBlendCrop)

print("\nPlotting images in each cluster:")
PlotClustersImages(clust)

print("\nPlotting Panorama for each cluster:")
PlotClusterPanoramas(clust)