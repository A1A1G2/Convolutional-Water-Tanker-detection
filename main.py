import os
import cv2
import numpy as np
from scipy import signal

def corr2d(kernel,image):

    rng = np.random.default_rng()
    img = image- image.mean()
    template = np.copy(kernel)
    template -= template.mean()
    img = img + rng.standard_normal(img.shape) * 50 
    result = signal.correlate2d(img, template, boundary='symm', mode='same')
    return result

def conv2d(image, kernel):
    (image_height, image_width) = image.shape[:2]
    (kernel_height, kernel_width) = kernel.shape[:2]

    buffHeight = int(kernel_height/2)
    buffWidth = int(kernel_width/2)

    output = np.zeros((image_height + buffHeight*2, image_width + buffWidth*2), dtype="float32")
    output[buffHeight:-buffHeight, buffWidth:-buffWidth] = image.astype(np.float32)/ 255.0

    (output_height,output_width) = output.shape[:2]

    result = np.zeros((image_height, image_width), dtype="float32")

    # Görüntü ve çekirdek boyutlarına göre döngü başlatın
    for y in np.arange(0, image_height):
        for x in np.arange(0, image_width):
            # Görüntüdeki bölgeyi alın ve çekirdek ile çarpın
            roi = output[y : y + kernel_height, x : x + kernel_width]
            result[y, x] = (np.multiply(roi, kernel)).sum()

    # Görüntüyü kenar boşluğunu keserek çıktı olarak döndürün
    return result

def createKernleByImage(image_path: str):
    filterImage = cv2.imread(image_path)
    filterImage = cv2.cvtColor(filterImage, cv2.COLOR_BGR2GRAY)
    
    if(filterImage.shape[0] % 2 == 0):
        filterImage = np.insert(filterImage, 0, 0, axis=0)
    if(filterImage.shape[1] % 2 == 0):
        filterImage = np.insert(filterImage, 0, 0, axis=1)

    kernel = (filterImage.astype(np.float32) / 255.0)*2 - 1
    if(kernel.sum() != 0):
        kernel = kernel / kernel.sum()
    return kernel




def main():
    kernel_path = "Data/filter/filter3.jpg"
    directory = "Data"


    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Resmi oku
            img_path = os.path.join(directory, filename)
            image = cv2.imread(img_path)
            print("-----------İslenecek resim: ",img_path.split("/")[-1].split(".")[0])
            
            # Resmi "images" listesine ekle
            if image is not None:
                print("\nCorr2D")
                ILImage = image.copy()
                findByCorr2D(ILImage,kernel_path)
                print("MyCode\n")
                ILImage = image.copy()
                findByMyCode(ILImage,kernel_path)
                print("OpenCV\n")
                ILImage = image.copy()
                findByOpenCV(ILImage,kernel_path)
        else:
            continue



    
    cv2.destroyAllWindows()


def findLocationsByLocalMaxima(image,featureMap,kernel):

    finded = 0
    ILRes = featureMap.copy()

    (h, w) = kernel.shape[:2]
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(featureMap)
    threshold = 0.8 * max_val

    ILRes[ILRes< threshold] = 0

    #local maximas
    kernelLM = np.ones((5, 5), dtype=np.uint8)
    kernelLM[2, 2] = 0
    imageLM = cv2.dilate(ILRes, kernelLM)
    localMaxima = np.array(ILRes > imageLM, dtype=np.uint8)
    
    closingKernel = np.ones((5, 5), np.uint8)
    localMaxima = cv2.morphologyEx(localMaxima, cv2.MORPH_CLOSE, closingKernel)

    loc = np.where( localMaxima > 0)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), 255, 2)
        finded +=1
    return finded
    
    

def findByOpenCV(image,kernelP):


    finded = 0
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.imread(kernelP)
    kernel = cv2.cvtColor(kernel, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(image_gray,kernel,cv2.TM_CCOEFF_NORMED)
    
    finded += findLocationsByLocalMaxima(image,res,kernel)
    print("finded: ",finded)

    cv2.imshow("featureMap", res)
    cv2.waitKey(0)
    cv2.imshow("OpenCV", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def findByMyCode(image,kernelP):

    kernel = createKernleByImage(kernelP)
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ILImage = image_gray.copy()
    finded = 0

    featureMap = conv2d(ILImage,kernel)
    cv2.imshow("featuremap", featureMap)
    cv2.waitKey(0)
    finded += findLocations(image,ILImage,featureMap,kernel)
    kernel = pooling(kernel)

    featureMap = conv2d(ILImage,kernel)
    cv2.imshow("featuremap", featureMap)
    cv2.waitKey(0)
    finded += findLocations(image,ILImage,featureMap,kernel)
    print("finded: ",finded)

    cv2.imshow("MyCode", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def findByCorr2D(image,kernelP):

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.imread(kernelP)
    kernel = cv2.cvtColor(kernel, cv2.COLOR_BGR2GRAY)

    kernel = kernel - kernel.mean()

    ILImage = image_gray.copy()
    featureMap = corr2d(kernel,image_gray)
    finded = findLocations(image,ILImage,featureMap,kernel)
    print("finded: ",finded)

    cv2.imshow("featuremap", featureMap)
    cv2.waitKey(0)

    cv2.imshow("Corr2D", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def findLocations(image,ILImage,featureMap,kernel):

    buffX = int(kernel.shape[1]/2)
    buffY = int(kernel.shape[0]/2)
    finded = 0
    min_val , max_val, min_loc, max_loc = cv2.minMaxLoc(featureMap)
    paramA = max_val*0.90
    while(True):
        min_val , max_val, min_loc, max_loc = cv2.minMaxLoc(featureMap)
        if((max_val < paramA and finded > 0)or finded>2):
            break
        minX = max_loc[0]-buffX if max_loc[0]-buffX > 0 else 0
        minY = max_loc[1]-buffY if max_loc[1]-buffY > 0 else 0

        maxX = max_loc[0]+buffX if max_loc[0]+buffX < featureMap.shape[0] else featureMap.shape[0]
        maxY = max_loc[1]+buffY if max_loc[1]+buffY < featureMap.shape[1] else featureMap.shape[1]

        cv2.rectangle(image, (minX, minY), (maxX, maxY), (0, 0, 255), 2)
        featureMap[minY:maxY, minX:maxX] = 0
        ILImage[minY:maxY, minX:maxX] = 0
        finded += 1
    return finded


def pooling(kernel):
    #pooling 2by2 average for kernel
    (kernel_height, kernel_width) = kernel.shape[:2]
    result = np.zeros((int(kernel_height/2), int(kernel_width/2)), dtype="float32")
    for y in np.arange(0, kernel_height, 2):
        yindex = int(y/2)
        for x in np.arange(0, kernel_width, 2):
            if(x+2 < kernel_width and y+2 < kernel_height):
                result[yindex, int(x/2)] = (kernel[y:y+2, x:x+2].sum())/4
    return result

def shrink(kernel,dNumber=2):
    (kernel_height, kernel_width) = kernel.shape[:2]
    result = np.zeros((int(kernel_height-dNumber), int(kernel_width-dNumber)), dtype="float32")
    for y in np.arange(0, kernel_height-dNumber):
        for x in np.arange(0, kernel_width-dNumber):
            if(x+2 < kernel_width and y+2 < kernel_height):
                result[y, x] = (kernel[y:y+dNumber, x:x+dNumber].sum())/(dNumber**2)
    return result



if __name__ == "__main__":
    main()