import numpy as np
import cv2
import utils
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

if __name__ == "__main__":
    #https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
    # load the image and convert it from BGR to RGB so that
    # we can dispaly it with matplotlib
    image = cv2.imread("boulder1.jpg")
    small = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
    small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
   
    # reshape the image to be a list of pixels
    small = small.reshape((small.shape[0] * small.shape[1], 3))

    # cluster the pixel intensities
    cluster = 4
    jobs = 4
    clt = KMeans(n_clusters = cluster, n_jobs= jobs, algorithm="full")
    clt.fit(small)

    #return an array of array of the main colors I.E [[215,211,205],[122,144,155]]
    mainColor = np.array(clt.cluster_centers_)
    print(mainColor)

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = utils.centroid_histogram(clt)
    bar = utils.plot_colors(hist, clt.cluster_centers_)
    
    # show our color bart
    #plt.figure()
    #plt.axis("off")
    #plt.imshow(bar)
    #plt.show()

    im = Image.open("boulder1.jpg")   
    out = Image.new('RGB', im.size, 0xffffff)

    tol = 20
    width, height = im.size
    for x in range(width):
        for y in range(height):
            r,g,b = im.getpixel((x,y))
            for rgbMainColor in mainColor:
                if abs(rgbMainColor[0] - r) < tol and abs(rgbMainColor[1] - g) < tol and abs(rgbMainColor[2] - b) < tol:
                    out.putpixel((x,y), (0,0,0))
                    break
                else
                    out.putpixel((x,y), (r,g,b))
                    

    out.save('out.png')

    outImg = cv2.imread('out.png')
    plt.imshow(outImg)
    plt.show()

