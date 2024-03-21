import cv2
import numpy as np
import utils as utils
import matplotlib.pyplot as plt

def LoG_kern(sigma):
    """
    Return a gaussian kernel for the specified parameter sigma.
    """
    ker_size = int(np.ceil(6*sigma))
    kernel = np.empty((ker_size, ker_size), dtype=float)
    center = ker_size//2
    
    for ((i, j) , _) in np.ndenumerate(kernel):
        x = i-center
        y = j-center
        kernel[i , j] = ((- 1 / (np.pi * sigma**2) ) * (1 - (x**2 + y**2) / (2 * sigma**2) ) ) * np.exp(-( (x**2 + y**2) / (2 * sigma**2)))
    
    return kernel

def padding(img, sigma):
    """
    Return a padded version of the input image.
    Padding is half of the kernel size.

    :param img:         input image
    :param sigma:       
    """
    padding = int(np.ceil(6*sigma)) //2         
    padded_img = np.pad(img, ((padding, padding), (padding, padding)))
    return padded_img

def convolution(img, kernel):
    """
    Convolve the input image and the LoG kernel.
    """
    kernel_size = kernel.shape[0]
    padding = kernel_size//2

    rows = img.shape[0] - 2 * padding
    cols = img.shape[1] - 2 * padding

    conv_img = np.zeros(shape=(rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            conv_img[i , j] = np.sum(img[i : i + kernel_size , j : j + kernel_size]*
                kernel)
            
    return conv_img

def circle_mat(sigma):
    """
    Returns a matrix with a circle of radius sqrt(2)*sigma.
    """
    r = int(np.sqrt(2)*sigma)
    A = np.arange(-r,r+1)**2
    dists = np.sqrt(A[:,None] + A)
    mat = np.zeros(dists.shape)
    for i in range(2*r+1):
        for j in range(2*r+1):
            if dists[i, j]< r:
                mat[i, j] = 1

    return mat


def detect_blob(img, sigma, threshold, mask):
    """
    Perform blob detection. Find maximum and minimum response of the Log filter inside a circle of 
    radius sqrt(2)*sigma. If the response satisfies the condition imposed by the threshold, the blob 
    coordinates are stored.

    :param img:
    :param simga:
    :param threshold:       Lower and upper bound.
    :param mask:            Circle matrix of radius sqrt(2)*sigma.
    """
    coord=[]
    (rows,cols) = img.shape
    m = mask.shape[0]
    pad = int(m//2)
    img_p = np.pad(img, ((pad, pad), (pad, pad)))
    for i in range(rows):
        for j in range(cols):
            slice_img = img_p[i:i+m,j:j+m] 
            response = slice_img*mask   # Element-wise multiplication with circle matrix
            maxim = np.amax(response)   # Finding maximum and minimum
            minim = np.amin(response)
            if minim < threshold[0]:    # Lower bound
                x,y = np.argwhere(response==minim)[0].flatten()
                coord.append((-pad+i+x,-pad+j+y, np.sqrt(2)*sigma)) # Save coord and radius
            if maxim > threshold[1]:    # Upper bound
                x,y = np.argwhere(response==maxim)[0].flatten()
                coord.append((-pad+i+x,-pad+j+y, np.sqrt(2)*sigma)) # Save coord and radius
        
    return coord

def LoG(path, sigma, threshold = (-0.3, 0.3)):
    """
    This function applies the LoG filter to an input image.

    :param path:        path to input image
    :param sigma:       sigma (or sigmas) of the gaussian. 
    :param threshold:   lower and upper bound for blob detection. Default (-0.3, 0.3)
    """
    img = cv2.imread(path, 0)       # Convert to greyscale
    img = np.interp(img, (img.min(), img.max()), (-1, 1))   # Normalise
    
    centers = []
    for s in sigma:
        kernel = LoG_kern(s)               # Create LoG kernel
        img_p = padding(img, s)            # Create a padded image to match the kernel dimension
        conv = convolution(img_p, kernel)  # Perform convolution
        plt.imshow(img, cmap='gray')
        mask = circle_mat(s)               # Create circle matrix of radius sqrt(2)*sigma
        centers.append(list(set(detect_blob(conv, s, threshold, mask))))    # Find centres
    _, ax = plt.subplots()
    ax.imshow(img,cmap="gray")
    for sigma in centers:
        for blob in sigma:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=1., fill=False)
            ax.add_patch(c)
    ax.plot()  
    plt.show()