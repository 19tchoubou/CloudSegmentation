from PIL import Image
import numpy as np
import cv2
from skimage.filters import gaussian
from skimage.segmentation import watershed

class Segmenter():
    def __init__(self):
        pass

    def _create_circular_mask(self, h, w, radius, center=None):
        """creates a circular mask, masking pixels further from center 
        than the radius

        Args:
            h (int): image height
            w (int): image width
            radius (int): radius of the applied mask
            center (tuple, optional): Mask center coordinates. Defaults to None.

        Returns:
            ndarray: the circular mask
        """
        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask

    def _sun_segmentation(self, img_to_modif, based_on_img, sun_thres, sun_val):
        # Modify the img_to_modif according to sun position on the based_on_img
        #It is better to manually fit sun_thres based on your own dataset. Usually, 1.5 < sun_thres < 2.5
        if not sun_thres:
            sun_thres = np.percentile(based_on_img, 90)
        img_to_modif[based_on_img > sun_thres] = sun_val

    def _dist_to_sun(self, sun_mask):
        """
        Args:
            sun_mask (ndarray): mask where sun pixels are set to True

        Returns:
            ndarray: distances to the sun
            bool: whether the image has a sun or not
        """
        #compute relative distances to sun center, on a sun-masked image
        M = cv2.moments(sun_mask)
        # retrieve sun center coordinates
        try:
            x_maxi, y_maxi = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]) #coordinates of the center of the sun
            has_sun = True
        except:
            x_maxi, y_maxi = sun_mask.shape[1]/2, sun_mask.shape[0]/2
            has_sun = False
        x, y = np.meshgrid(np.linspace(0, 1, sun_mask.shape[1]), np.linspace(0, 1, sun_mask.shape[0]))
        relative_dist = (x - x_maxi/sun_mask.shape[1])**2 + (y - y_maxi/sun_mask.shape[0])**2
        return relative_dist, has_sun


    def _high_pass_filter(self, img, sigma):
        """
        Args:
            img (ndarray): the image used to find the elevation map

        Returns:
            ndarray: the elevation map
        """
        mean = np.nanmean(img)
        img[np.isnan(img)] = mean
        gau = gaussian(img, sigma = sigma)
        gaubis = gaussian(img, sigma = sigma/1.6) #1.6 is said to be the best ratio in the literature
        filtered_image = gaubis - gau
        return filtered_image

    def _adaptative_threshold(self, image):
        """
        Args:
            image (ndarray): the image used to find the markers

        Returns:
            float: the threshold maximizing the Kullback-Leibler information between original and thresholded images
        """
        def target(hist, bins, t):
            m1 = np.sum(hist[bins <= t]*bins[bins <= t])
            m2 = np.sum(hist[bins >= t]*bins[bins >= t])
            s1, s2 = np.sum(hist[bins <= t]), np.sum(hist[bins >= t])
            mu1, mu2 = m1/s1, m2/s2
            res = -m1 * np.log(mu1) - m2 * np.log(mu2)
            return res


        def tar2(t, his, bin_edges):
            """ Applies target to each element of t, with the fixed parameters his, and bin_edges
            """
            get_target = lambda x: target(his, bin_edges, x)
            return np.vectorize(get_target)(t)

        im = image[~np.isnan(image)] 
        # Find the best segmentation threshold for the (R-B)/(R+B) image, based on its histogram
        his, bin_edges = np.histogram(im, bins = 100)
        his = his/np.sum(his)
        starter = np.percentile(im, 10)
        finisher = np.percentile(im, 90)
        t = np.linspace(starter, finisher, 100)
        bins = bin_edges[:-1]
        targets = tar2(t, his, bins)
        
        t_star = t[np.argmin(targets)]
        return t_star


    def _markers_from_threshold(self, markers, im, dist):
        """
        Args:
            markers (ndarray): the array specifying the position of marked pixels
            im (ndarray): the image used to find the markers
            dist (ndarray): the relative distances from the center of the sun

        Returns:
            ndarray: the elevation map
        """
        t_star_near = self._adaptative_threshold(im[(dist < 0.04)*(dist > 0.02)])
        t_star= self._adaptative_threshold(im[(dist < 0.06)*(dist > 0.04)])
        t_starf = self._adaptative_threshold(im[(dist < 0.1)*(dist > 0.06)])
        t_star_far = self._adaptative_threshold(im[dist > 0.06])
        
        markers[im > 1000*(dist < 0.06) + 1.1*t_star_far] = 2
        markers[im < -1000*(dist < 0.06) + t_star_far] = 1

        markers[im > 1000*np.abs(1 - (dist < 0.1)*(dist > 0.06)) + 1.2*t_starf] = 2
        markers[im < -1000*np.abs(1 - (dist < 0.1)*(dist > 0.06)) + t_starf] = 1
            
        markers[im > 1000*np.abs(1 - (dist < 0.06)*(dist > 0.04)) + 1.2*t_star] = 2
        markers[im < -1000*np.abs(1 - (dist < 0.06)*(dist > 0.04)) + t_star] = 1
            
        markers[im > 1000*np.abs(1 - (dist < 0.04)*(dist > 0.02)) + 1.1*t_star_near] = 2
        markers[im < -1000*np.abs(1 - (dist < 0.04)*(dist > 0.02)) + t_star_near] = 1

    def segment(self, image, radius):
        #Feature extraction: finding sum image, (R-B) / (R+B)
        image = np.array(image)
        h, w = image.shape[:2]


        # Square crop
        if w > h:
            image = image[:, w//2 - h//2 : w//2 + h//2]
        else:
            image = image[(h-w)//2 : (h+w)//2, :]
    
        # Circular mask
        mask = self._create_circular_mask(min(h, w), min(h, w), radius=radius)
        print(mask.shape)
        
        masked_img = image.copy()
        masked_img[~mask] = np.nan
        sum_image = masked_img.copy()

        sum_image = masked_img[:, :, 0] + masked_img[:, :, 1] + masked_img[:, :, 2]

        r_b_img = (masked_img[:, :, 0] - masked_img[:, :, 2])/(masked_img[:, :, 0] + masked_img[:, :, 2])
        r_b_img -= np.nanmin(r_b_img)
        r_b_img = r_b_img / np.nanmax(r_b_img)

        sun_mask = np.zeros_like(r_b_img)
        markers = np.zeros_like(r_b_img)

        self._sun_segmentation(sun_mask, sum_image, sun_thres = 2.95, sun_val=3)

        relative_dist, has_sun = self._dist_to_sun(sun_mask)

        filtered_image = self._high_pass_filter(r_b_img, sigma=4)

        if has_sun:
            self._markers_from_threshold(markers, r_b_img, relative_dist)

        else:
            t_star= self._adaptative_threshold(r_b_img)
            markers[r_b_img > t_star] = 2
            markers[r_b_img < t_star] = 1
            markers[~mask] = 0
            return markers

        segmentation = watershed(filtered_image, markers)
        segmentation[~mask] = 0
        self._sun_segmentation(segmentation, sum_image, sun_thres = 2.95, sun_val=3)
        return segmentation
