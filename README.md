# CloudSegmentation
Segmenting clouds from sky images using watershedding

## How to use this code

    import numpy as np
    from segmenter import Segmenter
    from PIL import Image
  
    image_to_segment = np.array(Image.open('data/fisheye.jpg'))/255
    segmenter = Segmenter()
    segmented_image = segmenter.segment(image_to_segment, radius = 140) #radius will need to be changed if you plan to use this code on your own images
    
## Results

![Results](https://github.com/19tchoubou/CloudSegmentation/blob/main/data/segmentation.PNG?raw=true)
