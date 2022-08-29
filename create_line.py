import cv2 
import os
# path 
path = r'C:\Users\mattp\Documents\thesis_Stuff\output\AKAZE\0011_0012_0.5.jpgAKAZE.jpg'
   
# Reading an image in default mode
image = cv2.imread(path)
   
# Window name in which image is displayed
window_name = 'Image'
  
# Start coordinate, here (0, 0)
# represents the top left corner of image

start_point = (image.shape[1]-200, 0)
  
# End coordinate, here (250, 250)
# represents the bottom right corner of image
end_point = (image.shape[1]-200, image.shape[0])
  
# Green color in BGR
color = (0, 255, 0)
  
# Line thickness of 9 px
thickness = 9
  
# Using cv2.line() method
# Draw a diagonal green line with thickness of 9 px
#image = cv2.line(image, start_point, end_point, color, thickness)
image2 = image[0:-1,image.shape[1]-200:-1]  
for subdir, dirs, files in os.walk("output"):
    if(subdir != "output"):
        print(subdir)
        for subdir2, dirs2, files in os.walk(subdir):
            #print(files)
            for file in files:
                test = (os.path.join(subdir, file))
                #print(test)
                img = cv2.imread(test)
                image2 = img[0:-1,img.shape[1]-200:-1]  
                #print(dirs)
                cv2.imwrite("cropped\\" + file + "_cropped.jpg", image2)
                #print(score)

# Displaying the image 
cv2.imshow(window_name, image2) 
cv2.waitKey(0)
