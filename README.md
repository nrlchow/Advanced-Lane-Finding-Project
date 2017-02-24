
**Advanced Lane Finding Project**

In this project, my goal is to write a software pipeline to identify the lane boundaries in a video.

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/undistort_road_image.png "Road Transformed"
[image3]: ./output_images/binary_combo_image.jpg "Binary Example"
[image4]: ./output_images/extracted_ROI.png "ROI Extract"
[image5]: ./output_images/binary_image_priorROI.png "Binary image priorROI"
[image6]: ./output_images/binary_image_ROI.png "Binary image on ROI"
[image7]: ./output_images/warped_straight_lines.png "Warp Example"
[image8]: ./output_images/warped_Histogram.jpg "Histogram"
[image9]: ./output_images/identified_pixels_fitLine.png "Fit Visual"
[image10]: ./output_images/filled-lane-boundary.png "Output"
[image11]: ./output_images/filled-lane-boundary2.png "Output"
[image12]: ./output_images/filled-lane-boundary3.png "Output"

[image_chess]: ./camera_cal/calibration1.jpg "Chessboard Example"
[chess_corner_detection]:  ./output_images/detected_corners_image.png "Chess corners detected]"

Video Reference:

### Project video

[video1]: 

[![Alt text](https://img.youtube.com/vi/p2uZCqJTjrc/0.jpg)](https://www.youtube.com/watch?v=p2uZCqJTjrc)


### README

### Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

#### 1.Camera Calibration
The code for this step is contained in the second code cell[2] of the IPython notebook located in "./Advanced-Lane-Finding.ipynb".
I computed camera calibration matrix to undistort the images.Camera lenses add distortions to the images whcih cuases misreading of lines and curvature.
I used 'objpoints' and 'imgpoints' to compute the camera calibration matrix.The 'object points',are (x, y, z) coordinates of the chessboard corners. 
Object points were appened with the (x,y) pixel position of each of the corners in the image everytime all chess board corners are detected in test images. 
Then I have computed distortion coefficients using the 'cv2.calibrateCamera()' function. 

Here OpenCV detects the chessboard squares:

chessboard Example
![alt text] [[image_chess] 

Chessboard corners detection Example 
 
![alt text][chess_corner_detection]


#### 2. Apply a distortion correction to raw images
I then used the camera distortion coefficients to correct the distortion. I used the 'cv2.undistort()' function to undistorting the images.
The code for this step is contained in the code cell[3] of the IPython notebook located in "./Advanced-Lane-Finding.ipynb".

Distortion corrected Chessboard Image Example
### 
![alt text][image1]


Distortion corrected raw road image Example
### 
![alt text] [image2]

#### 3. Use color transforms, gradients, etc., to create a thresholded binary image.

The code for this step is contained in the code cells [5][6][7][13] of the IPython notebook located in "./Advanced-Lane-Finding.ipynb".
In this step, I created a binary image to allow the lane lines to be clearly visible over noises.I have used gradient based thresholding technique to get that.
I used a combination of sobel operators over x,y direction and magnitude to filter out the noise.The sobel operator uses grayscale images and detects contrast and edges.
I also used HLS and HSV color thresholding algorithms to detect contrast between the lanes and the road.I found the S channel of HLS and V channel of HSV provided a good contrast 
and detect lines under shadows.I then combined gradient and color thresholds to produce bianry image. I also added region of interest mask over the image in order to rid of noise. 
I used a polygon mask which gives a triangler mask after clearing out noises.  In the code cell[13], I created a function named "binary_image_pipeline "that 
used thresholds of the x and y gradients,the overall gradient magnitude,and the gradient direction to focus on pixels that are likely to be part of the lane lines.



Here are the Region of Interest (ROI) vertices that I used,

roi_vertices = [230,720],[600,432],[680,432], [1150,720]

| ROI Vertices        
|:-------------:
| 230, 720
| 600, 432     
| 680, 432      
| 1150,720      


Binary Combo Image Example
![alt text] [image3] 

Extracted ROI Mask example

![alt text][image4] 

Binary image prior to applying ROI 
![alt text][image5]


Binary image after applying ROI 
![alt text][image6]


#### 4. Apply a perspective transform to rectify binary image ("birds-eye view").

The code for this step is contained in the code cell[9][10] of the IPython notebook located in "./Advanced-Lane-Finding.ipynb".
In order to calculate the road curvature from the camera of the vehicle, we need to transform the perspective so that lane lines look parallel.
I used open cv's perspective_transform function to get a birds eye view of the road that allows to calculate curvature.
I needed to select 4 points in the source and destination of the transform for perspective transform.The inputted source points warped to the output points or coordinates.
Because we want our lane lines to become parallel,I chose 2 points on each line of a straight road as input.
The output points I chose by roughly imaging where the points will be if the lane lines were paralel.
I verified that my perspective transform gave near parallel line by drawing the 'src' and 'dst' points onto the test images.
The inverse perspective tranform was later used do draw the lane on the original frame.

Here are my points:

src = (585, 460), (203, 720), (1127,720), (695,460)
dst = (320,0), (320,720), (960,720), (960,0)


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585,  460      | 320, 0      | 
| 203,  720      | 320, 720    |
| 1127, 720      | 960, 720    |
| 695,  460      | 960, 0      |


Perspective transform output

![alt text][image7]


#### 5. Detect lane pixels and fit to find the lane boundary.

The code for this step is contained in the code cells[16][18] of the IPython notebook located in "./Advanced-Lane-Finding.ipynb".
In this step,I got the peak of the left and right halves of the histogram which are treated as the starting point for the left and right lines. 
To do this,the function took the binary image as an input that stepped through the windows one by one and extracted left and right line pixel positions.
This code to do the above could be found in "line_sliding(binary_warped)" function in ipython notebook from line ... to...

Then, I used the above extracted lane lines pixels positions to represent a 2nd order polynomial using numpy.polyfit and poly1d.First it performed a least squares polynomial fit 
The polyfit functin returned a list of coefficients for left and right lines respectively.The polynomial coefficients are then evaluated to get x values for line y using ploy1d function. 
The code for can be found the same function from line number ... to line number ....Then I recast these x and y points in usable format for cv2.fillPoly().   

Warped Histogram Example

![alt text][image8]


Identified lanee pixels and fit line example

![alt text][image9]


#### 6. Determine the curvature of the lane and vehicle position with respect to center.

The code for this step is contained in the code cells[19] of the IPython notebook located in "./Advanced-Lane-Finding.ipynb".
The measure_curvature function was used to calculate the curvature of the road and find the vehicle posion with respect to center of the road.
The function took previsouly extracted x and y values as an input.In this step, I coverted the pixel based 2nd oder polynomial functions into real life measurement meters space.

The function calculated the curvature and offest. To calculate the offset it was assumed that the camera is pointing to the middle of the lane and used the pixel to 
meters conversion to calculate the offset. Negative number showed the vehicle on the left-hand side of the lane.

The lane center is the midpoint at the bottom of the image between the two lines that are detected. The distance of the lane center is the distance from the center of the lane.


#### 7. Warp the detected lane boundaries back onto the original image.

The code for this step is contained in the code cell [20] of the IPython notebook located in "./Advanced-Lane-Finding.ipynb".
The draw_lane function was the pipeline that was used to process images and video frame by frame.It first undistorted the image,and applied binary threshold,warp perspective.
The lane sliding functions implemented window sliding operations.It extracted line pixels,calculated curvature and distance off center using measure_curvature function.
Finally, it wrote the curvature and distance off centre data on image and video frame using open cv's putText function.  


### 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Here's a the final result of the process

Filled lane boundary Examples: 


![alt text][image10]
![alt text][image11]
![alt text][image12]


### Video

I used the draw_lane pipeline to fit new data into the polynomial that I already calculated. The pipeline was used to process video frame by frame.

Final output over the provided "project_video" video:

I also attempted to process challenge and harder challenge video though expected results were not produced.I required to create an additonal 
binary_image_pipeline_challenge function in order to exclude HSV color threshold from the function.


### Discussion
In this project,I used various thresholded techniques in order to detect the lane lines from raw pixels.It was an improvement over previsouly completed linear line detection project. 
I was able to detect the lane lines on low contrast environments and under shadows. The algorithm detected curvy lane lines using a 2nd order polynomial function 
before applying the gradient based edge detection and color transformation functions utilizing different binary algorithms in the preprocessing steps. 

The algorithm was still not robust enough to detect any given curvy lines.If the lines can't be detected for any reasons, the algorithm will likely fail. 
This algorithm used only center camera position to detect line center offset. It can be made more robsut by taking multiple camera positions into algorithm.
The pipeline can also be made more robust through averaging over a few video frames 

As the pipelines and methods are camera dependent, It requires camera calibration process. It also required parameters changes to make them workable under given data images and video frames.
The method will likely fail where there is only one lane line or more than two lane lines for some reason like remarking of lane line during roadwork.
The algorithm could be improved to work under such given conditions. 
