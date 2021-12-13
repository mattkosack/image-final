import numpy as np
import matplotlib.pylab as plt
import ipywidgets as widgets
import cv2

from video import run_video, run_video_2

"""
This code needs some reorganization, but I did not want to mess with the file structure
for the jupyter notebook. So all the functions used from video are just recopied here.
"""


## CONSTANTS ##
# these colors are in RGBA
COLORS = {
        'clear': (255, 255, 255, 255),
        'red': (255, 0, 0, 255),
        'green': (0, 255, 0, 255),
        'blue': (0, 0, 255, 255),
        'yellow': (255, 255, 0, 255),
        'cyan': (0, 255, 255, 255),
        'magenta': (255, 0, 255, 255)
}

THICKNESS = 8 # Thickness of circles
RADIUS = 3 # Radius of circle

DISTANCE_FROM_EDGE = 5 # Distance from edge to exclude from the hull

BTN_X_START, BTN_Y_START = 10, 50 # Start and end positions of clear button

BUTTONS = {color: slice(10+50*i, 50+50*i) for i, color in enumerate(('clear', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta'))}

def equalizeHistogram(im, clahe=None):
    """
    This applies histogram qualization to the luminance of the given color image (which must be
    grayscale or RGB). The luminance is essentially the grayscale of the image, but this applies
    it to that and then adjusts the color image to match that luminance.

    If this is provided with a CLAHE object, it is used to perform the CLAHE instead of global
    histogram equalization.
    """
    if im.ndim == 3 and im.shape[2] >= 3:
        ycrcb = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)   # or color.rgb2ycbcr(im)
        if clahe is not None:
            ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        else:
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])  # or exposure.equalize_hist(im)
        im = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)   # or color.ycbcr2rgb(im)
    else:  # assume grayscale
        im = cv2.equalizeHist(im) if clahe is None else clahe.apply(im)
    return im


def image_in_range(im, colorized=True):
    """
    Returns a binary image where Trues (actually 255) in the result are when the HSV version is in the
    range of the global variables lower and upper.

    If colorized is True then the returned image has the pixels set to the color of the original image
    wherever the binary image is True.
    """
    he = equalizeHistogram(im, clahe)
    # Convert to HSV, openCV images are BGR
    hsvim = cv2.cvtColor(he, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsvim, lower1, upper1) > 0) | (cv2.inRange(hsvim, lower2, upper2) > 0)  # or ((lower1 <= im).all(2) & (im <= upper1).all(2)) | ((lower2 <= im).all(2) & (im <= upper2).all(2))
    return color_mask(im, mask) if colorized else mask

def color_mask(im, mask):
    """
    Takes a color image and a mask and returns a new image that is black outside the mask but colored
    in the mask.
    """
    return mask[:, :, None] * im

def reset_canvas():
    """
    Resets the canvas that holds the points to draw circles on and
    adds the color buttons.

    Returns:
        the canvas numpy array
    """
    canvas = np.zeros((360, 640, 4), 'uint8')
    for color, position in BUTTONS.items():
        canvas[BTN_X_START:BTN_Y_START, position] = COLORS[color]

    cv2.putText(canvas, "Clear", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0));
    return canvas
canvas = reset_canvas()

def centroid(max_contour):
    """
    Find the center of the given contour.

    Arguments:
        max_contour - an array for the largest contour

    Returns:
        the x and y of the center point
    """
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy

def is_point_near_edge(point, shape, dist):
    """
    Return whether point is within distance from edge.
    """
    return (shape[0] - dist < point[1] or shape[1] - dist < point[0] or 
            dist > point[1] or dist > point[0])

def determine_color(finger_tip):
    """
    Adjusts the drawing color based on finger location.
    Clears the canvas if the tip is on the "clear color."

    Arguments:
        finger_tip - the x and y coordinate of the finger tip
    """
    global canvas, chosen_color
    for color, position in BUTTONS.items():
        if BTN_X_START < finger_tip[1] < BTN_Y_START and position.start < finger_tip[0] < position.stop:
            if color == 'clear':
                canvas = reset_canvas()
            else:
                chosen_color = COLORS[color]

def finger_paint(im):
    """
    Finds the largest contour and the center of that contour.
    Then finds the convex hull and removes points near the edge from the hull.
    Finds the furthest point away from the center of the contour (the fingertip).
    Then checks what color to draw on the canvas and draws a circle at that point (or clears the canvas).
    Finally, adds the canvas to the image for a "painted" image.

    Arguments:
        im - an image

    Returns:
        The altered "painted" image
    """
    try:
        intial_segmentation = image_in_range(im, False)
        contours, _ = cv2.findContours(intial_segmentation.view('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        center_point = centroid(largest_contour)
        if center_point is not None:
            # cv2.drawMarker(im, center_point, (255,0,0))
            hull = cv2.convexHull(largest_contour).squeeze()

            # remove points from hull within x pixels of edges
            hull = np.array([point for point in hull if not is_point_near_edge(point, im.shape, DISTANCE_FROM_EDGE)])
            # cv2.drawContours(im, [hull], -1, (0, 0, 255), 2)

            finger_tip_point = hull[np.argmax(np.linalg.norm(hull-center_point, axis=1))]
            determine_color(finger_tip_point)

            # Draw the circle on the canvas
            cv2.circle(canvas, finger_tip_point, RADIUS, chosen_color, THICKNESS)

    except (cv2.error, ValueError, AttributeError, IndexError) as error:
        pass

    return (im + canvas[:, :, :3] * (canvas[:, :, 3:] / 255)).clip(0, 255).astype('uint8')


def run_video(process_frame=lambda im:im, fps=None, width=640, camera_num=0, return_orig=False):
    """
    TODO: This works okay in just a python script, but it could probably be extended to open a different window
    when used in a kernel. But I think that's where all the thread stuff comes from... I'll look into it eventually.
    

    Runs OpenCV video from a connected camera as Jupyter notebook output. Each frame from the camera
    is given to process_frame before being displayed. The default does no processing. The display is
    limited to the given number of frames per second (default is the camera's default, typically 25
    to 30). It can go below this, but will not go above it. If there is more than one camera
    connected, settings camera_num will select which camera to use.
    
    The video will continue being run until the code is interrupted with the stop button in Jupyter
    notebook.
    """
    # Open the video capture
    video_capture = cv2.VideoCapture(camera_num)
    try:
        if not video_capture.isOpened(): return  # if we did not successfully gain access to the camera

        # Setup the video capture
        if fps is not None: video_capture.set(cv2.CAP_PROP_FPS, fps)  # set the capturing FPS if provided
        w, h = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width is None or width == w:
            output_shape = h, w
        else:
            output_shape = (width * h // w, width)
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, output_shape[0])

        while True:
            # Try to get the first frame
            is_capturing, frame = video_capture.read()
            if frame is None: return # no first frame
            if frame.shape != output_shape:
                frame = cv2.resize(frame, output_shape[::-1])

            # Process the first frame and display it
            start = time()
            # Do not need to change to rgb, all functions used are opencv
            im = process_frame(frame)
            stop = time()

            fps = round(1/(stop-start))

            # The color for this text is in BGR
            im = cv2.putText(im, f"{fps}", (10, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))

            # Display the resulting frame
            cv2.imshow('image', im)

            # Press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        return frame if return_orig else im  # returns the final processed image or original frame
    finally:
        video_capture.release()
        cv2.destroyAllWindows()



def main():
    global lower1, upper1, lower2, upper2, clahe, canvas, chosen_color

    lower1 = np.array([0, 60, 48], np.uint8)
    upper1 = np.array([60, 255, 255], np.uint8)

    lower2 = np.array([135, 60, 100], np.uint8)
    upper2 = np.array([179, 255, 255], np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(21,21))

    # Color testing if needed. Might not be as useful here since we didn't adjust for the lack of widgets...
    # im = run_video(image_in_range, camera_num=1, return_orig=True)

    chosen_color = COLORS['cyan'] # Default color to cyan

    canvas = reset_canvas()

    # Run the painting
    # im = run_video(finger_paint, camera_num=1, return_orig=True)

    im = run_video_2(finger_paint, camera_num=1, return_orig=True)



if __name__=='__main__':
    main()