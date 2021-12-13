from io import BytesIO
from time import time
from threading import Thread
from queue import Queue, Full

import ipywidgets as widgets
from IPython.display import display

import PIL.Image
import PIL.ImageDraw

import cv2


def bgr2rgb(im):
    """
    Converts image from BGR (blue, green, red) to RGB. OpenCV use BGR instead of RGB in some cases,
    however RGB is the standard for matplotlib.

    This will also convert RGB to BGR.
    """
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def __image_encode(im, fps=""):
    """
    Takes a numpy array and encodes it into a PNG.
    This will automatically deal with floating-point images.
    Adds the FPS string to the top-left corner of the image.
    """
    if im.dtype.kind == 'f':
        im = im * 255
        im = im.clip(0, 255, im).astype('uint8')
    #return cv2.imencode('.png', bgr2rgb(im))[1].tostring()
    im_out = PIL.Image.fromarray(im)
    if fps:
        PIL.ImageDraw.Draw(im_out).text((3, 3), str(fps), 255)
    f = BytesIO()
    im_out.save(f, format='png')
    return f.getvalue()

def run_video(process_frame=lambda im:im, fps=None, width=640, camera_num=0, return_orig=False):
    """
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

        # Try to get the first frame
        is_capturing, frame = video_capture.read()
        if frame is None: return # no first frame
        if frame.shape != output_shape:
           frame = cv2.resize(frame, output_shape[::-1])

        # Process the first frame and display it
        im = process_frame(bgr2rgb(frame))
        image = widgets.Image()
        image.value = __image_encode(im)
        display(image, display_id=True)

        # Start video capturing thread
        queue = Queue(1)
        __capture_frames_thread.is_processing = True
        thread = Thread(target=__capture_frames_thread, args=(video_capture, queue, output_shape), daemon=True)
        thread.start()

        while thread.is_alive() and __capture_frames_thread.is_processing:
            # Keep getting new frames while they are available
            try:
                # Get the next image
                frame = queue.get()
                if frame is None: break # no next frame

                # Process the frame
                start = time()  # start time for computing FPS
                im = process_frame(frame)
                stop = time()

                # Update the display
                image.value = __image_encode(im, round(1/(stop-start), 1))
            except KeyboardInterrupt: break  # watch for a keyboard interrupt (stop button) to stop the script gracefully
        __capture_frames_thread.is_processing = False
        return frame if return_orig else im  # returns the final processed image or original frame
    finally:
        __capture_frames_thread.is_processing = False
        video_capture.release()


def __capture_frames_thread(video_capture, queue, output_shape):
    is_capturing = True
    frame = None
    reversed_output_shape = output_shape[::-1]
    try:
        # Keep getting new frames while they are available and we haven't been interruppted
        while is_capturing and __capture_frames_thread.is_processing:
            # Get the next frame
            is_capturing, frame = video_capture.read(frame)
            if frame is None: break # no next frame
            if frame.shape != output_shape:
               frame = cv2.resize(frame, reversed_output_shape)
            queue.put(bgr2rgb(frame))
    finally:
        __capture_frames_thread.is_processing = False
        try:
            queue.put(None, timeout=1)
        except Full:
            pass

def run_video_2(process_frame=lambda im:im, fps=None, width=640, camera_num=0, return_orig=False):
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
