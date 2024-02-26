# By Abdullah As-Sadeed

import tempfile

import streamlit as sl
import PIL
import cv2
from ultralytics import YOLO

# Local modules
import settings

if __name__ == "__main__":
    TITLE = "Bitscoper CV WorkBench"
    ICON = "🔬"

    DETECT_MODEL = "Object Detection"
    SEGMENT_MODEL = "Object Segmentation"
    POSE_MODEL = "Pose Detection"
    MODELS = [DETECT_MODEL, SEGMENT_MODEL, POSE_MODEL]

    NANO_WEIGHT = "Nano"
    SMALL_WEIGHT = "Small"
    MEDIUM_WEIGHT = "Medium"
    LARGE_WEIGHT = "Large"
    EXTRA_LARGE_WEIGHT = "Extra Large"
    WEIGHTS = [
        NANO_WEIGHT,
        SMALL_WEIGHT,
        MEDIUM_WEIGHT,
        LARGE_WEIGHT,
        EXTRA_LARGE_WEIGHT,
    ]

    MINIMUM_CONFIDENCE = 25

    DEFAULT_CONFIDENCE = 40

    IMAGE_SOURCE = "Image"
    VIDEO_SOURCE = "Video"
    WEBCAM_SOURCE = "Webcam"
    RTSP_SOURCE = "RTSP"
    SOURCES = [IMAGE_SOURCE, VIDEO_SOURCE, WEBCAM_SOURCE, RTSP_SOURCE]

    IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp"]

    VIDEO_EXTENSIONS = ["mp4"]

    ASPECT_RATIO_16_9 = "16:9"
    ASPECT_RATIO_4_3 = "4:3"
    ASPECT_RATIO_CUSTOM = "Custom"
    ASPECT_RATIOS = [ASPECT_RATIO_16_9, ASPECT_RATIO_4_3, ASPECT_RATIO_CUSTOM]

    VIDEO_HEIGHTS = [144, 240, 360, 480, 720, 1080, 1440, 2160]

    DEFAULT_VIDEO_WIDTH = 640
    DEFAULT_VIDEO_HEIGHT = 480

    DEFAULT_WEBCAM_NUMBER = 0

    def select_video_size(width=None, height=None):
        combined_widths = []
        for aspect_ratio in ASPECT_RATIOS:
            if aspect_ratio == ASPECT_RATIO_16_9:
                aspect_ratio = 16 / 9
            elif aspect_ratio == ASPECT_RATIO_4_3:
                aspect_ratio = 4 / 3
            elif aspect_ratio == ASPECT_RATIO_CUSTOM:
                continue
            else:
                sl.error("Failed to determine aspect ratio!")
            for height in VIDEO_HEIGHTS:
                width = int(height * aspect_ratio)
                combined_widths.append(width)
        minimum_height = min(VIDEO_HEIGHTS)
        maximum_height = max(VIDEO_HEIGHTS)
        minimum_width = min(combined_widths)
        maximum_width = max(combined_widths)

        aspect_ratio = sl.sidebar.radio(
            "Select Aspect Ratio", ASPECT_RATIOS, horizontal=True
        )
        if (aspect_ratio == ASPECT_RATIO_16_9) or (aspect_ratio == ASPECT_RATIO_4_3):
            if aspect_ratio == ASPECT_RATIO_16_9:
                aspect_ratio = 16 / 9
            elif aspect_ratio == ASPECT_RATIO_4_3:
                aspect_ratio = 4 / 3
            else:
                sl.error("Failed to determine aspect ratio!")
            widths = []
            resolutions = []
            for height in VIDEO_HEIGHTS:
                width = int(height * aspect_ratio)
                widths.append(width)
                resolutions.append(f"{height}p: {width} x {height}")

            resolution = sl.sidebar.selectbox("Select Resolution", resolutions)

            width, height = map(int, resolution.split(":")[1].strip().split("x"))
        elif aspect_ratio == ASPECT_RATIO_CUSTOM:
            width = sl.sidebar.number_input(
                "Set Video Width",
                format="%d",
                min_value=minimum_width,
                max_value=maximum_width,
                step=4,
                value=DEFAULT_VIDEO_WIDTH,
            )
            height = sl.sidebar.number_input(
                "Set Video Height",
                format="%d",
                min_value=minimum_height,
                max_value=maximum_height,
                step=4,
                value=DEFAULT_VIDEO_HEIGHT,
            )
            width = int(width)
            height = int(height)
        else:
            sl.error("Failed to select aspect ratio!")
        return width, height

    def display_result_frames(streamlit_frame, source_frame, is_use_full_width):
        if tracker == "No":
            model_output = model(source_frame, conf=confidence)
        elif (tracker == "bytetrack.yaml") or (tracker == "botsort.yaml"):
            model_output = model.track(
                source_frame, conf=confidence, persist=True, tracker=tracker
            )
        else:
            sl.error("Failed to determine tracker!")

        plotted_frame = model_output[0].plot()
        streamlit_frame.image(
            plotted_frame,
            caption="Result Video",
            channels="BGR",
            use_column_width=is_use_full_width,
        )

    sl.set_page_config(
        page_title=TITLE,
        page_icon=ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    sl.title(TITLE)

    sl.sidebar.header("Model Settings")

    model_type = sl.sidebar.radio("Select Model", MODELS)

    model_weight = sl.sidebar.selectbox("Select Weight", WEIGHTS)

    confidence = (
        float(
            sl.sidebar.slider(
                "Set Model Confidence",
                min_value=MINIMUM_CONFIDENCE,
                max_value=100,
                value=DEFAULT_CONFIDENCE,
            )
        )
        / 100
    )

    tracker = sl.sidebar.selectbox(
        "Select Tracker", ["bytetrack.yaml", "botsort.yaml", "No"]
    )

    if model_type == DETECT_MODEL:
        model_suffix = ""
    elif model_type == SEGMENT_MODEL:
        model_suffix = "-seg"
    elif model_type == POSE_MODEL:
        model_suffix = "-pose"
    else:
        sl.error("Failed to select model!")

    if model_weight == NANO_WEIGHT:
        model_weight_suffix = "n"
    elif model_weight == SMALL_WEIGHT:
        model_weight_suffix = "s"
    elif model_weight == MEDIUM_WEIGHT:
        model_weight_suffix = "m"
    elif model_weight == LARGE_WEIGHT:
        model_weight_suffix = "l"
    elif model_weight == EXTRA_LARGE_WEIGHT:
        model_weight_suffix = "x"
    else:
        sl.error("Failed to select weight!")

    model_path = (
        str(settings.MODEL_DIRECTORY)
        + "/yolov8"
        + model_weight_suffix
        + model_suffix
        + ".pt"
    )

    try:
        model = YOLO(model_path)
    except Exception as exception:
        sl.error(f"Failed to load model.\nCheck the path: {model_path}: {exception}")

    sl.sidebar.header("Input Settings")

    source_radio = sl.sidebar.radio("Select Source", SOURCES)

    source_image = None

    if source_radio == IMAGE_SOURCE:
        source_image = sl.sidebar.file_uploader(
            "Select an Image File", type=IMAGE_EXTENSIONS, accept_multiple_files=False
        )

        column_1, column_2 = sl.columns(2)

        with column_1:
            try:
                if source_image is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    sl.image(
                        default_image_path,
                        caption="Default Image",
                        use_column_width=True,
                    )
                else:
                    uploaded_image = PIL.Image.open(source_image)
                    sl.image(
                        source_image, caption="Uploaded Image", use_column_width=True
                    )
            except Exception as exception:
                sl.error(f"Error occurred while opening the image: {exception}")
        with column_2:
            if source_image is None:
                default_result_image_path = str(settings.DEFAULT_RESULT_IMAGE)
                default_result_image = PIL.Image.open(default_result_image_path)
                sl.image(
                    default_result_image_path,
                    caption="Result Image",
                    use_column_width=True,
                )
            else:
                if sl.sidebar.button("Run"):
                    if tracker == "No":
                        resource = model(uploaded_image, conf=confidence)
                    else:
                        resource = model.track(
                            uploaded_image,
                            conf=confidence,
                            persist=True,
                            tracker=tracker,
                        )
                    boxes = resource[0].boxes
                    plotted_resource = resource[0].plot()[:, :, ::-1]
                    sl.image(
                        plotted_resource,
                        caption="Result Image",
                        use_column_width=True,
                    )
                    sl.snow()
                    try:
                        with sl.expander("Results"):
                            for box in boxes:
                                sl.write(box.data)
                    except Exception as exception:
                        sl.write("No image is uploaded yet!")
                        sl.write(exception)
    elif source_radio == VIDEO_SOURCE:
        source_video = sl.sidebar.file_uploader(
            "Select a Video File", type=VIDEO_EXTENSIONS, accept_multiple_files=False
        )
        if source_video is not None:
            temporary_file = tempfile.NamedTemporaryFile(delete=False)
            temporary_file.write(source_video.read())
            video_capture = cv2.VideoCapture(temporary_file.name)

            column_1, column_2 = sl.columns(2)

            with column_1:
                sl.video(source_video)
            with column_2:
                if sl.sidebar.button("Run"):
                    try:
                        st_frame = sl.empty()
                        while video_capture.isOpened():
                            success, image = video_capture.read()
                            if success:
                                display_result_frames(
                                    st_frame, image, is_use_full_width=True
                                )
                            else:
                                video_capture.release()
                                break
                    except Exception as exception:
                        sl.error(f"Error loading video: {exception}")
                    finally:
                        temporary_file.close()
    elif source_radio == WEBCAM_SOURCE:
        source_webcam = sl.sidebar.number_input(
            "Set Webcam Serial",
            format="%d",
            min_value=0,
            step=1,
            value=DEFAULT_WEBCAM_NUMBER,
        )
        source_webcam = int(source_webcam)

        source_width, source_height = select_video_size()

        if sl.sidebar.button("Run"):
            try:
                video_capture = cv2.VideoCapture(source_webcam)
                video_capture.set(3, source_width)
                video_capture.set(4, source_height)
                st_frame = sl.empty()
                while video_capture.isOpened():
                    success, image = video_capture.read()
                    if success:
                        display_result_frames(st_frame, image, is_use_full_width=False)
                    else:
                        video_capture.release()
                        break
            except Exception as exception:
                sl.error(f"Error loading video: {exception}")

    elif source_radio == RTSP_SOURCE:
        source_rtsp = sl.sidebar.text_input(
            "Set RTSP Stream URL", placeholder="Write a RTSP Stream URL"
        )

        if sl.sidebar.button("Run"):
            try:
                video_capture = cv2.VideoCapture(source_rtsp)
                st_frame = sl.empty()
                while video_capture.isOpened():
                    success, image = video_capture.read()
                    if success:
                        display_result_frames(st_frame, image, is_use_full_width=False)
                    else:
                        video_capture.release()
                        break
            except Exception as exception:
                sl.error(f"Error loading RTSP stream: {exception}")

    else:
        sl.error("Failed to select source!")
else:
    print("Run as\nstreamlit run main.py")
exit()
