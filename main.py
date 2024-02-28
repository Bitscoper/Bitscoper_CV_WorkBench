# By Abdullah As-Sadeed

import tempfile
import streamlit as sl
import numpy as np
import mediapipe as mp
import cv2
import PIL
from ultralytics import YOLO

# Local modules
import Default_Paths
import Default_Settings

if __name__ == "__main__":
    TITLE = "Bitscoper CV WorkBench"

    MEDIAPIPE_LIBRARY = "MediaPipe"
    YOLO_LIBRARY = "Ultralytics YOLOv8"
    LIBRRARIES = [YOLO_LIBRARY, MEDIAPIPE_LIBRARY]

    YOLO_DETECT_MODEL = "Object Detection"
    YOLO_SEGMENT_MODEL = "Object Segmentation"
    YOLO_POSE_MODEL = "Pose Detection"
    YOLO_MODELS = [YOLO_DETECT_MODEL, YOLO_SEGMENT_MODEL, YOLO_POSE_MODEL]

    YOLO_NANO_MODEL_WEIGHT = "Nano"
    YOLO_SMALL_MODEL_WEIGHT = "Small"
    YOLO_MEDIUM_MODEL_WEIGHT = "Medium"
    YOLO_LARGE_MODEL_WEIGHT = "Large"
    YOLO_EXTRA_LARGE_MODEL_WEIGHT = "Extra Large"
    YOLO_MODEL_WEIGHTS = [
        YOLO_NANO_MODEL_WEIGHT,
        YOLO_SMALL_MODEL_WEIGHT,
        YOLO_MEDIUM_MODEL_WEIGHT,
        YOLO_LARGE_MODEL_WEIGHT,
        YOLO_EXTRA_LARGE_MODEL_WEIGHT,
    ]

    MEDIAPIPE_MODELS = {
        "hands": mp.solutions.hands.Hands,
    }

    IMAGE_FILE_SOURCE = "Image File"
    VIDEO_FILE_SOURCE = "Video File"
    WEBCAM_SOURCE = "Webcam"
    RTSP_SOURCE = "RTSP"
    SOURCES = [IMAGE_FILE_SOURCE, VIDEO_FILE_SOURCE, WEBCAM_SOURCE, RTSP_SOURCE]

    FLIP_HORIZONTALLY = "Horizontally"
    FLIP_VERTICALLY = "Vertically"
    FLIP_BOTH_AXIS = "Both Axis"
    FLIP_NONE = "No"
    FLIP = [FLIP_NONE, FLIP_HORIZONTALLY, FLIP_VERTICALLY, FLIP_BOTH_AXIS]

    IMAGE_FILE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp"]
    VIDEO_FILE_EXTENSIONS = ["mp4"]

    ASPECT_RATIO_16_9 = "16:9"
    ASPECT_RATIO_4_3 = "4:3"
    ASPECT_RATIO_CUSTOM = "Custom"
    ASPECT_RATIOS = [ASPECT_RATIO_16_9, ASPECT_RATIO_4_3, ASPECT_RATIO_CUSTOM]

    VIDEO_HEIGHTS = [144, 240, 360, 480, 720, 1080, 1440, 2160]

    def Select_Video_Resolution(width=None, height=None):
        combined_widths = []

        for aspect_ratio in ASPECT_RATIOS:
            if aspect_ratio == ASPECT_RATIO_16_9:
                aspect_ratio = 16 / 9

            elif aspect_ratio == ASPECT_RATIO_4_3:
                aspect_ratio = 4 / 3

            elif aspect_ratio == ASPECT_RATIO_CUSTOM:
                continue

            else:
                sl.error("Error determining aspect ratio!")

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
                sl.error("Error determining aspect ratio!")

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
                value=Default_Settings.DEFAULT_VIDEO_WIDTH,
            )

            height = sl.sidebar.number_input(
                "Set Video Height",
                format="%d",
                min_value=minimum_height,
                max_value=maximum_height,
                step=4,
                value=Default_Settings.DEFAULT_VIDEO_HEIGHT,
            )

            width = int(width)
            height = int(height)

        else:
            sl.error("Error determining aspect ratio!")

        return width, height

    def Convert_Hex_Color_to_BGR_Tuple(hex_color):
        hex_color = hex_color.lstrip("#")
        BGR_Tuple = tuple(int(hex_color[i : i + 2], 16) for i in (4, 2, 0))

        return BGR_Tuple

    def Process_and_Display_Frame(streamlit_frame, source_frame):
        if flip_code is not None:
            source_frame = cv2.flip(source_frame, flip_code)

        if library == YOLO_LIBRARY:
            if YOLO_tracker == "None":
                YOLO_output = YOLO_model(source_frame, conf=YOLO_confidence)

            elif (YOLO_tracker == "bytetrack.yaml") or (YOLO_tracker == "botsort.yaml"):
                YOLO_output = YOLO_model.track(
                    source_frame,
                    conf=YOLO_confidence,
                    persist=True,
                    tracker=YOLO_tracker,
                )

            else:
                sl.error("Error determining tracker!")

            frame = YOLO_output[0].plot()

            streamlit_frame.image(frame, caption="Result", channels="BGR")

        elif library == MEDIAPIPE_LIBRARY:
            mp_drawing = mp.solutions.drawing_utils

            with MediaPipe_model(
                min_detection_confidence=MediaPipe_detection_confidence,
                min_tracking_confidence=MediaPipe_tracking_confidence,
            ) as model:
                frame = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False

                results = model.process(frame)

                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if MediaPipe_model_type == "hands" and results.multi_hand_landmarks:
                    mp_hands = mp.solutions.hands

                    for num, hand in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(
                            frame,
                            hand,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(
                                color=MediaPipe_connection_color,
                                thickness=MediaPipe_connection_thickness,
                                circle_radius=MediaPipe_connection_radius,
                            ),
                            mp_drawing.DrawingSpec(
                                color=MediaPipe_landmark_color,
                                thickness=MediaPipe_landmark_thickness,
                            ),
                        )

                streamlit_frame.image(frame, caption="Result", channels="BGR")

    sl.set_page_config(
        page_title=TITLE,
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # sl.title(TITLE)
    sl.sidebar.header(TITLE)

    library = sl.sidebar.selectbox("Select Library", LIBRRARIES)

    if library == YOLO_LIBRARY:
        YOLO_model_type = sl.sidebar.radio("Select Model", YOLO_MODELS)

        YOLO_model_weight = sl.sidebar.selectbox(
            "Select Model Weight", YOLO_MODEL_WEIGHTS
        )

        if YOLO_model_type == YOLO_DETECT_MODEL:
            YOLO_model_suffix = ""

        elif YOLO_model_type == YOLO_SEGMENT_MODEL:
            YOLO_model_suffix = "-seg"

        elif YOLO_model_type == YOLO_POSE_MODEL:
            YOLO_model_suffix = "-pose"

        else:
            sl.error("Error determining model!")

        if YOLO_model_weight == YOLO_NANO_MODEL_WEIGHT:
            YOLO_model_weight_suffix = "n"

        elif YOLO_model_weight == YOLO_SMALL_MODEL_WEIGHT:
            YOLO_model_weight_suffix = "s"

        elif YOLO_model_weight == YOLO_MEDIUM_MODEL_WEIGHT:
            YOLO_model_weight_suffix = "m"

        elif YOLO_model_weight == YOLO_LARGE_MODEL_WEIGHT:
            YOLO_model_weight_suffix = "l"

        elif YOLO_model_weight == YOLO_EXTRA_LARGE_MODEL_WEIGHT:
            YOLO_model_weight_suffix = "x"

        else:
            sl.error("Error determining model weight!")

        YOLO_model_path = (
            str(Default_Paths.YOLO_DEFAULT_MODEL_DIRECTORY)
            + "/yolov8"
            + YOLO_model_weight_suffix
            + YOLO_model_suffix
            + ".pt"
        )

        try:
            YOLO_model = YOLO(YOLO_model_path)

        except Exception as exception:
            sl.error(
                f"Error loading model.\nCheck the path: {YOLO_model_path}: {exception}"
            )

        YOLO_tracker = sl.sidebar.selectbox(
            "Select Tracker", ["bytetrack.yaml", "botsort.yaml", "None"]
        )

        YOLO_confidence = (
            float(
                sl.sidebar.slider(
                    "Set Confidence",
                    min_value=0,
                    max_value=100,
                    value=Default_Settings.YOLO_DEFAULT_CONFIDENCE,
                )
            )
            / 100
        )

    elif library == MEDIAPIPE_LIBRARY:
        MediaPipe_model_type = sl.sidebar.radio(
            "Select Model", list(MEDIAPIPE_MODELS.keys())
        )
        MediaPipe_model = MEDIAPIPE_MODELS[MediaPipe_model_type]

        MediaPipe_detection_confidence = (
            float(
                sl.sidebar.slider(
                    "Set Detection Confidence",
                    min_value=0,
                    max_value=100,
                    value=Default_Settings.MEDIAPIPE_DEFAULT_DETECTION_CONFIDENCE,
                )
            )
            / 100
        )

        MediaPipe_tracking_confidence = (
            float(
                sl.sidebar.slider(
                    "Set Tracking Confidence",
                    min_value=0,
                    max_value=100,
                    value=Default_Settings.MEDIAPIPE_DEFAULT_TRACKING_CONFIDENCE,
                )
            )
            / 100
        )

        MediaPipe_connection_color = Convert_Hex_Color_to_BGR_Tuple(
            sl.sidebar.color_picker(
                "Pick Connection Color",
                Default_Settings.MEDIAPIPE_DEFAULT_CONNECTION_COLOR,
            )
        )

        MediaPipe_connection_thickness = int(
            sl.sidebar.number_input(
                "Set Connection Thickness",
                format="%d",
                min_value=0,
                step=1,
                value=Default_Settings.MEDIAPIPE_DEFAULT_CONNECTION_THICKNESS,
            )
        )

        MediaPipe_connection_radius = int(
            sl.sidebar.number_input(
                "Set Landmark Radius",
                format="%d",
                min_value=0,
                step=1,
                value=Default_Settings.MEDIAPIPE_DEFAULT_CONNECTION_RADIUS,
            )
        )

        MediaPipe_landmark_color = Convert_Hex_Color_to_BGR_Tuple(
            sl.sidebar.color_picker(
                "Pick Landmark Color", Default_Settings.MEDIAPIPE_DEFAULT_LANDMARK_COLOR
            )
        )

        MediaPipe_landmark_thickness = int(
            sl.sidebar.number_input(
                "Set Landmark Thickness",
                format="%d",
                min_value=0,
                step=1,
                value=Default_Settings.MEDIAPIPE_DEFAULT_LANDMARK_THICKNESS,
            )
        )

    else:
        sl.error("Error determining library!")

    source_selection = sl.sidebar.radio("Select Source", SOURCES)

    if source_selection == WEBCAM_SOURCE:
        flip_horizontally_index = FLIP.index(FLIP_HORIZONTALLY)
        FLIP = (
            [FLIP[flip_horizontally_index]]
            + FLIP[:flip_horizontally_index]
            + FLIP[flip_horizontally_index + 1 :]
        )  # Move "Horizontally" to the top

    flip = sl.sidebar.selectbox("Flip Source", FLIP)

    if flip == FLIP_HORIZONTALLY:
        flip_code = 1

    elif flip == FLIP_VERTICALLY:
        flip_code = 0

    elif flip == FLIP_BOTH_AXIS:
        flip_code = -1

    elif flip == FLIP_NONE:
        flip_code = None

    else:
        sl.error("Error determining flip!")

    if source_selection == IMAGE_FILE_SOURCE:
        source_image = sl.sidebar.file_uploader(
            "Select an Image File",
            type=IMAGE_FILE_EXTENSIONS,
            accept_multiple_files=False,
        )

        if source_image is not None and sl.sidebar.button("Run"):
            column_1, column_2 = sl.columns(2)

            with column_1:
                uploaded_image = PIL.Image.open(source_image)

                sl.image(source_image, caption="Source")

            with column_2:
                streamlit_frame = sl.empty()

                frame = np.array(uploaded_image)
                frame = frame[:, :, ::-1].copy()

                Process_and_Display_Frame(streamlit_frame, frame)

    elif source_selection == VIDEO_FILE_SOURCE:
        source_video = sl.sidebar.file_uploader(
            "Select a Video File",
            type=VIDEO_FILE_EXTENSIONS,
            accept_multiple_files=False,
        )

        if source_video is not None:
            temporary_file = tempfile.NamedTemporaryFile(delete=False)
            temporary_file.write(source_video.read())

            column_1, column_2 = sl.columns(2)

            with column_1:
                sl.video(source_video)

            with column_2:
                if sl.sidebar.button("Run"):
                    try:
                        video_capture = cv2.VideoCapture(temporary_file.name)

                        streamlit_frame = sl.empty()

                        while video_capture.isOpened():
                            success, frame = video_capture.read()

                            if success:
                                Process_and_Display_Frame(streamlit_frame, frame)

                            else:
                                video_capture.release()
                                break

                    except Exception as exception:
                        sl.error(f"Error loading video: {exception}")

                    finally:
                        temporary_file.close()

    elif source_selection == WEBCAM_SOURCE:
        source_webcam_ID = int(
            sl.sidebar.number_input(
                "Enter Webcam ID",
                format="%d",
                min_value=0,
                step=1,
                value=Default_Settings.DEFAULT_WEBCAM_ID,
            )
        )

        source_width, source_height = Select_Video_Resolution()

        if source_webcam_ID is not None and sl.sidebar.button("Run"):
            try:
                video_capture = cv2.VideoCapture(source_webcam_ID)
                video_capture.set(3, source_width)
                video_capture.set(4, source_height)

                streamlit_frame = sl.empty()

                while video_capture.isOpened():
                    success, frame = video_capture.read()

                    if success:
                        Process_and_Display_Frame(streamlit_frame, frame)

                    else:
                        video_capture.release()
                        break

            except Exception as exception:
                sl.error(f"Error loading webcam stream: {exception}")

    elif source_selection == RTSP_SOURCE:
        source_RTSP_URL = sl.sidebar.text_input(
            "Set RTSP URL", placeholder="Paste a RTSP URL"
        )

        if source_RTSP_URL is not None and sl.sidebar.button("Run"):
            try:
                video_capture = cv2.VideoCapture(source_RTSP_URL)

                streamlit_frame = sl.empty()

                while video_capture.isOpened():
                    success, frame = video_capture.read()

                    if success:
                        Process_and_Display_Frame(streamlit_frame, frame)

                    else:
                        video_capture.release()
                        break

            except Exception as exception:
                sl.error(f"Error loading RTSP stream: {exception}")

    else:
        sl.error("Error determining source!")

else:
    print("Run as\npython -m streamlit run main.py")

exit()
