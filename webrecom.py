import streamlit as st
import streamlit_webrtc
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

model = load_model("model.h5")
label = np.load("labels.npy")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Music Recommender using facial recognition")

if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not (emotion):
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"


class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]

            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1,
                                                                         circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


if st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

btn = st.button("Recommend me songs")


def happy():
    webbrowser.open(f"https://www.youtube.com/watch?v=ZbZSe6N_BXs&list=PLplXQ2cg9B_qrNvF8KaDew3EetUqO8jBo")
    np.save("emotion.npy", np.array([""]))
    st.session_state["run"] = "false"


def greeting():
    webbrowser.open(f"https://www.youtube.com/watch?v=kVi8ICWu3WI")
    np.save("emotion.npy", np.array([""]))
    st.session_state["run"] = "false"


def rock():
    webbrowser.open(f"https://www.youtube.com/watch?v=tAGnKpE4NCI&list=PLZN_exA7d4RVmCQrG5VlWIjMOkMFZVVOc")
    np.save("emotion.npy", np.array([""]))
    st.session_state["run"] = "false"


def suprise():
    webbrowser.open(f"https://www.youtube.com/watch?v=RBumgq5yVrA&list=PLLd27tZalu3zRpolGDrklbbS1T-L5Lc7g")
    np.save("emotion.npy", np.array([""]))
    st.session_state["run"] = "false"


def neutral():
    webbrowser.open(f"https://www.youtube.com/watch?v=u6wOyMUs74I&list=PLgzTt0k8mXzEpH7-dOCHqRZOsakqXmzmG")
    np.save("emotion.npy", np.array([""]))
    st.session_state["run"] = "false"


def default_function():
    st.warning("Please let me capture your emotion first")


functions = {
    "happy": happy,
    "greeting": greeting,
    "rock": rock,
    "neutral": neutral,
    "suprise": suprise,
}

if btn:
    if not (emotion):

        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        key1 = emotion
        functions.get(key1, default_function)()
