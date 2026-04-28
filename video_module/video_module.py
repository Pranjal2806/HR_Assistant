# video_module.py

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import uuid
import json
import os


ROOM_FILE = "rooms.json"


# ==========================
# Load Rooms
# ==========================
def load_rooms():

    if not os.path.exists(ROOM_FILE):
        return {}

    with open(ROOM_FILE, "r") as f:
        return json.load(f)


# ==========================
# Save Rooms
# ==========================
def save_rooms(data):

    with open(ROOM_FILE, "w") as f:
        json.dump(data, f)


# ==========================
# Main Function
# ==========================
def run_video_interview():

    st.title("Live Interview System")

    role = st.radio(
        "Select Role",
        ["HR", "Candidate"]
    )

    st.divider()

    # ==========================
    # Initialize Session State
    # ==========================
    if "current_room_id" not in st.session_state:
        st.session_state["current_room_id"] = None

    # ==========================
    # HR SECTION
    # ==========================
    if role == "HR":

        st.subheader("Create Interview Room")

        room_name = st.text_input("Room Name")

        room_password = st.text_input(
            "Create Password",
            type="password"
        )

        # ==========================
        # Create Room
        # ==========================
        if st.button("Create Room"):

            if room_name and room_password:

                room_id = str(uuid.uuid4())[:8]

                rooms = load_rooms()

                rooms[room_id] = {
                    "room_name": room_name,
                    "password": room_password
                }

                save_rooms(rooms)

                # Save room id in session
                st.session_state["current_room_id"] = room_id

                st.success("Interview room created successfully.")

            else:
                st.error("Please enter all details.")

        # ==========================
        # Display Room Details
        # ==========================
        current_room_id = st.session_state.get("current_room_id")

        if current_room_id:

            rooms = load_rooms()

            if current_room_id in rooms:

                st.write(f"Room ID: {current_room_id}")

                st.write(
                    f"Password: {rooms[current_room_id]['password']}"
                )

                st.info(
                    "Share the Room ID and Password with candidate."
                )

                st.subheader("Start Interview")

                # ==========================
                # Start HR WebRTC
                # ==========================
                webrtc_streamer(
                    key=current_room_id,
                    mode=WebRtcMode.SENDRECV,
                    media_stream_constraints={
                        "video": True,
                        "audio": True
                    },
                    async_processing=True
                )

    # ==========================
    # CANDIDATE SECTION
    # ==========================
    elif role == "Candidate":

        st.subheader("Join Interview")

        room_id = st.text_input("Enter Room ID")

        password = st.text_input(
            "Enter Password",
            type="password"
        )

        if room_id and password:

            rooms = load_rooms()

            # ==========================
            # Check Room
            # ==========================
            if room_id in rooms:

                saved_password = rooms[room_id]["password"]

                # ==========================
                # Validate Password
                # ==========================
                if password == saved_password:

                    st.success("Access granted.")

                    st.write(f"Connected to Room: {room_id}")

                    # ==========================
                    # Start Candidate WebRTC
                    # ==========================
                    webrtc_streamer(
                        key=room_id,
                        mode=WebRtcMode.SENDRECV,
                        media_stream_constraints={
                            "video": True,
                            "audio": True
                        },
                        async_processing=True
                    )

                else:
                    st.error("Incorrect password.")

            else:
                st.error("Room not found.")