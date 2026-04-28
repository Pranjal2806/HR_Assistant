# video_module.py

import streamlit as st
from streamlit_webrtc import webrtc_streamer
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
    # HR SECTION
    # ==========================
    if role == "HR":

        st.subheader("Create Interview Room")

        room_name = st.text_input("Room Name")

        room_password = st.text_input(
            "Create Password",
            type="password"
        )

        if st.button("Create Room"):

            if room_name and room_password:

                room_id = str(uuid.uuid4())[:8]

                rooms = load_rooms()

                rooms[room_id] = {
                    "room_name": room_name,
                    "password": room_password
                }

                save_rooms(rooms)

                st.success("Interview room created successfully.")

                st.write(f"Room ID: {room_id}")
                st.write(f"Password: {room_password}")

                st.info(
                    "Share the Room ID and Password with candidate."
                )

            else:
                st.error("Please enter all details.")

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

            # Check Room
            if room_id in rooms:

                saved_password = rooms[room_id]["password"]

                # Validate Password
                if password == saved_password:

                    st.success("Access granted.")

                    st.write(f"Connected to Room: {room_id}")

                    # Start WebRTC
                    webrtc_streamer(
                        key=room_id,
                        media_stream_constraints={
                            "video": True,
                            "audio": True
                        }
                    )

                else:
                    st.error("Incorrect password.")

            else:
                st.error("Room not found.")