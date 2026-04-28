# video_module.py

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import uuid


# Store active rooms temporarily
if "active_rooms" not in st.session_state:
    st.session_state.active_rooms = {}


def run_video_interview():

    st.title("Live Interview System")

    role = st.radio(
        "Select Role",
        ["HR", "Candidate"]
    )

    st.divider()

    # ==========================
    # HR Section
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

                # Save room information
                st.session_state.active_rooms[room_id] = {
                    "room_name": room_name,
                    "password": room_password
                }

                st.success("Interview room created successfully.")

                st.write(f"Room ID: {room_id}")
                st.write(f"Password: {room_password}")

                st.info(
                    "Share the Room ID and Password with the candidate."
                )

            else:
                st.error("Please enter all details.")

    # ==========================
    # Candidate Section
    # ==========================
    elif role == "Candidate":

        st.subheader("Join Interview")

        room_id = st.text_input("Enter Room ID")

        password = st.text_input(
            "Enter Password",
            type="password"
        )

        if st.button("Join Room"):

            # Check whether room exists
            if room_id in st.session_state.active_rooms:

                saved_password = st.session_state.active_rooms[room_id]["password"]

                # Validate password
                if password == saved_password:

                    st.success("Access granted.")

                    st.write(f"Connected to Room: {room_id}")

                    # Start video call
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