import streamlit as st
import threading
from run_copy import run_curl_counter, run_squat_counter, run_situp_counter, run_shoulder_press_counter, run_pushup_counter

def start_workout(name, exercise, sets, reps, break_duration):
    sets = int(sets)
    reps = int(reps)
    break_duration = int(break_duration)
    if not name:
        st.error("Please enter your name.")
        return
    def run():
        if exercise == "Dumbbell Curls":
            run_curl_counter(name, sets, reps, break_duration)
        elif exercise == "Squats":
            run_squat_counter(name, sets, reps, break_duration)
        elif exercise == "Sit-ups":
            run_situp_counter(name, sets, reps, break_duration)
        elif exercise == "Shoulder Press":
            run_shoulder_press_counter(name, sets, reps, break_duration)
        elif exercise == "Push-ups":
            run_pushup_counter(name, sets, reps, break_duration)
    threading.Thread(target=run, daemon=True).start()
    st.success(f"Starting {exercise} for {name}. The camera window will open.")

st.title("🏋️ Exercise Counter (Camera will open on start)")
name = st.text_input("Enter your name")
exercise = st.selectbox(
    "Select exercise",
    ("Dumbbell Curls", "Squats", "Sit-ups", "Shoulder Press", "Push-ups")
)
sets = st.number_input("Number of sets", min_value=1, value=2)
reps = st.number_input("Reps per set", min_value=1, value=13)
break_duration = st.number_input("Break duration (seconds)", min_value=1, value=20)

if st.button("Start Workout"):
    start_workout(name, exercise, sets, reps, break_duration)
