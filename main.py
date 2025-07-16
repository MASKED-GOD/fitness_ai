import tkinter as tk
from tkinter import ttk, messagebox
from run_copy import run_curl_counter, run_squat_counter, run_situp_counter, run_shoulder_press_counter, run_pushup_counter
import threading

def start_exercise():
    name = name_var.get().strip()
    exercise = exercise_var.get()
    try:
        sets = int(sets_var.get())
        reps = int(reps_var.get())
        break_duration = int(break_var.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for sets, reps, and break duration.")
        return
    if not name:
        messagebox.showerror("Input Error", "Please enter your name.")
        return
    if exercise == "Dumbbell Curls":
        threading.Thread(target=run_curl_counter, args=(name, sets, reps, break_duration), daemon=True).start()
    elif exercise == "Squats":
        threading.Thread(target=run_squat_counter, args=(name, sets, reps, break_duration), daemon=True).start()
    elif exercise == "Sit-ups":
        threading.Thread(target=run_situp_counter, args=(name, sets, reps, break_duration), daemon=True).start()
    elif exercise == "Shoulder Press":
        threading.Thread(target=run_shoulder_press_counter, args=(name, sets, reps, break_duration), daemon=True).start()
    elif exercise == "Push-ups":
        threading.Thread(target=run_pushup_counter, args=(name, sets, reps, break_duration), daemon=True).start()
    else:
        messagebox.showinfo("Coming Soon", "Other exercises coming soon!")

root = tk.Tk()
root.title("Exercise Counter")
root.geometry("350x350")

# Name Entry
name_label = tk.Label(root, text="Enter your name:")
name_label.pack(pady=(20, 5))
name_var = tk.StringVar()
name_entry = tk.Entry(root, textvariable=name_var, width=30)
name_entry.pack()

# Exercise Dropdown
exercise_label = tk.Label(root, text="Select exercise:")
exercise_label.pack(pady=(15, 5))
exercise_var = tk.StringVar(value="Dumbbell Curls")
exercise_dropdown = ttk.Combobox(root, textvariable=exercise_var, state="readonly", width=27)
exercise_dropdown['values'] = ("Dumbbell Curls", "Squats", "Sit-ups", "Shoulder Press", "Push-ups", "Other")
exercise_dropdown.pack()

# Sets Entry
sets_label = tk.Label(root, text="Number of sets:")
sets_label.pack(pady=(15, 5))
sets_var = tk.StringVar(value="2")
sets_entry = tk.Entry(root, textvariable=sets_var, width=30)
sets_entry.pack()

# Reps Entry
reps_label = tk.Label(root, text="Reps per set:")
reps_label.pack(pady=(15, 5))
reps_var = tk.StringVar(value="13")
reps_entry = tk.Entry(root, textvariable=reps_var, width=30)
reps_entry.pack()

# Break Duration Entry
break_label = tk.Label(root, text="Break duration (seconds):")
break_label.pack(pady=(15, 5))
break_var = tk.StringVar(value="20")
break_entry = tk.Entry(root, textvariable=break_var, width=30)
break_entry.pack()

# Start Button
start_button = tk.Button(root, text="Start", command=start_exercise, width=15)
start_button.pack(pady=20)

root.mainloop() 