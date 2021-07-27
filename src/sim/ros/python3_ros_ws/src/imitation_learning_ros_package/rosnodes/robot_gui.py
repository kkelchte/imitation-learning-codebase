from tkinter import Tk, Label, Canvas, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np

color_font = "#141E61"
color_bg = "#EEEEEE"
color_vel_0 = "#787A91"
color_vel_1 = "#0F044C"

window = Tk()
window.title("Robot Information")
window.geometry("500x800")
font = "-*-lucidatypewriter-medium-r-*-*-*-140-*-*-*-*-*-*"

# Write stats
frame = Frame(window)
frame.grid(row=0, column=0, sticky="n")
fsm = Label(
    frame, text="FSM state: Running", font=(font, 20), fg=color_font, bg=color_bg
).grid(column=0, row=0)
battery = Label(
    frame, text="Battery level: 80%", font=(font, 20), fg=color_font, bg=color_bg
).grid(column=0, row=1)

# Draw images
img_canvas = Canvas(window, width=480, height=240)
img_canvas.grid(row=1, column=0)
img_canvas.configure(bg=color_bg)
image = Image.fromarray(np.random.randint(0, 255, size=(200, 200, 3), dtype=np.uint8))
img = ImageTk.PhotoImage(image)
img_canvas.create_image(20, 20, anchor="nw", image=img)

mask = Image.fromarray(np.random.randint(0, 255, size=(200, 200, 3), dtype=np.uint8))
mask = ImageTk.PhotoImage(mask)
img_canvas.create_image(260, 20, anchor="nw", image=mask)

# Draw velocity commands
cmd_canvas = Canvas(window, width=480, height=260)
cmd_canvas.grid(row=2, column=0)
cmd_canvas.configure(bg=color_bg)

bar_width = 20
sec_bar_width = 15
bar_length = 200
max_vel = 1

# x velocity
x_vel = -0.5
cmd_canvas.create_rectangle(
    120 - int(bar_width / 2),
    20,
    120 + int(bar_width / 2),
    20 + bar_length,
    fill=color_vel_0,
)
cmd_canvas.create_rectangle(
    120 - int(bar_width / 2),
    20,
    120 + int(bar_width / 2),
    20 + bar_length // 2,
    fill=color_vel_0,
)
cmd_canvas.create_rectangle(
    120 - sec_bar_width // 2,
    20 + bar_length // 2,
    120 + sec_bar_width // 2,
    20 + (bar_length // 2) - x_vel * (bar_length // 2),
    fill=color_vel_1,
)
cmd_canvas.create_text(120, 10, text=f"x: {x_vel}", font=(font, 20))

# y velocity
y_vel = -0.75
cmd_canvas.create_rectangle(
    20, 235 - bar_width // 2, 20 + bar_length, 235 + bar_width // 2, fill=color_vel_0,
)
cmd_canvas.create_rectangle(
    20,
    235 - bar_width // 2,
    20 + bar_length // 2,
    235 + bar_width // 2,
    fill=color_vel_0,
)
cmd_canvas.create_rectangle(
    20 + bar_length // 2,
    235 - sec_bar_width // 2,
    20 + (bar_length // 2) + y_vel * (bar_length // 2),
    235 + sec_bar_width // 2,
    fill=color_vel_1,
)
cmd_canvas.create_text(
    10, 205, text=f"y: {y_vel}", anchor="nw", font=(font, 20), fill=color_font
)

# z velocity
z_vel = -0.01
n_z_vel = z_vel / max_vel
cmd_canvas.create_rectangle(
    350 - bar_width // 2, 20, 350 + bar_width // 2, 20 + bar_length, fill=color_vel_0,
)
cmd_canvas.create_rectangle(
    350 - bar_width // 2,
    20,
    350 + bar_width // 2,
    20 + bar_length // 2,
    fill=color_vel_0,
)
cmd_canvas.create_rectangle(
    350 - sec_bar_width // 2,
    20 + bar_length // 2,
    350 + sec_bar_width // 2,
    20 + (bar_length // 2) - int(n_z_vel * (bar_length // 2)),
    fill=color_vel_1,
)
cmd_canvas.create_text(350, 10, text=f"z: {z_vel}", font=(font, 20))

# yaw velocity
yaw_vel = 0.3
cmd_canvas.create_rectangle(
    250,
    235 - int(bar_width / 2),
    250 + bar_length,
    235 + int(bar_width / 2),
    fill=color_vel_0,
)
cmd_canvas.create_rectangle(
    250,
    235 - int(bar_width / 2),
    250 + bar_length // 2,
    235 + int(bar_width / 2),
    fill=color_vel_0,
)
cmd_canvas.create_rectangle(
    250 + bar_length // 2,
    235 - sec_bar_width // 2,
    250 + (bar_length // 2) + yaw_vel * (bar_length // 2),
    235 + sec_bar_width // 2,
    fill=color_vel_1,
)
cmd_canvas.create_text(
    240, 205, text="\u03c8" + str(yaw_vel), anchor="nw", font=(font, 20)
)

# Draw waypoints
reference_point = [1, -2, 0.3]

wp_canvas = Canvas(window, width=480, height=260)
wp_canvas.grid(row=3, column=0)
wp_canvas.configure(bg=color_bg)

fig, ax = plt.subplots(1, 2, figsize=(9, 4), dpi=50)
# Reference point
ax[0].annotate(
    "",
    xy=(reference_point[1], reference_point[0]),
    xytext=(0, 0),
    arrowprops=dict(arrowstyle="->"),
)
ax[0].set_xlim(-4, 4)
ax[0].set_ylim(0, 4)
ax[0].set_title("Reference Point")

# Trajectory
trajectory = [[1, 0], [1.2, 0.5]]
ax[1].scatter([p[0] for p in trajectory], [p[1] for p in trajectory])
ax[1].set_title("Trajectory")

chart_type = FigureCanvasTkAgg(fig, wp_canvas)
chart_type.get_tk_widget().pack()
plt.tight_layout()

window.mainloop()
