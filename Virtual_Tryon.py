import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()

person_path = askopenfilename(
    title="Select Person Image"
)

shirt_path = askopenfilename(
    title="Select Shirt PNG"
)

person = cv2.imread(person_path)
shirt = cv2.imread(
    shirt_path,
    cv2.IMREAD_UNCHANGED
)

person = cv2.cvtColor(
    person,
    cv2.COLOR_BGR2RGB
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

result = pose.process(person)

output = person.copy()

if result.pose_landmarks:

    h, w = person.shape[:2]

    left = result.pose_landmarks.landmark[11]
    right = result.pose_landmarks.landmark[12]

    lx = int(left.x * w)
    ly = int(left.y * h)

    rx = int(right.x * w)
    ry = int(right.y * h)

    shirt_w = abs(rx - lx) + 120
    shirt_h = int(shirt_w * 1.7)

    shirt = cv2.resize(
        shirt,
        (shirt_w, shirt_h)
    )

    x = min(lx, rx) - 60
    y = ly - 30

    x = max(0, x)
    y = max(0, y)

    shirt = shirt[
        :min(shirt_h, h-y),
        :min(shirt_w, w-x)
    ]

    alpha = shirt[:,:,3]/255

    for c in range(3):

        output[
            y:y+shirt.shape[0],
            x:x+shirt.shape[1],
            c
        ] = (
            alpha*shirt[:,:,c]
            +
            (1-alpha)*
            output[
                y:y+shirt.shape[0],
                x:x+shirt.shape[1],
                c
            ]
        )

plt.figure(
    figsize=(12,7)
)

plt.subplot(1,2,1)
plt.imshow(person)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(output)
plt.title("Try-On Output")
plt.axis("off")

plt.savefig(
    "virtual_tryon_output.png"
)

plt.show()
