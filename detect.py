import cv2
import numpy as np
import pandas as pd

def getColorName(R, G, B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
displayed_colors = {} 
display_duration = 50
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_range = np.array([20, 50, 50])
        upper_range = np.array([40, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_range, upper_range)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                color = frame[cy, cx]
                detected_color = getColorName(*color)
                print(f"Detected Color: {detected_color}")
                cv2.putText(frame, f"{detected_color}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                displayed_colors[detected_color] = display_duration
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
        for color_name, counter in list(displayed_colors.items()):
            if counter > 0:
                cv2.putText(frame, f"{color_name}", (10, 30 + (50 * (display_duration - counter))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                displayed_colors[color_name] -= 1
            else:
                del displayed_colors[color_name]
        cv2.imshow('Video Feed', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
