


import time
import os
import cv2
import numpy as np

# Camera
cap = cv2.VideoCapture(0)

# Drawing state
canvas = None
colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255)]
color_index = 0
draw_color = colors[color_index]
prev_x, prev_y = None, None

# Gesture debounce state
last_color_gesture = False
last_clear_gesture = False
last_blur_gesture = False
blur = False
DEBOUNCE_DELAY = 0.5
last_action_time = 0

# Simple hand detector using HSV color detection (skin color)
def detect_hand_center(frame):
    """
    Simple hand detection using skin color in HSV space.
    Returns the center of the largest skin-colored region or None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 500:  # Minimum area threshold
        return None, None
    
    # Get center of contour
    M = cv2.moments(largest_contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy), largest_contour
    
    return None, largest_contour

try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Simple hand detection
        hand_pos, hand_contour = detect_hand_center(frame)
        
        if hand_pos is not None:
            x1, y1 = hand_pos
            h, w, _ = frame.shape
            
            # Simple drawing mode: draw when hand is detected
            # For simplicity, we'll just draw continuously when hand is in upper half
            if y1 < h // 2:  # Drawing region is upper half
                if prev_x is None:
                    prev_x, prev_y = x1, y1
                cv2.line(canvas, (prev_x, prev_y), (x1, y1), draw_color, 8)
                prev_x, prev_y = x1, y1
            else:
                prev_x, prev_y = None, None
            
            # Draw hand center
            cv2.circle(frame, (x1, y1), 5, (0, 255, 255), -1)
            
            # Color change: hand at bottom of frame
            color_gesture = y1 > h * 0.8
            if color_gesture and not last_color_gesture and (time.time() - last_action_time) > DEBOUNCE_DELAY:
                color_index = (color_index + 1) % len(colors)
                draw_color = colors[color_index]
                last_action_time = time.time()
                print(f"Color changed to {color_index + 1}")
            last_color_gesture = color_gesture
            
            # Clear canvas: hand at left side
            clear_gesture = x1 < w * 0.2
            if clear_gesture and not last_clear_gesture and (time.time() - last_action_time) > DEBOUNCE_DELAY:
                canvas = np.zeros_like(frame)
                last_action_time = time.time()
                print("Canvas cleared")
            last_clear_gesture = clear_gesture
            
            # Blur toggle: hand at right side
            blur_gesture = x1 > w * 0.8
            if blur_gesture and not last_blur_gesture and (time.time() - last_action_time) > DEBOUNCE_DELAY:
                blur = not blur
                last_action_time = time.time()
                print(f"Blur toggled: {blur}")
            last_blur_gesture = blur_gesture
        else:
            prev_x, prev_y = None, None

        # Apply blur if toggled (background only)
        display = frame.copy()
        if blur:
            display = cv2.GaussianBlur(display, (21, 21), 0)

        # Combine canvas and camera feed
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        display = cv2.bitwise_and(display, inv)
        display = cv2.bitwise_or(display, canvas)

        # Draw UI text
        color_names = ["Magenta", "Red", "Green", "Cyan", "Blue"]
        cv2.putText(display, f"Color: {color_names[color_index]} (Left=Clear, Right=Blur, Bottom=Color)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"Blur: {'ON' if blur else 'OFF'} | Press 's' to save, 'q' to quit", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Finger Drawing App", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            fname = os.path.join(os.getcwd(), f"drawing_{int(time.time())}.png")
            cv2.imwrite(fname, display)
            print(f"Saved: {fname}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")
