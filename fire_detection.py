import cv2
import numpy as np

def detect_fire(frame):
    """Detect fire in the frame."""
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for fire-like colors
    lower_fire = np.array([18, 50, 50])  # Lower bound for fire colors
    upper_fire = np.array([35, 255, 255])  # Upper bound for fire colors

    # Create a mask for fire-like colors
    mask = cv2.inRange(hsv_frame, lower_fire, upper_fire)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_detected = False  # Flag to indicate if fire is detected
    large_fire_detected = False  # Flag to indicate if large fire is detected

    for contour in contours:
        # Calculate the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Define thresholds for small and large fires (adjust as needed)
        if area > 500:  # Small fire threshold
            fire_detected = True
            if area > 2000:  # Large fire threshold
                large_fire_detected = True
                # Draw a red bounding box for large fires
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "LARGE FIRE ALERT!", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Draw a yellow bounding box for small fires
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "FIRE", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame, fire_detected, large_fire_detected

def fire_detection_live_feed():
    """Live fire detection feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to exit the live feed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect fire in the frame
        frame, fire_detected, large_fire_detected = detect_fire(frame)

        # Display fire status
        if fire_detected:
            cv2.putText(frame, "Fire Detected!", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if large_fire_detected:
            cv2.putText(frame, "LARGE FIRE ALERT!", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the live feed
        cv2.imshow("Fire Detection", frame)

        # Press 'q' to quit the live feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run fire detection directly
if __name__ == "__main__":
    fire_detection_live_feed()