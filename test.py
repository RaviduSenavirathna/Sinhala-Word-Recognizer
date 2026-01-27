import cv2
import os
import time

LABEL = "Z" #folder label

SAVE_DIR = f"data_raw/custom_train/{LABEL}"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

count = len(os.listdir(SAVE_DIR))  # continue from existing count
saving = False
last_save = 0

print("Press 's' to start/stop saving frames. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

 

    cv2.putText(frame, f"LABEL: {LABEL} | Saved: {count} | Saving: {saving}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Record Custom Data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        saving = not saving

    # Save 1 frame every 0.12 seconds while saving
    if saving and (time.time() - last_save) > 0.05:
        path = os.path.join(SAVE_DIR, f"{LABEL}_{count:06d}.jpg")
        cv2.imwrite(path, frame)
        count += 1
        last_save = time.time()

cap.release()
cv2.destroyAllWindows()