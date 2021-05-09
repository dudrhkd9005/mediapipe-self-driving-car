import csv

import cv2
import mediapipe as mp
import numpy as np

def initCSVFile(images, filename):
    landmarks = ["class"]
    nums = len(images.pose_landmarks.landmark) + len(images.face_landmarks.landmark)
    for val in range(1, nums + 1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

    with open(filename, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)



if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, images.face_landmarks, mp_holistic.FACE_CONNECTIONS, mp_drawing.DrawingSpec(color=(77, 115, 12), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(77, 230, 119), thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(image, images.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(119, 20, 78), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(119, 40, 250), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(image, images.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(76, 20, 14), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(76, 40, 120), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(image, images.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(240, 115, 63), thickness=2, circle_radius=4),mp_drawing.DrawingSpec(color=(240, 65, 230), thickness=2, circle_radius=2))

            cv2.imshow('Cameras', image)

            if cv2.waitKey(10) & 0xFF == ord('s'):
                try:
                    initCSVFile(images, 'models.csv')
                    break
                except:
                    print("not detect face and body")
                    continue


    cap.release()
    cv2.destroyAllWindows()

    input_class = input("class name: ")
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            mp_drawing.draw_landmarks(image, images.face_landmarks, mp_holistic.FACE_CONNECTIONS, mp_drawing.DrawingSpec(color=(77, 115, 12), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(77, 230, 119), thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(image, images.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(119, 20, 78), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(119, 40, 250), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(image, images.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(76, 20, 14), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(76, 40, 120), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(image, images.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(240, 115, 63), thickness=2, circle_radius=4),mp_drawing.DrawingSpec(color=(240, 65, 230), thickness=2, circle_radius=2))

            # Show Image
            cv2.imshow('Cameras', image)

            try:
                pose = images.pose_landmarks.landmark
                pose_row = list(
                    np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                face = images.face_landmarks.landmark
                face_row = list(
                    np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                row = pose_row + face_row
                row.insert(0, input_class)

                with open("models.csv", mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
            except:
                print("not detect face and body")
                continue

            if cv2.waitKey(10) & 0xFF == ord('s'):
                print("Update class")
                break

    cap.release()
    cv2.destroyAllWindows()
    exit()
