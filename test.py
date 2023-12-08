import cv2
import mediapipe as mp
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 载入自己训练的 YOLOv5 pt
model_yolo = torch.hub.load("C:/Users/USER/dec_logic/yolov5-master2_trainok",
                            'custom',
                            "C:/Users/USER/dec_logic/yolov5-master2_trainok/runs/train/exp14/weights/best.pt",
                            source='local')

# 初始化 mediapipe
mp_drawing = mp.solutions.drawing_utils  # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_hands = mp.solutions.hands  # mediapipe 偵測手掌方法

# 中文字型檔
font_text_24 = ImageFont.truetype("jf-openhuninn-1.1.ttf", 24, encoding="utf-8")
font_text_36 = ImageFont.truetype("jf-openhuninn-1.1.ttf", 36, encoding="utf-8")
font_text_48 = ImageFont.truetype("jf-openhuninn-1.1.ttf", 48, encoding="utf-8")


# 顯示中文字之副程式
def add_chinese_font_to_image(img, text, left, top, text_color=(0, 255, 0), font_text=font_text_24):

    # 判斷圖片格式，若為 OpenCV 格式，就做陣列與 BGR->RGB 的轉變
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 將 img 切換成 PIL 格式
    draw = ImageDraw.Draw(img)

    # 利用 draw 進行文字繪製
    draw.text((left, top), text, text_color, font=font_text)

    # 返回繪製好的 圖片+文字
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def draw_circle(frame, detections):
    num_objects = len(detections)  # 当前帧中检测到的物体数量

    if num_objects == 5:  # 如果检测到五个物体
        # 在 frame 右下角显示绿色圆圈
        cv2.circle(frame, (frame.shape[1] - 50, frame.shape[0] - 50), 20, (0, 255, 0), -1)
    else:
        # 在 frame 右下角显示红色圆圈
        cv2.circle(frame, (frame.shape[1] - 50, frame.shape[0] - 50), 20, (0, 0, 255), -1)

    return frame





# 讀取影片
video_path = "allori.mp4"
cap = cv2.VideoCapture(video_path)

# mediapipe 启动偵測手掌
hands_detector = mp_hands.Hands(
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# 創建視窗並讀取第一幀影像以獲取影片的寬高資訊
ret, frame = cap.read()
height, width, _ = frame.shape

# 指定圓形的位置和大小
x, y_red = width - 150, height - 50
y_green = y_red - 40  # 調整兩個圓形的垂直間距
circle_radius = 13
circle_color = (0, 255, 0)  # BGR 格式的顏色碼 (Green)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # 运行 YOLOv5 模型进行物体检测
    results_yolo = model_yolo(frame)
    detections = results_yolo.pred[0]

    # 在图像上绘制物体检测结果
    for det in detections:
        cv2.rectangle(frame, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), (13, 23, 227), 2)
        cv2.putText(frame, f'{model_yolo.names[int(det[5])]} {det[4]:.2f}', (int(det[0]), int(det[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (13, 23, 227), 2)

    # 偵測手掌
    results_hands = hands_detector.process(frame)

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # 在影像上繪製紅色圓形
    cv2.circle(frame, (x, y_red), circle_radius, (0, 0, 255), -1)  # 紅色 (BGR 格式)

    # 在影像上繪製綠色圓形
    cv2.circle(frame, (x, y_green), circle_radius, (0, 255, 0), -1)  # 綠色 (BGR 格式)

    # 在每一帧上添加文字
    frame = add_chinese_font_to_image(frame, "SOP偵測中：請按照SOP標準流程進行作業", 10, 500, font_text=font_text_24)

    cv2.imshow(' AI DETECTOR  :  YOLOv5 + MediaPipe', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
