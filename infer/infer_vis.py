import cv2
import os
import argparse
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# --- 1. Argument Parser 설정 ---
parser = argparse.ArgumentParser(description='YOLO Inference for Video or Image')
parser.add_argument('--mode', type=str, required=True, choices=['video', 'image'], 
                    help='Inference mode: video or image')
parser.add_argument('--source', type=str, required=True,
                    help='Path to source video or image file')
parser.add_argument('--model', type=str, required=True,
                    help='Path to YOLO model (.pt file)')
parser.add_argument('--conf', type=float, default=0.5,
                    help='Confidence threshold (default: 0.5)')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Output directory (default: yolo_train/runs/infer)')

args = parser.parse_args()

# --- 2. 설정 ---
MODEL_PATH = args.model
SOURCE_PATH = args.source
CONF_THRESHOLD = args.conf
MODE = args.mode

# 출력 디렉토리 설정 (기본: yolo_train/runs/infer)
script_dir = Path(__file__).resolve().parent  # infer.py가 있는 디렉토리 (절대 경로)
if args.output_dir is None:
    output_base_dir = script_dir.parent / 'runs' / 'infer'
else:
    output_base_dir = Path(args.output_dir).resolve()

# 출력 디렉토리 생성 (runs/infer/날짜_시간)
timestamp = datetime.now().strftime('%m%d_%H%M%S')
output_dir = output_base_dir / timestamp
output_dir.mkdir(parents=True, exist_ok=True)

# 출력 파일 경로 설정
source_name = Path(SOURCE_PATH).stem
if MODE == 'video':
    OUTPUT_PATH = output_dir / f"{source_name}_result.mp4"
else:
    OUTPUT_PATH = output_dir / f"{source_name}_result.jpg"

print("✓ 전체 화면을 감지합니다.")

CLASS_DICT = {
    0: 'hand', 1: 'chickenmayo', 2: 'seaweed_soup', 3: 'condition_stick',
    4: 'pepero_original', 5: 'pulmuone_spring_water', 6: 'samdasoo',
    7: 'creeat_protein_bar'
}

COLOR_MAP = {
    0: (255, 158, 66),   # hand - (R=66, G=158, B=255) -> (B=255, G=158, R=66)
    1: (40, 181, 224),   # chickenmayo - (R=224, G=181, B=40) -> (B=40, G=181, R=224)
    2: (209, 247, 84),   # seaweed_soup - (R=84, G=247, B=209) -> (B=209, G=247, R=84)
    3: (148, 148, 255),  # condition_stick - (R=255, G=148, B=148) -> (B=148, G=148, R=255)
    4: (110, 255, 84),   # pepero_original - (R=84, G=255, B=110) -> (B=110, G=255, R=84)
    5: (235, 213, 47),   # pulmuone_spring_water - (R=47, G=213, B=235) -> (B=235, G=213, R=47)
    6: (247, 84, 171),   # samdasoo - (R=171, G=84, B=247) -> (B=247, G=84, R=171)
    7: (180, 119, 31)    # creeat_protein_bar - (R=31, G=119, B=180) -> (B=180, G=119, R=31)
}

# 폰트 및 라인 설정
FONT_SCALE = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICKNESS = 2
TEXT_THICKNESS = 1

# --- 3. 모델 로드 ---
try:
    model = YOLO(MODEL_PATH)
    print(f"모델 로드 성공: {MODEL_PATH}")
except Exception as e:
    print(f"오류: 모델 로드 실패. {e}")
    exit()

# --- 4. Bbox 그리기 함수 ---
def draw_bbox(frame, box, midpoint_y=None):
    """Bbox와 라벨을 그리는 함수"""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # 중심점 필터링 (midpoint_y가 있으면 적용)
    if midpoint_y is not None:
        center_y = (y1 + y2) / 2
        if center_y >= midpoint_y:  # 중심이 중앙선 아래면 스킵
            return False, None, None
    
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    
    label_name = CLASS_DICT.get(cls_id, f'Class {cls_id}')
    label = f'{label_name}: {conf:.2f}'
    
    color = COLOR_MAP.get(cls_id, (255, 255, 255))
    
    # Bbox 그리기
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    
    # 텍스트 배경 계산 및 그리기
    (w, h), _ = cv2.getTextSize(label, FONT, FONT_SCALE, TEXT_THICKNESS)
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + h + 10
    cv2.rectangle(frame, (x1, text_y - h - 4), (x1 + w, text_y), color, -1)
    
    # 텍스트 (검은색)
    cv2.putText(frame, label, (x1, text_y - 3), FONT, FONT_SCALE, 
                (0, 0, 0), TEXT_THICKNESS, cv2.LINE_AA)
    
    return True, cls_id, label_name

# --- 5. Inference 실행 ---
print(f"\n{'='*60}")
print(f"Inference Mode: {MODE.upper()}")
print(f"Source: {SOURCE_PATH}")
print(f"Model: {MODEL_PATH}")
print(f"Confidence Threshold: {CONF_THRESHOLD}")
print(f"Output: {OUTPUT_PATH}")
print(f"{'='*60}\n")

if MODE == 'video':
    # === VIDEO 모드 ===
    cap = cv2.VideoCapture(SOURCE_PATH)
    if not cap.isOpened():
        print(f"오류: 비디오 파일을 열 수 없습니다. {SOURCE_PATH}")
        exit()
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    midpoint_y = None
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (frame_width, frame_height))
    
    print(f"비디오 처리 시작... (해상도: {frame_width}x{frame_height}, FPS: {fps:.2f})")
    print("필터링 없음 (전체 화면 감지)\n")
    print("--- 감지 로그 ---")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        
        detected_in_this_frame = False
        for box in results[0].boxes:
            is_detected, cls_id, label_name = draw_bbox(frame, box, midpoint_y)
            if is_detected:
                conf = float(box.conf[0])
                print(f"  [Frame {frame_count:04d}] ID={cls_id}, Name={label_name} (Conf: {conf:.2f})")
                detected_in_this_frame = True
        
        out.write(frame)
        frame_count += 1
        
        # 진행 상황 표시
        if not detected_in_this_frame and frame_count % (int(fps) * 5) == 0:
            print(f"  ... (Processing frame {frame_count}) ...")
    
    cap.release()
    out.release()
    
    print(f"\n총 {frame_count} 프레임 처리 완료")
    print(f"결과 저장: {OUTPUT_PATH}")

else:
    # === IMAGE 모드 ===
    frame = cv2.imread(SOURCE_PATH)
    if frame is None:
        print(f"오류: 이미지 파일을 열 수 없습니다. {SOURCE_PATH}")
        exit()
    
    frame_height = frame.shape[0]
    midpoint_y = None
    
    print(f"이미지 처리 시작... (해상도: {frame.shape[1]}x{frame.shape[0]})")
    print("필터링 없음 (전체 화면 감지)\n")
    
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    
    print("--- 감지 결과 ---")
    detection_count = 0
    for box in results[0].boxes:
        is_detected, cls_id, label_name = draw_bbox(frame, box, midpoint_y)
        if is_detected:
            conf = float(box.conf[0])
            print(f"  ID={cls_id}, Name={label_name} (Conf: {conf:.2f})")
            detection_count += 1
    
    cv2.imwrite(str(OUTPUT_PATH), frame)
    
    print(f"\n총 {detection_count}개 객체 감지")
    print(f"결과 저장: {OUTPUT_PATH}")

print(f"\n{'='*60}")
print("Inference 완료!")
print(f"{'='*60}")