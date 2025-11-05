import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import platform
from ultralytics import YOLO
import cv2
import numpy as np
import time
import argparse

# --- 1. 기본 설정 ---

# 클래스 이름
CLASS_NAMES = {
    0: 'hand',
    1: 'chickenmayo',
    2: 'seaweed_soup',
    3: 'condition_stick',
    4: 'pepero_original',
    5: 'pulmuone_spring_water',
    6: 'samdasoo', 
    7: 'creeat_protein_bar'
}

# (B, G, R) 포맷의 클래스별 색상 (시각화용)
COLORS = [
    (0,0,0),
    (255, 158, 66),
    (40, 181, 224),
    (209, 247, 84),
    (148, 148, 255),
    (110, 255, 84),
    (235, 213, 47),
    (247, 84, 171)
]

# (시각화용) 바운딩 박스 라벨 폰트 크기
FONT_SCALE = 0.4
FONT_THICKNESS = 1
BOX_THICKNESS = 2


# --- 2. Argument Parser ---

def parse_arguments():
    """커맨드라인 argument 파싱"""
    parser = argparse.ArgumentParser(description='YOLO 모델 평가 및 시각화')
    
    parser.add_argument('--model', type=str, required=True,
                        help='모델 경로 (.pt 파일)')
    parser.add_argument('--test-dir', type=str, default='/data/CRK/new_dataset/train_model/7subset_/test',
                        help='테스트 데이터셋 폴더 (images/, labels/ 포함)')
    parser.add_argument('--project', type=str, default='/home/yeojin/yolo_train/runs/infer',
                        help='예측 결과를 저장할 상위 폴더 (default: /home/yeojin/yolo_train/runs/infer)')
    parser.add_argument('--name', type=str, default='grab_mid_pred_i0.5_hsv-s0.2_hsv-v0.25',
                        help='예측 결과를 저장할 하위 폴더 이름 (default: exp)')
    parser.add_argument('--vis-name', type=str, default='grab_mid_vis',
                        help='비교 이미지 저장 폴더 이름 (default: grab_mid_vis)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU 임계값 (default: 0.5)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence 임계값 (default: 0.5)')
    
    return parser.parse_args()


# --- 3. IoU 계산 함수 ---

def calculate_iou(box1, box2):
    """YOLO 포맷 (xc, yc, w, h)의 두 박스 IoU 계산"""
    def box_to_corners(box):
        x_c, y_c, w, h = box
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return x1, y1, x2, y2

    b1_x1, b1_y1, b1_x2, b1_y2 = box_to_corners(box1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box_to_corners(box2)
    
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    if union_area == 0:
        return 0.0
    iou = inter_area / union_area
    return iou

# --- 4. 한글 폰트 설정 (Matplotlib) ---

def set_korean_font():
    system_name = platform.system()
    try:
        if system_name == "Darwin":  # Mac
            plt.rcParams["font.family"] = "AppleGothic"
        elif system_name == "Windows":  # Windows
            plt.rcParams["font.family"] = "Malgun Gothic"
        else:  # Linux (Ubuntu/Colab)
            plt.rcParams["font.family"] = "NanumGothic"
        plt.rcParams["axes.unicode_minus"] = False
        print(" 한글 폰트 설정 완료.")
    except Exception as e:
        print(f" 한글 폰트 설정 실패 (그래프 한글 깨짐 가능성): {e}")

# --- 5. (시각화용) 박스 그리기 헬퍼 함수 ---
def draw_on_image(image, boxes, class_names, colors, is_gt=False, img_width=0, img_height=0, font_scale=0.6, font_thickness=2, box_thickness=2):
    if is_gt:
        # GT (YOLO normalized format)
        for box_data in boxes:
            gt_class = int(box_data[0])
            xc, yc, bw, bh = [float(v) for v in box_data[1:5]]
            x1 = int((xc - bw / 2) * img_width)
            y1 = int((yc - bh / 2) * img_height)
            x2 = int((xc + bw / 2) * img_width)
            y2 = int((yc + bh / 2) * img_height)
            label_text = f"GT: {class_names.get(gt_class, 'Unknown')}"
            color = colors[gt_class % len(colors)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
    else:
        # Predictions (YOLO result.boxes object)
        for box in boxes:
            pred_class = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
            
            # Confidence score 추가
            label_text = f"Pred: {class_names.get(pred_class, 'Unknown')} ({conf:.2f})"
            color = colors[pred_class % len(colors)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
    return image


# --- 6. 메인 실행 함수 ---

def run_evaluation_and_visualization(model_path, test_dir, project_dir, name_dir, 
                                      vis_name_dir, iou_threshold, conf_threshold):
    """
    평가 및 시각화 실행
    
    Args:
        model_path: 모델 경로
        test_dir: 테스트 데이터셋 폴더
        project_dir: 예측 결과 저장 상위 폴더
        name_dir: 예측 결과 저장 하위 폴더 이름
        vis_name_dir: 비교 이미지 저장 폴더 이름
        iou_threshold: IoU 임계값
        conf_threshold: Confidence 임계값
    """
    
    # 경로 계산
    SOURCE_IMG_DIR = os.path.join(test_dir, "images")
    GT_DIR = os.path.join(test_dir, "labels")
    PRED_DIR = os.path.join(project_dir, name_dir, "labels")
    OUTPUT_CHART_FILE = os.path.join(project_dir, name_dir, "target_success_rate_chart.png")
    VIS_SAVE_DIR = os.path.join(project_dir, name_dir, vis_name_dir)
    
    # === 1단계: YOLO PREDICT 실행 (평가용 .txt 파일 생성) ===
    print("="*60)
    print(f"1단계: YOLO Predict를 시작합니다... (평가용 .txt 파일 생성)")
    print(f"  > 모델: {model_path}")
    print(f"  > 이미지 소스: {SOURCE_IMG_DIR}")
    print(f"  > .txt 저장 위치: {os.path.join(project_dir, name_dir)}")
    print("="*60)

    try:
        model = YOLO(model_path)
        model.predict(
            source=SOURCE_IMG_DIR,
            project=project_dir,
            name=name_dir,
            save_txt=True,
            save_conf=True,
            exist_ok=True
        )
        print("\n✅ 1단계: YOLO Predict 완료.")
        
    except Exception as e:
        print(f"\n[오류] 1단계 YOLO Predict 실행 중 문제 발생: {e}")
        print("모델 경로, 소스 경로, 'ultralytics' 라이브러리 설치를 확인하세요.")
        return

    # === ❗ 2단계: 타겟 탐지 성공률 분석 (필터링 적용) ===
    print("\n" + "="*60)
    print(f"2단계: 타겟 탐지 성공률 분석을 시작합니다...")
    print(f" ❗ [필터 적용됨]: GT 중심점이 이미지 상단 50% (Y < 0.5)인 대상만 집계")
    print(f"  > GT 라벨: {GT_DIR}")
    print(f"  > Pred 라벨: {PRED_DIR}")
    print(f"  > IoU 임계값: {iou_threshold}")
    print("="*60)
    
    class_stats = defaultdict(lambda: {'total_gt': 0, 'hits': 0})
    total_valid_gt_objects = 0  # ❗ 분모가 될 '찐 GT'의 총 개수
    total_hits = 0              # ❗ '찐 GT' 중 성공한 개수

    gt_files = glob.glob(os.path.join(GT_DIR, "*.txt"))
    total_file_count = len(gt_files) # 참고용 전체 파일 수

    if total_file_count == 0:
        print(f"[오류] GT 파일을 찾을 수 없습니다. 경로를 확인하세요: {GT_DIR}")
        return
        
    if not os.path.exists(PRED_DIR):
        print(f"[오류] Predict 결과 폴더를 찾을 수 없습니다. 1단계가 실패했을 수 있습니다. 경로: {PRED_DIR}")
        return

    for gt_file_path in gt_files:
        basename = os.path.basename(gt_file_path)
        pred_file_path = os.path.join(PRED_DIR, basename)
        
        # --- 1. Pred 파일 먼저 읽기 (매칭을 위해) ---
        pred_boxes_by_class = defaultdict(list)
        if os.path.exists(pred_file_path):
            try:
                with open(pred_file_path, 'r') as f:
                    for pred_line in f:
                        pred_data = pred_line.strip().split()
                        pred_class = int(pred_data[0])
                        pred_box = [float(v) for v in pred_data[1:5]]
                        pred_boxes_by_class[pred_class].append(pred_box)
            except Exception:
                pass # Pred 파일 읽기 실패해도 계속 진행
                        
        # --- 2. GT 파일 읽으면서 필터링 및 매칭 ---
        try:
            with open(gt_file_path, 'r') as f:
                # ❗❗❗ [수정] 파일의 '모든' GT 객체를 확인
                for gt_line in f: 
                    gt_data = gt_line.strip().split()
                    if not gt_data: 
                        continue
                    
                    gt_class = int(gt_data[0])
                    gt_box = [float(v) for v in gt_data[1:5]]
                    gt_center_y = gt_box[1] # YOLO 포맷의 Yc 좌표
                    
                    # ❗❗❗ [핵심] GT 필터링 로직
                    # 중심점이 상단 절반(Y < 0.5)에 있는지 확인
                    if gt_center_y < 0.5:
                        # 이 GT는 '찐 GT' (집계 대상)
                        class_stats[gt_class]['total_gt'] += 1
                        total_valid_gt_objects += 1
                        
                        gt_is_hit = False
                        
                        # 이 '찐 GT'와 일치하는 Pred가 있는지 확인
                        if gt_class in pred_boxes_by_class:
                            for pred_box in pred_boxes_by_class[gt_class]:
                                # 1. 클래스 일치 (이미 확인됨)
                                # 2. IoU가 임계값 이상인가?
                                iou = calculate_iou(gt_box, pred_box)
                                if iou >= iou_threshold:
                                    gt_is_hit = True
                                    break # 이 GT는 '성공' 처리. Pred 루프 중단.
                        
                        if gt_is_hit:
                            class_stats[gt_class]['hits'] += 1
                            total_hits += 1
                            
                    else:
                        # gt_center_y >= 0.5 이므로, 이 GT는 집계 대상에서 제외
                        pass
                        
        except Exception:
            continue # GT 파일 읽기 오류시 다음 파일로
            
    # === ❗ 3단계: 성공률 결과 출력 (터미널) ===
    print("\n---  타겟 탐지 성공률 (Class-wise, ❗상단 50% 필터 적용) ---")
    sorted_class_ids = sorted(class_stats.keys())
    
    labels_for_plot = []
    rates_for_plot = []
    texts_for_plot = []

    for class_id in sorted_class_ids:
        class_name = CLASS_NAMES.get(class_id, f"Unknown_ID_{class_id}")
        total = class_stats[class_id]['total_gt']
        hits = class_stats[class_id]['hits']
        
        if total > 0:
            rate = (hits / total) * 100
            print(f"  ▶ {class_name:<25} (ID {class_id}): {hits} / {total} (성공률: {rate:.2f}%)")
            labels_for_plot.append(class_name)
            rates_for_plot.append(rate)
            texts_for_plot.append(f"{hits}/{total}")
        else:
            # 이 클래스의 GT가 모두 하단에만 있었던 경우
            print(f"  ▶ {class_name:<25} (ID {class_id}): 0 / 0 (상단 50%에 GT 없음)")

    print("-------------------------------------------")
    # ❗ [수정] 분모를 total_valid_gt_objects로 변경
    total_rate = (total_hits / total_valid_gt_objects) * 100 if total_valid_gt_objects > 0 else 0
    print(f" ★ 전체 성공률: {total_hits} / {total_valid_gt_objects} ( {total_rate:.2f}% )")
    print(f"  (참고: 전체 GT 파일 {total_file_count}개에서 {total_valid_gt_objects}개의 '찐 GT'를 집계함)")
    print("="*60)

    # === 4단계: 성공률 결과 시각화 (차트 저장) ===
    # (4단계는 원본 코드와 동일)
    print(f"\n4단계: 성공률 그래프를 저장합니다...")
    set_korean_font()

    try:
        fig, ax = plt.subplots(figsize=(max(10, len(labels_for_plot) * 1.5), 7))
        # 0개의 '찐 GT'가 발견된 경우, rates_for_plot이 비어있을 수 있음
        if rates_for_plot:
            colors = plt.cm.viridis_r(plt.Normalize(min(rates_for_plot), max(rates_for_plot))(rates_for_plot))
        else:
            colors = 'blue' # 기본 색상
        
        bar_container = ax.bar(labels_for_plot, rates_for_plot, color=colors)
        
        ax.set_ylabel('Success rate (%)', fontsize=12)
        ax.set_title('Target Detection Rate (by class) - Top 50% Filtered', fontsize=16, pad=20) # ❗제목 수정
        ax.set_ylim(0, 110) 
        if rates_for_plot: # 데이터가 있을 때만 라벨 표시
            ax.bar_label(bar_container, fmt='%.1f%%', fontsize=10, padding=3)
        plt.xticks(rotation=30, ha='right', fontsize=11)
        
        for i, txt in enumerate(texts_for_plot):
            ax.text(i, -5, txt, ha='center', va='top', fontsize=10, color='gray')
            
        plt.tight_layout(pad=2.0)
        plt.savefig(OUTPUT_CHART_FILE)
        
        print(f" 4단계: 그래프 저장 완료!\n  > 저장 경로: {OUTPUT_CHART_FILE}")

    except Exception as e:
        print(f"[오류] 그래프 저장 중 오류 발생: {e}")
        print("matplotlib 또는 한글 폰트 설정을 확인하세요.")

    # === 5단계: GT / Pred 비교 이미지 생성 ===
    # (5단계는 원본 코드와 동일 - 시각화는 필터링 없이 모두 보여줌)
    print("\n" + "="*60)
    print(f"5단계: GT/Pred 비교 시각화를 시작합니다...")
    print(f" (참고: 시각화는 필터링과 관계없이 모든 GT와 Pred를 그립니다)")
    print(f" > 이미지 저장 위치: {VIS_SAVE_DIR}")
    print(f" > Conf 임계값: {conf_threshold}")
    print("="*60)

    os.makedirs(VIS_SAVE_DIR, exist_ok=True)
    
    try:
        results_generator = model.predict(
            source=SOURCE_IMG_DIR, 
            stream=True, 
            conf=conf_threshold,
            save=False,
            save_txt=False
        )
    except Exception as e:
        print(f"[오류] 5단계 Predict 실행 실패: {e}")
        return

    vis_images_count = 0
    start_time = time.time()

    for result in results_generator:
        image_path = result.path
        basename = os.path.basename(image_path)
        
        img = cv2.imread(image_path)
        if img is None: continue
            
        h, w, _ = img.shape
        
        img_gt = img.copy()
        img_pred = img.copy()

        # (좌측) Ground Truth 그리기
        gt_boxes = []
        gt_label_path = os.path.join(GT_DIR, os.path.splitext(basename)[0] + ".txt")
        if os.path.exists(gt_label_path):
            try:
                with open(gt_label_path, 'r') as f:
                    for line in f:
                        gt_boxes.append(line.strip().split())
            except Exception:
                pass 
        
        img_gt = draw_on_image(img_gt, gt_boxes, CLASS_NAMES, COLORS, 
                                is_gt=True, img_width=w, img_height=h,
                                font_scale=FONT_SCALE, font_thickness=FONT_THICKNESS, box_thickness=BOX_THICKNESS)

        # (우측) Prediction 그리기
        img_pred = draw_on_image(img_pred, result.boxes, CLASS_NAMES, COLORS, 
                                  is_gt=False,
                                  font_scale=FONT_SCALE, font_thickness=FONT_THICKNESS, box_thickness=BOX_THICKNESS)
        
        # 타이틀 추가
        cv2.putText(img_gt, "Ground Truth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
        cv2.putText(img_gt, "Ground Truth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(img_pred, "Prediction", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
        cv2.putText(img_pred, "Prediction", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        # 두 이미지 수평으로 합치기
        comparison_img = np.hstack((img_gt, img_pred))
        
        # 결과 이미지 저장
        save_path = os.path.join(VIS_SAVE_DIR, basename)
        cv2.imwrite(save_path, comparison_img)
        
        vis_images_count += 1
        if vis_images_count % 10 == 0:
            print(f"  ... {vis_images_count}개 비교 이미지 생성 완료.")

    end_time = time.time()
    print("\n" + "="*60)
    print(" 5단계: 모든 시각화 작업 완료!")
    print(f" > 총 {vis_images_count}개의 비교 이미지를 생성했습니다.")
    print(f" > 총 소요 시간: {end_time - start_time:.2f} 초")
    print(f" > 결과물 저장 위치: {VIS_SAVE_DIR}")
    print("="*60)


# --- 스크립트 실행 ---
if __name__ == "__main__":
    args = parse_arguments()
    
    run_evaluation_and_visualization(
        model_path=args.model,
        test_dir=args.test_dir,
        project_dir=args.project,
        name_dir=args.name,
        vis_name_dir=args.vis_name,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold
    )