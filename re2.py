import os
import time
import numpy as np
import pandas as pd
import torch
import streamlit as st
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher
import csv

# 환경 설정
if "TORCH_HOME" in os.environ:
    del os.environ["TORCH_HOME"]
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 차량 감지 박스 설정 (1번 자리부터 5번 자리까지 설정)
fixed_detection_boxes = [
    (300, 600, 330, 630),  # 1번 자리
    (620, 600, 650, 630),  # 2번 자리
    (920, 600, 950, 630),  # 3번 자리
    (1220, 600, 1250, 630),  # 4번 자리
    (1620, 600, 1650, 630)   # 5번 자리
]

# 사전 정의된 번호판 목록
defined_plates = [
    "57나2599", "05누4450", "160허6663"
]

# OCR 텍스트와 사전 정의된 번호판 간 유사도를 비교해 가장 유사한 번호판을 반환하는 함수
def match_defined_plate(ocr_text, defined_plates):
    """
    OCR로 인식된 텍스트와 사전 정의된 번호판 목록을 비교하여
    가장 유사한 번호판 텍스트를 반환하는 함수.
    """
    highest_similarity = 0
    best_match = ocr_text  # 기본적으로 원본 OCR 결과 반환

    # 사전 정의된 번호판 목록과 비교
    for plate in defined_plates:
        # 문자열 유사도를 계산
        similarity = SequenceMatcher(None, ocr_text, plate).ratio()

        # 가장 높은 유사도를 가진 번호판 선택
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = plate

    # 유사도가 0.7 이상일 때만 교정된 번호판을 반환
    if highest_similarity >= 0.3:
        return best_match
    else:
        return ocr_text  # 유사도가 낮으면 원본 OCR 텍스트 반환
    
# 주차 상태와 로그 초기화
detection_count = defaultdict(int)
parking_log = []
log_file_path = "log.csv"

# Streamlit 페이지 설
st.set_page_config(layout="wide", page_title="주차장 관리 시스템")
st.title("주차장 관리 시스템")
left_column, right_column = st.columns([1.7, 2])

# 그래프와 테이블 위치 설정
with right_column:
    st.write("주차 로그 및 실시간 통계")
    parking_count_data = pd.DataFrame(columns=["주차 대수"])  # 초기값이 빈 DataFrame으로 시작
    parking_count_chart = st.line_chart(parking_count_data)  # 초기 차트 생성
    parking_log_table = st.empty()  # 로그 테이블 위치

# 포함 여부 계산 함수
def calculate_inclusion(main_box, check_box):
    xA = max(main_box[0], check_box[0])
    yA = max(main_box[1], check_box[1])
    xB = min(main_box[2], check_box[2])
    yB = min(main_box[3], check_box[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    checkBoxArea = (check_box[2] - check_box[0]) * (check_box[3] - check_box[1])
    inclusion_ratio = interArea / float(checkBoxArea)
    return inclusion_ratio

# 실시간 주차 대수 기록 함수
def update_parking_count_chart(occupied_spots):
    global parking_count_data  # parking_count_data를 전역 변수로 선언
    # 새로운 데이터 추가 (최근 10개 항목 유지)
    new_data = pd.DataFrame({"주차 대수": [occupied_spots]})  # 새 데이터 생성
    parking_count_data = pd.concat([parking_count_data, new_data], ignore_index=True).tail(10)  # 데이터 합치기
    parking_count_chart.line_chart(parking_count_data)  # 그래프 업데이트


# 실시간 주차 로그 테이블 업데이트 함수 (최신 정보만 반영)
def update_parking_log_table():
    if os.path.exists(log_file_path):
        df = pd.read_csv(log_file_path)
        # 시간 데이터 변환 및 필터링
        df['시간'] = pd.to_datetime(df['시간'], errors='coerce')
        
        # 현재 시간 계산
        current_time = datetime.now()

        # 출차된 차량은 출차 여부를 "yes"로 갱신
        for index, row in df.iterrows():
            if row['출차여부'] == "no":
                parked_time = row['시간']
                # 차량이 주차된 지 일정 시간이 지났다면 (예: 1시간 이상)
                if (current_time - parked_time).total_seconds() > 3600:  # 1시간 이상
                    df.at[index, '출차여부'] = "yes"
        
        # 출차된 차량의 로그를 갱신된 상태로 저장
        df.to_csv(log_file_path, index=False, encoding='utf-8')

        # 감지된 차량만 필터링하여 테이블에 표시 (각 자리에서 감지된 차량만 표기)
        latest_logs = df[df['출차여부'] == "no"].sort_values(by='시간', ascending=False).drop_duplicates(subset=['자리번호'], keep='first')
        
        # 테이블 갱신 (중복되지 않는 최신 차량 정보만 표시)
        parking_log_table.empty()  # 이전 표 삭제
        parking_log_table.table(latest_logs[['자리번호', '차량번호', '시간', '출차여부']])

    time.sleep(1) # 1초 간격

# 왼쪽 비디오 스트림 처리 함수
def left():
    with left_column:
        frame_window = st.empty()
        video_path = "1080p-1m.mp4"
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        vehicle_model = YOLO('yolov8n.pt')
        plate_model = YOLO('yolov8_license_plate.pt')
        ocr = PaddleOCR(use_angle_cls=True, lang='korean')
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % fps == 0:
                vehicle_results = vehicle_model(frame)
                box_status = {box: (0, 255, 255) for box in fixed_detection_boxes}
                occupied_spots = 0  # 감지된 주차 대수 초기화

                for vehicle_result in vehicle_results:
                    boxes = vehicle_result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls in [2, 3, 7]:  # 차량 클래스만 감지
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            vehicle_box = (x1, y1, x2, y2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            cropped_vehicle = frame[y1:y2, x1:x2]
                            plate_results = plate_model(cropped_vehicle)

                            for plate_result in plate_results:
                                # 번호판을 인식하여 텍스트로 변환
                                for plate_box in plate_result.boxes:
                                    x_plate1, y_plate1, x_plate2, y_plate2 = map(int, plate_box.xyxy[0])
                                    cropped_plate = cropped_vehicle[y_plate1:y_plate2, x_plate1:x_plate2]
                                    ocr_results = ocr.ocr(cropped_plate, cls=True)

                                    for line in ocr_results:
                                        if line:  # None 체크 추가
                                            for res in line:
                                                ocr_text = res[1][0]  # OCR 결과에서 텍스트 추출
                                                
                                                # 유사도 함수 호출하여 최종 텍스트 교정
                                                plate_text = match_defined_plate(ocr_text, defined_plates)

                                                if plate_text:
                                                    for fixed_box in fixed_detection_boxes:
                                                        if calculate_inclusion(vehicle_box, fixed_box) >= 0.95:
                                                            cv2.putText(cropped_vehicle, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                                            detection_count[fixed_box] += 1
                                                            if detection_count[fixed_box] >= 5:
                                                                box_status[fixed_box] = (0, 0, 255)
                                                                occupied_spots += 1  # 감지된 주차 대수 증가
                                                                parking_log.append({
                                                                    "자리번호": fixed_detection_boxes.index(fixed_box) + 1,
                                                                    "차량번호": plate_text.strip(),
                                                                    "시간": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                    "출차여부": "no"
                                                                })
                                                                with open(log_file_path, mode='a', newline='', encoding='utf-8') as f:
                                                                    writer = csv.DictWriter(f, fieldnames=parking_log[-1].keys())
                                                                    if f.tell() == 0:
                                                                        writer.writeheader()
                                                                    writer.writerow(parking_log[-1])
                                                            else:
                                                                box_status[fixed_box] = (0, 128, 255)


                for fixed_box, color in box_status.items():
                    if color == (0, 255, 255):
                        detection_count[fixed_box] = 0
                    cv2.rectangle(frame, (fixed_box[0], fixed_box[1]), (fixed_box[2], fixed_box[3]), color, 2)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame, use_column_width=True)

                # 실시간 주차 대수 그래프 업데이트
                update_parking_count_chart(occupied_spots)

                # 실시간 주차 로그 테이블 업데이트
                update_parking_log_table()

            frame_count += 1

if __name__ == "__main__":
    left()
