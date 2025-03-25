import cv2
import numpy as np
import os as os
def cartoonize_image(img_path, output_path="cartoon_output.jpg"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, img_path)
    output_path = os.path.join(script_dir, output_path)
    # 이미지 읽기
    img = cv2.imread(img_path)
    img = cv2.resize(img, (800, 800))  # 해상도 조정 (선택사항)

    # 1. 노이즈 제거를 위한 bilateral 필터 적용
    smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. 엣지 검출을 위한 그레이스케일 변환 후 median 블러 적용
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)

    # 3. 엣지 검출 (adaptive threshold 사용)
    edges = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, blockSize=5, C=2)

    # 4. 엣지를 컬러로 변환
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 5. 스무딩된 이미지와 엣지를 결합
    cartoon = cv2.bitwise_and(smooth, edges_colored)

    # 결과 저장 및 보기
    cv2.imwrite(output_path, cartoon)
    print(f"카툰 스타일 이미지가 저장되었습니다: {output_path}")

    return cartoon

# 사용 예시
cartoonize_image("Good_Example.jpg")
