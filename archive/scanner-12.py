# -*- coding: utf-8 -*-
# scanner-app 改善版：OCR方向判定、薄い文字の強調、外周の縁除去
# document.jpg のあるフォルダーから実行してください。

import cv2
import numpy as np
import pytesseract
import os
import shutil
import sys
INPUT_FILE = 'document.jpg'
OUTPUT_FILE = 'document_1.jpg'
PAPER_SIZE = 'A4'
DPI = 200
SHOW_FINAL_PREVIEW = True
DEBUG_PREVIEW = False
DEBUG_WAIT = 1200
SHOW_OCR_SCORES = True
GUIDE_MARGIN_X = 0.1
GUIDE_MARGIN_Y = 0.1
MIN_AREA_RATIO = 0.15
MAX_AREA_RATIO = 0.95
PAPER_SIZES_MM = {'A4': (210, 297), 'B5': (182, 257)}

def setup_tesseract():
    candidates = [shutil.which('tesseract'), os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Tesseract-OCR', 'tesseract.exe'), 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe', 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe']
    for path in candidates:
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return True
    return False

def get_ocr_language():
    try:
        languages = pytesseract.get_languages(config='')
    except Exception:
        languages = []
    print('Tesseract languages :', languages)
    if 'jpn' in languages and 'eng' in languages:
        return 'jpn+eng'
    if 'jpn' in languages:
        return 'jpn'
    if 'eng' in languages:
        print('警告：日本語OCRデータ jpn がありません。')
        print('日本語書類の方向判定精度が低下する可能性があります。')
        return 'eng'
    return None

def order_points(pts):
    pts = pts.reshape(4, 2).astype('float32')
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))
    if max_width <= 0 or max_height <= 0:
        return None
    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype='float32')
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return warped

def approximate_quadrilateral(contour):
    peri = cv2.arcLength(contour, True)
    epsilon_values = [0.02, 0.025, 0.03]
    for epsilon in epsilon_values:
        approx = cv2.approxPolyDP(contour, epsilon * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        return approx
    return None

def calculate_document_score(approx, area, guide_corners, guide_area, guide_center, image_diagonal):
    rect = order_points(approx)
    document_center = np.mean(rect, axis=0)
    center_distance = np.linalg.norm(document_center - guide_center)
    center_score = 1.0 - min(center_distance / (image_diagonal * 0.35), 1.0)
    corner_distances = []
    for (document_point, guide_point) in zip(rect, guide_corners):
        distance = np.linalg.norm(document_point - guide_point)
        corner_distances.append(distance)
    mean_corner_distance = np.mean(corner_distances)
    corner_score = 1.0 - min(mean_corner_distance / (image_diagonal * 0.35), 1.0)
    size_difference = abs(area - guide_area) / guide_area
    size_score = 1.0 - min(size_difference, 1.0)
    score = corner_score * 0.45 + size_score * 0.35 + center_score * 0.2
    return score

def rotate_image(image, rotation):
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image.copy()

def mm_to_px(mm, dpi):
    return int(round(mm / 25.4 * dpi))

def get_paper_pixel_size(paper, dpi, orientation):
    if paper not in PAPER_SIZES_MM:
        raise ValueError('PAPER_SIZE は A4 または B5 を指定してください。')
    (width_mm, height_mm) = PAPER_SIZES_MM[paper]
    width = mm_to_px(width_mm, dpi)
    height = mm_to_px(height_mm, dpi)
    if orientation == 'portrait':
        return (width, height)
    return (height, width)

def get_ocr_candidate_size(paper, orientation):
    OCR_DPI = 110
    return get_paper_pixel_size(paper, OCR_DPI, orientation)

def prepare_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    (_, binary) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def calculate_ocr_score(image, language):
    prepared = prepare_for_ocr(image)
    try:
        data = pytesseract.image_to_data(prepared, lang=language, config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
    except Exception as error:
        print('OCRエラー :', error)
        return -1
    weighted_conf_sum = 0.0
    weight_sum = 0
    character_count = 0
    detected_items = 0
    for (text, confidence) in zip(data['text'], data['conf']):
        text = text.strip()
        if not text:
            continue
        try:
            confidence = float(confidence)
        except ValueError:
            continue
        if confidence < 0:
            continue
        char_length = max(len(text), 1)
        weighted_conf_sum += confidence * char_length
        weight_sum += char_length
        character_count += char_length
        detected_items += 1
    if weight_sum == 0:
        return 0
    average_confidence = weighted_conf_sum / weight_sum
    character_bonus = min(character_count, 120) / 120 * 15
    item_bonus = min(detected_items, 30) / 30 * 5
    score = average_confidence * 0.8 + character_bonus + item_bonus
    return score

def detect_best_orientation(warped, paper, language):
    rotations = [0, 90, 180, 270]
    orientations = ['portrait', 'landscape']
    best_score = -1
    best_rotation = 0
    best_orientation = 'portrait'
    for rotation in rotations:
        rotated = rotate_image(warped, rotation)
        for orientation in orientations:
            (candidate_width, candidate_height) = get_ocr_candidate_size(paper, orientation)
            candidate = cv2.resize(rotated, (candidate_width, candidate_height), interpolation=cv2.INTER_AREA)
            score = calculate_ocr_score(candidate, language)
            if SHOW_OCR_SCORES:
                print(f'OCR候補 : rotation={rotation:3d}°, orientation={orientation:9s}, score={score:.2f}')
            if score > best_score:
                best_score = score
                best_rotation = rotation
                best_orientation = orientation
    return (best_rotation, best_orientation, best_score)

def enhance_document(image):
    """背景を白く整え、薄い文字を残して濃くする。強い二値化はしない。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), 25)
    background = np.maximum(background, 1)
    normalized = cv2.divide(gray, np.maximum(background, 1), scale=255)
    low = 0.0
    high = 245.0
    scan = np.clip((normalized.astype(np.float32) - low) * 255 / (high - low), 0, 255)
    scan = np.uint8(np.clip(255 * (scan / 255) ** 1.8, 0, 255))
    blur = cv2.GaussianBlur(scan, (0, 0), 0.8)
    scan = cv2.addWeighted(scan, 1.35, blur, -0.35, 0)
    return clean_outer_border(scan)

def clean_outer_border(image):
    """外周0.6%以内の、辺に沿って長く続く暗い縁だけを白くする。"""
    result = image.copy()
    (h, w) = result.shape
    for view in (result, result[::-1, :], result.T, result.T[::-1, :]):
        band = max(1, round(view.shape[0] * 0.006))
        for i in range(band):
            dark = view[i] < 245
            if i < max(1, round(view.shape[0] * 0.0015)) or dark.mean() >= 0.3:
                view[i] = 255
    return result


def refine_document_corners(contour, corners):
    """輪郭の各辺を直線近似し、その交点で四隅を精密化する。"""
    rect = order_points(corners)
    points = contour.reshape(-1, 2).astype(np.float32)
    lines = []
    for i in range(4):
        a, b = rect[i], rect[(i + 1) % 4]
        direction = b - a
        length = np.linalg.norm(direction)
        if length < 20:
            return rect
        unit = direction / length
        relative = points - a
        along = relative @ unit
        distance = np.abs(relative[:, 0] * unit[1] - relative[:, 1] * unit[0])
        selected = points[(distance < max(3, length * 0.008)) &
                          (along > length * 0.08) & (along < length * 0.92)]
        if len(selected) < 12:
            return rect
        vx, vy, x, y = cv2.fitLine(selected, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x, y]), np.array([vx, vy])))
    refined = []
    for i in range(4):
        a, u = lines[(i - 1) % 4]
        b, v = lines[i]
        matrix = np.column_stack((u, -v))
        if abs(np.linalg.det(matrix)) < 0.1:
            return rect
        t = np.linalg.solve(matrix, b - a)[0]
        refined.append(a + t * u)
    refined = np.array(refined, dtype=np.float32)
    if np.max(np.linalg.norm(refined - rect, axis=1)) > np.min(
            np.linalg.norm(np.roll(rect, -1, axis=0) - rect, axis=1)) * 0.03:
        return rect
    return refined

def estimate_skew(image):
    """長い横罫線から小さな残留傾きを推定。十分な線がなければ補正しない。"""
    gray = enhance_document(image)
    h, w = gray.shape
    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=60,
                           minLineLength=int(w * 0.35), maxLineGap=20)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= 2 and min(y1, y2) > h * 0.05 and max(y1, y2) < h * 0.95:
            angles.append(angle)
    if len(angles) < 4:
        return 0.0
    median = float(np.median(angles))
    if np.median(np.abs(np.array(angles) - median)) > 0.3:
        return 0.0
    return median if abs(median) > 0.05 else 0.0

def render_final(image, corners, rotation, size, angle):
    """元画像から最終サイズへ一度だけ補間して、文字のぼけを抑える。"""
    w, h = size
    rect = np.roll(order_points(corners), rotation // 90, axis=0)
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
    perspective = cv2.getPerspectiveTransform(rect, dst)
    # 紙の端に文字がある場合も切らないよう、回転分だけ縮小して収める。
    radians = np.radians(abs(angle))
    scale = min(w / (w*np.cos(radians) + h*np.sin(radians)),
                h / (h*np.cos(radians) + w*np.sin(radians)))
    affine = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, scale)
    matrix = np.vstack((affine, [0, 0, 1])) @ perspective
    return cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

def resize_for_display(image, max_width=1000, max_height=850):
    (height, width) = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return image
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def main():
    if not setup_tesseract():
        print('Tesseract OCR本体が見つかりません。')
        print('C:\\Program Files\\Tesseract-OCR\\tesseract.exe')
        print('などにTesseractをインストールしてください。')
        raise RuntimeError("処理を続行できません。直前のメッセージを確認してください。")
    OCR_LANGUAGE = get_ocr_language()
    if OCR_LANGUAGE is None:
        print('OCR用言語データがありません。')
        raise RuntimeError("処理を続行できません。直前のメッセージを確認してください。")
    print(f'OCR language : {OCR_LANGUAGE}')
    img = cv2.imdecode(np.fromfile(INPUT_FILE, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f'{INPUT_FILE} を読み込めませんでした。')
        raise RuntimeError("処理を続行できません。直前のメッセージを確認してください。")
    (height, width) = img.shape[:2]
    image_area = width * height
    image_diagonal = np.sqrt(width ** 2 + height ** 2)
    guide_x1 = int(width * GUIDE_MARGIN_X)
    guide_y1 = int(height * GUIDE_MARGIN_Y)
    guide_x2 = int(width * (1 - GUIDE_MARGIN_X))
    guide_y2 = int(height * (1 - GUIDE_MARGIN_Y))
    guide_area = (guide_x2 - guide_x1) * (guide_y2 - guide_y1)
    guide_center = np.array([(guide_x1 + guide_x2) / 2, (guide_y1 + guide_y2) / 2], dtype='float32')
    guide_corners = np.array([[guide_x1, guide_y1], [guide_x2, guide_y1], [guide_x2, guide_y2], [guide_x1, guide_y2]], dtype='float32')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edge_closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=1)
    (contours, _) = cv2.findContours(edge_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    document = None
    best_document_score = -1
    best_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        if area_ratio < MIN_AREA_RATIO:
            continue
        if area_ratio > MAX_AREA_RATIO:
            continue
        approx = approximate_quadrilateral(contour)
        if approx is None:
            continue
        rect = order_points(approx)
        center = np.mean(rect, axis=0)
        if not (guide_x1 <= center[0] <= guide_x2 and guide_y1 <= center[1] <= guide_y2):
            continue
        score = calculate_document_score(approx, area, guide_corners, guide_area, guide_center, image_diagonal)
        if score > best_document_score:
            document = approx.copy()
            best_contour = contour.copy()
            best_document_score = score
            best_area = area
    if document is None:
        print('書類を検出できませんでした。')
        cv2.destroyAllWindows()
        raise RuntimeError("処理を続行できません。直前のメッセージを確認してください。")
    document = refine_document_corners(best_contour, document)
    print('書類を検出しました。')
    print(f'Detection score : {best_document_score:.3f}')
    print(f'Area ratio : {best_area / image_area:.3f}')
    if DEBUG_PREVIEW:
        debug_image = img.copy()
        cv2.rectangle(debug_image, (guide_x1, guide_y1), (guide_x2, guide_y2), (255, 0, 0), 3)
        cv2.drawContours(debug_image, [document], -1, (0, 255, 0), 4)
        ordered_document = order_points(document)
        for point in ordered_document:
            x = int(point[0])
            y = int(point[1])
            cv2.circle(debug_image, (x, y), 8, (0, 0, 255), -1)
        cv2.imshow('Edge', resize_for_display(edge_closed))
        cv2.imshow('Document Detection', resize_for_display(debug_image))
        cv2.waitKey(DEBUG_WAIT)
        cv2.destroyAllWindows()
    warped = four_point_transform(img, document)
    if warped is None:
        print('透視変換に失敗しました。')
        raise RuntimeError("処理を続行できません。直前のメッセージを確認してください。")
    print()
    print('文字方向と用紙方向を判定しています...')
    (best_rotation, best_orientation, best_ocr_score) = detect_best_orientation(warped, PAPER_SIZE, OCR_LANGUAGE)
    print()
    print('===== OCR判定結果 =====')
    print(f'回転 : {best_rotation}度')
    if best_orientation == 'portrait':
        print('用紙方向 : 縦')
    else:
        print('用紙方向 : 横')
    print(f'OCR score : {best_ocr_score:.2f}')
    oriented = rotate_image(warped, best_rotation)
    (target_width, target_height) = get_paper_pixel_size(PAPER_SIZE, DPI, best_orientation)
    paper_corrected = cv2.resize(oriented, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    angle = estimate_skew(paper_corrected)
    print(f'残留傾き補正 : {angle:.3f}度')
    paper_corrected = render_final(img, document, best_rotation, (target_width, target_height), angle)
    final_scan = enhance_document(paper_corrected)
    save_success, encoded = cv2.imencode('.jpg', final_scan, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if save_success:
        encoded.tofile(OUTPUT_FILE)
    if save_success:
        print()
        print('===== 保存完了 =====')
        print(f'保存先 : {OUTPUT_FILE}')
        print(f'用紙 : {PAPER_SIZE}')
        print('方向 : ' + ('縦' if best_orientation == 'portrait' else '横'))
        print(f'DPI設定 : {DPI}')
        print(f'画像サイズ : {final_scan.shape[1]} x {final_scan.shape[0]} px')
    else:
        print(f'{OUTPUT_FILE} の保存に失敗しました。')
    cv2.destroyAllWindows()
    if SHOW_FINAL_PREVIEW:
        preview = resize_for_display(final_scan)
        cv2.imshow('Final Scan', preview)
        print()
        print('完成画像を表示しています。')
        print('何かキーを押すと閉じます。')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
