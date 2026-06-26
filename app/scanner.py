import cv2
import numpy as np
from pathlib import Path


# ====== 入力・出力ファイルの設定 ======
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "document.jpg"
OUTPUT_PATH = BASE_DIR / "images" / "after.jpg"


def load_image(path):
    """画像を読み込む。日本語パスでも読み込めるようにする。"""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"画像ファイルが存在しません: {path}")

    image_array = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {path}")

    return image


def resize_image(image, height=800):
    """処理しやすい大きさに画像を縮小する。"""
    ratio = height / image.shape[0]
    width = int(image.shape[1] * ratio)
    resized = cv2.resize(image, (width, height))
    return resized, ratio


def order_points(points):
    """四隅の点を 左上・右上・右下・左下 の順番に並べる。"""
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]  # 左上
    rect[2] = points[np.argmax(s)]  # 右下

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]  # 右上
    rect[3] = points[np.argmax(diff)]  # 左下

    return rect


def four_point_transform(image, points):
    """検出した四隅を使って、斜めの書類を正面から見た形に補正する。"""
    rect = order_points(points)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


def detect_document(image):
    """画像の中から書類らしい四角形を検出する。"""
    resized, ratio = resize_image(image)
    original_resized = resized.copy()

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)

        # 小さすぎる輪郭は書類ではない可能性が高いので除外する
        if area < 10000:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # 四角形なら書類として扱う
        if len(approx) == 4:
            points = approx.reshape(4, 2)

            # 縮小した画像で検出した点を、元画像サイズに戻す
            points = points / ratio

            cv2.drawContours(original_resized, [approx], -1, (0, 255, 0), 2)
            cv2.imshow("Document Detection", original_resized)

            return points

    return None


def remove_shadow(image):
    """影や照明ムラを軽減する。"""
    rgb_planes = cv2.split(image)
    result_planes = []

    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        background = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, background)
        normalized = cv2.normalize(
            diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
        result_planes.append(normalized)

    result = cv2.merge(result_planes)
    return result


def create_scan(image):
    """カラー画像をスキャン画像のような白黒画像に変換する。"""
    shadow_removed = remove_shadow(image)

    gray = cv2.cvtColor(shadow_removed, cv2.COLOR_BGR2GRAY)

    scan = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10,
    )

    return shadow_removed, scan


def main():
    """アプリ全体の実行処理。"""
    image = load_image(INPUT_PATH)

    document_points = detect_document(image)

    if document_points is None:
        print("書類が検出できませんでした。元画像をそのまま処理します。")
        warped = image
    else:
        warped = four_point_transform(image, document_points)

    shadow_removed, scan = create_scan(warped)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), scan)

    cv2.imshow("Shadow Removed", shadow_removed)
    cv2.imshow("Warped", warped)
    cv2.imshow("Scanned", scan)

    print(f"スキャン画像を保存しました: {OUTPUT_PATH}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()