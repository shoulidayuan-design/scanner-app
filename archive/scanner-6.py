import cv2
import numpy as np


# ==============================
# 4点の並び順を整理
# ==============================
def order_points(pts):

    pts = pts.reshape(4, 2)

    rect = np.zeros((4, 2), dtype="float32")

    # x + y
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下

    # y - x
    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    return rect


# ==============================
# 透視変換
# ==============================
def four_point_transform(image, pts):

    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    # 横幅
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)

    maxWidth = max(
        int(widthA),
        int(widthB)
    )

    # 高さ
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxHeight = max(
        int(heightA),
        int(heightB)
    )

    # 異常なサイズを防止
    if maxWidth <= 0 or maxHeight <= 0:
        return None

    # 変換後の座標
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # 透視変換行列
    M = cv2.getPerspectiveTransform(
        rect,
        dst
    )

    # 透視変換
    warped = cv2.warpPerspective(
        image,
        M,
        (maxWidth, maxHeight)
    )

    return warped


# ==============================
# 画像読み込み
# ==============================
img = cv2.imread("document.jpg")


# 画像が読み込めなかった場合
if img is None:

    print("画像を読み込めませんでした。")
    print("document.jpg のファイル名や保存場所を確認してください。")

    exit()


# ==============================
# 撮影ガイド作成
# ==============================

height, width = img.shape[:2]


# ガイドの位置
# 画像の端から10%ずつ内側
guide_x1 = int(width * 0.10)
guide_y1 = int(height * 0.10)

guide_x2 = int(width * 0.90)
guide_y2 = int(height * 0.90)


# ガイドの中だけを切り出す
roi = img[
    guide_y1:guide_y2,
    guide_x1:guide_x2
]


# ==============================
# ROI内で書類を探す
# ==============================

# グレースケール化
gray = cv2.cvtColor(
    roi,
    cv2.COLOR_BGR2GRAY
)


# ノイズ除去
blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# エッジ検出
edge = cv2.Canny(
    blur,
    50,
    150
)


# ==============================
# 輪郭検出
# ==============================

contours, _ = cv2.findContours(
    edge,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# 検出した書類
document = None


# 最大面積
max_area = 0


# ROI全体の面積
roi_area = (
    roi.shape[0]
    * roi.shape[1]
)


# ==============================
# 四角形を探す
# ==============================

for c in contours:

    # 輪郭の面積
    area = cv2.contourArea(c)


    # --------------------------
    # 小さい輪郭を除外
    # --------------------------

    # ガイド領域の30%未満なら無視
    #
    # 書類内部の表・写真・枠などを
    # 誤認識しにくくする
    if area < roi_area * 0.30:
        continue


    # 輪郭の長さ
    peri = cv2.arcLength(
        c,
        True
    )


    # 輪郭を多角形に近似
    approx = cv2.approxPolyDP(
        c,
        0.02 * peri,
        True
    )


    # --------------------------
    # 4点か確認
    # --------------------------

    if len(approx) != 4:
        continue


    # --------------------------
    # 凸四角形か確認
    # --------------------------

    if not cv2.isContourConvex(approx):
        continue


    # --------------------------
    # 一番大きな四角形を採用
    # --------------------------

    if area > max_area:

        # ROI座標をコピー
        document_candidate = approx.copy()


        # ROI座標
        # ↓
        # 元画像の座標へ変換
        document_candidate[:, 0, 0] += guide_x1
        document_candidate[:, 0, 1] += guide_y1


        document = document_candidate

        max_area = area


# ==============================
# 表示用画像
# ==============================

result = img.copy()


# ==============================
# 撮影ガイド表示
# ==============================

cv2.rectangle(
    result,
    (guide_x1, guide_y1),
    (guide_x2, guide_y2),
    (255, 0, 0),
    3
)


# ガイド説明
cv2.putText(
    result,
    "Place document inside guide",
    (guide_x1, max(30, guide_y1 - 15)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 0, 0),
    2
)


# ==============================
# 書類が見つかった場合
# ==============================

if document is not None:

    # 書類輪郭を緑色で表示
    cv2.drawContours(
        result,
        [document],
        -1,
        (0, 255, 0),
        3
    )


    # ==========================
    # 透視変換
    # ==========================

    warped = four_point_transform(
        img,
        document
    )


    if warped is not None:

        # ======================
        # グレースケール
        # ======================

        scan_gray = cv2.cvtColor(
            warped,
            cv2.COLOR_BGR2GRAY
        )


        # ======================
        # 背景推定
        # ======================

        background = cv2.medianBlur(
            scan_gray,
            31
        )


        # ======================
        # 影除去
        # ======================

        shadow_removed = cv2.divide(
            scan_gray,
            background,
            scale=255
        )


        # ======================
        # 二値化
        # ======================

        scan = cv2.adaptiveThreshold(
            shadow_removed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )


        # ======================
        # 結果表示
        # ======================

        cv2.imshow(
            "Perspective Transform",
            warped
        )

        cv2.imshow(
            "Shadow Removed",
            shadow_removed
        )

        cv2.imshow(
            "Scanned",
            scan
        )


else:

    print("ガイド内から書類を検出できませんでした。")


# ==============================
# 確認用表示
# ==============================

cv2.imshow(
    "Edge",
    edge
)

cv2.imshow(
    "Document Detection",
    result
)


cv2.waitKey(0)

cv2.destroyAllWindows()