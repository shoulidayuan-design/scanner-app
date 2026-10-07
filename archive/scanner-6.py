import cv2
import numpy as np


# =========================================================
# 設定
# =========================================================

# 撮影ガイドの位置
# 画像端から10%内側
GUIDE_MARGIN_X = 0.10
GUIDE_MARGIN_Y = 0.10

# 書類候補として認める最低面積
# 画像全体の15%以上
MIN_AREA_RATIO = 0.15

# 画像全体とほぼ同じ大きさの輪郭は除外
MAX_AREA_RATIO = 0.95


# =========================================================
# 4点の並び順を整理
# =========================================================
def order_points(pts):

    pts = pts.reshape(4, 2).astype("float32")

    rect = np.zeros((4, 2), dtype="float32")

    # x + y
    s = pts.sum(axis=1)

    # 左上
    rect[0] = pts[np.argmin(s)]

    # 右下
    rect[2] = pts[np.argmax(s)]

    # y - x
    diff = np.diff(pts, axis=1).reshape(-1)

    # 右上
    rect[1] = pts[np.argmin(diff)]

    # 左下
    rect[3] = pts[np.argmax(diff)]

    return rect


# =========================================================
# 透視変換
# =========================================================
def four_point_transform(image, pts):

    rect = order_points(pts)

    tl, tr, br, bl = rect

    # -------------------------
    # 横幅
    # -------------------------

    widthA = np.linalg.norm(br - bl)

    widthB = np.linalg.norm(tr - tl)

    maxWidth = max(
        int(widthA),
        int(widthB)
    )

    # -------------------------
    # 高さ
    # -------------------------

    heightA = np.linalg.norm(tr - br)

    heightB = np.linalg.norm(tl - bl)

    maxHeight = max(
        int(heightA),
        int(heightB)
    )

    # サイズがおかしい場合
    if maxWidth <= 0 or maxHeight <= 0:
        return None

    # -------------------------
    # 変換後の座標
    # -------------------------

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # -------------------------
    # 透視変換行列
    # -------------------------

    M = cv2.getPerspectiveTransform(
        rect,
        dst
    )

    # -------------------------
    # 透視変換
    # -------------------------

    warped = cv2.warpPerspective(
        image,
        M,
        (maxWidth, maxHeight)
    )

    return warped


# =========================================================
# 輪郭を四角形へ近似
# =========================================================
def approximate_quadrilateral(contour):

    peri = cv2.arcLength(
        contour,
        True
    )

    # 少しずつ近似強度を変更する
    # 0.02だけで4点にならない場合への対策
    epsilon_values = [
        0.02,
        0.025,
        0.03
    ]

    for epsilon in epsilon_values:

        approx = cv2.approxPolyDP(
            contour,
            epsilon * peri,
            True
        )

        # 4点でなければ次へ
        if len(approx) != 4:
            continue

        # 凹んだ四角形は除外
        if not cv2.isContourConvex(approx):
            continue

        return approx

    return None


# =========================================================
# 書類候補の点数を計算
# =========================================================
def calculate_document_score(
    approx,
    area,
    guide_corners,
    guide_area,
    guide_center,
    image_diagonal
):

    # 四隅を
    # 左上、右上、右下、左下
    # の順番にする
    rect = order_points(approx)

    # -------------------------
    # 候補の中心
    # -------------------------

    document_center = np.mean(
        rect,
        axis=0
    )

    # -------------------------
    # 中心位置の評価
    # -------------------------

    center_distance = np.linalg.norm(
        document_center - guide_center
    )

    # 近いほど1
    # 遠いほど0
    center_score = 1.0 - min(
        center_distance / (image_diagonal * 0.35),
        1.0
    )

    # -------------------------
    # 四隅の位置を評価
    # -------------------------

    corner_distances = []

    for document_point, guide_point in zip(
        rect,
        guide_corners
    ):

        distance = np.linalg.norm(
            document_point - guide_point
        )

        corner_distances.append(
            distance
        )

    mean_corner_distance = np.mean(
        corner_distances
    )

    # ガイド四隅に近いほど高得点
    corner_score = 1.0 - min(
        mean_corner_distance
        / (image_diagonal * 0.35),
        1.0
    )

    # -------------------------
    # 大きさを評価
    # -------------------------

    size_difference = abs(
        area - guide_area
    ) / guide_area

    size_score = 1.0 - min(
        size_difference,
        1.0
    )

    # -------------------------
    # 総合点
    # -------------------------

    score = (
        corner_score * 0.45
        + size_score * 0.35
        + center_score * 0.20
    )

    return score


# =========================================================
# 画像読み込み
# =========================================================

img = cv2.imread(
    "document.jpg"
)


# 読み込み失敗
if img is None:

    print(
        "画像を読み込めませんでした。"
    )

    print(
        "document.jpg のファイル名や保存場所を確認してください。"
    )

    exit()


# =========================================================
# 画像サイズ
# =========================================================

height, width = img.shape[:2]

image_area = (
    width * height
)

image_diagonal = np.sqrt(
    width ** 2
    + height ** 2
)


# =========================================================
# 撮影ガイド
# =========================================================

guide_x1 = int(
    width * GUIDE_MARGIN_X
)

guide_y1 = int(
    height * GUIDE_MARGIN_Y
)

guide_x2 = int(
    width * (1.0 - GUIDE_MARGIN_X)
)

guide_y2 = int(
    height * (1.0 - GUIDE_MARGIN_Y)
)


# ガイドの面積
guide_area = (
    (guide_x2 - guide_x1)
    *
    (guide_y2 - guide_y1)
)


# ガイド中央
guide_center = np.array([
    (guide_x1 + guide_x2) / 2,
    (guide_y1 + guide_y2) / 2
], dtype="float32")


# ガイドの四隅
#
# 左上
# 右上
# 右下
# 左下
guide_corners = np.array([
    [guide_x1, guide_y1],
    [guide_x2, guide_y1],
    [guide_x2, guide_y2],
    [guide_x1, guide_y2]
], dtype="float32")


# =========================================================
# グレースケール化
# =========================================================

# 注意
#
# 以前はガイド内だけをROIとして
# 処理していた。
#
# 今回は画像全体を使用する。

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)


# =========================================================
# ノイズ除去
# =========================================================

blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# =========================================================
# エッジ検出
# =========================================================

edge = cv2.Canny(
    blur,
    50,
    150
)


# =========================================================
# 切れたエッジを多少つなげる
# =========================================================

kernel = np.ones(
    (3, 3),
    np.uint8
)

edge_closed = cv2.morphologyEx(
    edge,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1
)


# =========================================================
# 輪郭検出
# =========================================================

contours, _ = cv2.findContours(
    edge_closed,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)


# =========================================================
# 書類候補探索
# =========================================================

document = None

best_score = -1

best_area = 0


for contour in contours:

    # -------------------------
    # 面積
    # -------------------------

    area = cv2.contourArea(
        contour
    )

    area_ratio = (
        area / image_area
    )

    # -------------------------
    # 小さすぎる輪郭を除外
    # -------------------------

    if area_ratio < MIN_AREA_RATIO:
        continue

    # -------------------------
    # 大きすぎる輪郭を除外
    # -------------------------

    if area_ratio > MAX_AREA_RATIO:
        continue

    # -------------------------
    # 四角形に近似
    # -------------------------

    approx = approximate_quadrilateral(
        contour
    )

    if approx is None:
        continue

    # -------------------------
    # 四角形の中心
    # -------------------------

    rect = order_points(
        approx
    )

    center = np.mean(
        rect,
        axis=0
    )

    # -------------------------
    # 中心がガイド周辺にないものを除外
    # -------------------------

    # 書類の中心がガイドの中なら候補にする
    if not (
        guide_x1 <= center[0] <= guide_x2
        and
        guide_y1 <= center[1] <= guide_y2
    ):
        continue

    # -------------------------
    # 書類らしさを点数化
    # -------------------------

    score = calculate_document_score(
        approx,
        area,
        guide_corners,
        guide_area,
        guide_center,
        image_diagonal
    )

    # -------------------------
    # 最も点数の高いものを採用
    # -------------------------

    if score > best_score:

        document = approx.copy()

        best_score = score

        best_area = area


# =========================================================
# 表示用画像
# =========================================================

result = img.copy()


# =========================================================
# 撮影ガイドを表示
# =========================================================

cv2.rectangle(
    result,
    (guide_x1, guide_y1),
    (guide_x2, guide_y2),
    (255, 0, 0),
    3
)


# =========================================================
# ガイド説明
# =========================================================

text_y = guide_y1 - 15

if text_y < 30:
    text_y = guide_y1 + 30


cv2.putText(
    result,
    "Place document inside guide",
    (guide_x1, text_y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 0, 0),
    2
)


# =========================================================
# 書類を検出できた場合
# =========================================================

if document is not None:

    print(
        "書類を検出しました。"
    )

    print(
        f"Score : {best_score:.3f}"
    )

    print(
        f"Area ratio : {best_area / image_area:.3f}"
    )

    # -------------------------
    # 検出した書類を緑色で表示
    # -------------------------

    cv2.drawContours(
        result,
        [document],
        -1,
        (0, 255, 0),
        4
    )

    # -------------------------
    # 四隅にも印を付ける
    # -------------------------

    ordered_document = order_points(
        document
    )

    for point in ordered_document:

        x = int(
            point[0]
        )

        y = int(
            point[1]
        )

        cv2.circle(
            result,
            (x, y),
            8,
            (0, 0, 255),
            -1
        )

    # =====================================================
    # 透視変換
    # =====================================================

    warped = four_point_transform(
        img,
        document
    )

    if warped is not None:

        # =================================================
        # グレースケール化
        # =================================================

        scan_gray = cv2.cvtColor(
            warped,
            cv2.COLOR_BGR2GRAY
        )

        # =================================================
        # 背景推定
        # =================================================

        background = cv2.medianBlur(
            scan_gray,
            31
        )

        # =================================================
        # 影除去
        # =================================================

        shadow_removed = cv2.divide(
            scan_gray,
            background,
            scale=255
        )

        # =================================================
        # 二値化
        # =================================================

        scan = cv2.adaptiveThreshold(
            shadow_removed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        # =================================================
        # 結果表示
        # =================================================

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


# =========================================================
# 書類を検出できなかった場合
# =========================================================

else:

    print(
        "書類を検出できませんでした。"
    )


# =========================================================
# 確認用
# =========================================================

cv2.imshow(
    "Edge",
    edge_closed
)

cv2.imshow(
    "Document Detection",
    result
)


# =========================================================
# キー入力待ち
# =========================================================

cv2.waitKey(0)

cv2.destroyAllWindows()