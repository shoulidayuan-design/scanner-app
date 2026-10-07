import cv2
import numpy as np


# =========================================================
# 設定
# =========================================================

# 入力画像
INPUT_FILE = "document.jpg"

# 出力画像
OUTPUT_FILE = "document_1.jpg"

# 撮影ガイドの位置
GUIDE_MARGIN_X = 0.10
GUIDE_MARGIN_Y = 0.10

# 書類候補として認める面積
MIN_AREA_RATIO = 0.15
MAX_AREA_RATIO = 0.95


# =========================================================
# 4点の順番を整理
# 左上 → 右上 → 右下 → 左下
# =========================================================
def order_points(pts):

    pts = pts.reshape(4, 2).astype("float32")

    rect = np.zeros(
        (4, 2),
        dtype="float32"
    )

    # x + y
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下

    # y - x
    diff = np.diff(
        pts,
        axis=1
    ).reshape(-1)

    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    return rect


# =========================================================
# 透視変換
# =========================================================
def four_point_transform(image, pts):

    rect = order_points(pts)

    tl, tr, br, bl = rect

    # 横幅
    widthA = np.linalg.norm(
        br - bl
    )

    widthB = np.linalg.norm(
        tr - tl
    )

    maxWidth = max(
        int(widthA),
        int(widthB)
    )

    # 高さ
    heightA = np.linalg.norm(
        tr - br
    )

    heightB = np.linalg.norm(
        tl - bl
    )

    maxHeight = max(
        int(heightA),
        int(heightB)
    )

    # 異常なサイズなら終了
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


# =========================================================
# 輪郭を四角形へ近似
# =========================================================
def approximate_quadrilateral(contour):

    peri = cv2.arcLength(
        contour,
        True
    )

    # 0.02で4点にならない場合もあるため
    # 数段階試す
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

        # 凹んだ形は書類候補から除外
        if not cv2.isContourConvex(approx):
            continue

        return approx

    return None


# =========================================================
# 書類候補の点数計算
# =========================================================
def calculate_document_score(
    approx,
    area,
    guide_corners,
    guide_area,
    guide_center,
    image_diagonal
):

    rect = order_points(
        approx
    )

    # -------------------------
    # 書類候補の中心
    # -------------------------

    document_center = np.mean(
        rect,
        axis=0
    )

    # -------------------------
    # ガイド中心との距離
    # -------------------------

    center_distance = np.linalg.norm(
        document_center - guide_center
    )

    center_score = 1.0 - min(
        center_distance
        / (image_diagonal * 0.35),
        1.0
    )

    # -------------------------
    # ガイド四隅との距離
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

    corner_score = 1.0 - min(
        mean_corner_distance
        / (image_diagonal * 0.35),
        1.0
    )

    # -------------------------
    # ガイドとの面積差
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
        +
        size_score * 0.35
        +
        center_score * 0.20
    )

    return score


# =========================================================
# 画像読み込み
# =========================================================

img = cv2.imread(
    INPUT_FILE
)

if img is None:

    print(
        f"{INPUT_FILE} を読み込めませんでした。"
    )

    print(
        "ファイル名や保存場所を確認してください。"
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
    +
    height ** 2
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


# ガイド面積
guide_area = (
    (guide_x2 - guide_x1)
    *
    (guide_y2 - guide_y1)
)


# ガイド中心
guide_center = np.array([
    (guide_x1 + guide_x2) / 2,
    (guide_y1 + guide_y2) / 2
], dtype="float32")


# ガイド四隅
guide_corners = np.array([
    [guide_x1, guide_y1],
    [guide_x2, guide_y1],
    [guide_x2, guide_y2],
    [guide_x1, guide_y2]
], dtype="float32")


# =========================================================
# グレースケール
# =========================================================

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
# エッジの切れ目をつなぐ
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

    # 面積
    area = cv2.contourArea(
        contour
    )

    area_ratio = (
        area / image_area
    )


    # -------------------------
    # 小さすぎるものを除外
    # -------------------------

    if area_ratio < MIN_AREA_RATIO:
        continue


    # -------------------------
    # 大きすぎるものを除外
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
    # 候補の中心
    # -------------------------

    rect = order_points(
        approx
    )

    center = np.mean(
        rect,
        axis=0
    )


    # -------------------------
    # 中心がガイド内か確認
    # -------------------------

    if not (
        guide_x1 <= center[0] <= guide_x2
        and
        guide_y1 <= center[1] <= guide_y2
    ):
        continue


    # -------------------------
    # ガイドとの近さを点数化
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
    # 最も高得点の候補を採用
    # -------------------------

    if score > best_score:

        document = approx.copy()

        best_score = score

        best_area = area


# =========================================================
# 確認用画像
# =========================================================

result = img.copy()


# =========================================================
# 撮影ガイド表示
# =========================================================

cv2.rectangle(
    result,
    (guide_x1, guide_y1),
    (guide_x2, guide_y2),
    (255, 0, 0),
    3
)


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
# 書類検出成功
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


    # =====================================================
    # 検出輪郭を表示
    # =====================================================

    cv2.drawContours(
        result,
        [document],
        -1,
        (0, 255, 0),
        4
    )


    # =====================================================
    # 四隅を赤丸で表示
    # =====================================================

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
        # グレースケール
        # =================================================

        scan_gray = cv2.cvtColor(
            warped,
            cv2.COLOR_BGR2GRAY
        )


        # =================================================
        # 軽いノイズ除去
        # =================================================

        denoised = cv2.fastNlMeansDenoising(
            scan_gray,
            None,
            h=5,
            templateWindowSize=7,
            searchWindowSize=21
        )


        # =================================================
        # 背景推定
        #
        # 大きなぼかしによって
        # 照明ムラ・影を推定する
        # =================================================

        background = cv2.GaussianBlur(
            denoised,
            (0, 0),
            25
        )


        # =================================================
        # 影除去
        # =================================================

        shadow_removed = cv2.divide(
            denoised,
            background,
            scale=255
        )


        # =================================================
        # コントラスト調整
        #
        # 紙を白く保ちつつ、
        # 文字や罫線を少し濃くする
        # =================================================

        scan = cv2.convertScaleAbs(
            shadow_removed,
            alpha=1.15,
            beta=-20
        )


        # =================================================
        # 軽いシャープ化
        # =================================================

        blur_for_sharp = cv2.GaussianBlur(
            scan,
            (0, 0),
            1.0
        )

        final_scan = cv2.addWeighted(
            scan,
            1.4,
            blur_for_sharp,
            -0.4,
            0
        )


        # =================================================
        # ファイル保存
        # =================================================

        save_success = cv2.imwrite(
            OUTPUT_FILE,
            final_scan
        )


        if save_success:

            print(
                f"{OUTPUT_FILE} として保存しました。"
            )

        else:

            print(
                f"{OUTPUT_FILE} の保存に失敗しました。"
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
            "Final Scan",
            final_scan
        )


    else:

        print(
            "透視変換に失敗しました。"
        )


# =========================================================
# 書類検出失敗
# =========================================================

else:

    print(
        "書類を検出できませんでした。"
    )


# =========================================================
# 確認用表示
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