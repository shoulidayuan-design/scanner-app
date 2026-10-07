import cv2
import numpy as np
import pytesseract

import os
import shutil
import sys


# =========================================================
# 設定
# =========================================================

# 入力画像
INPUT_FILE = "document.jpg"

# 出力画像
OUTPUT_FILE = "document_1.jpg"


# ---------------------------------------------------------
# 用紙サイズ
#
# "A4"
# "B5"
#
# 縦・横はOCRで自動判定
# ---------------------------------------------------------
PAPER_SIZE = "A4"


# ---------------------------------------------------------
# 最終出力DPI
#
# 150 : 軽量
# 200 : おすすめ
# 300 : 高画質
# ---------------------------------------------------------
DPI = 200


# ---------------------------------------------------------
# 最終画像を表示するか
# ---------------------------------------------------------
SHOW_FINAL_PREVIEW = True


# ---------------------------------------------------------
# デバッグ表示
#
# False:
#   途中ウィンドウを表示しない
#
# True:
#   エッジと書類検出結果を一時表示
# ---------------------------------------------------------
DEBUG_PREVIEW = False

DEBUG_WAIT = 1200


# ---------------------------------------------------------
# OCR候補ごとの結果をコンソール表示するか
# ---------------------------------------------------------
SHOW_OCR_SCORES = True


# ---------------------------------------------------------
# 撮影ガイド
# ---------------------------------------------------------
GUIDE_MARGIN_X = 0.10
GUIDE_MARGIN_Y = 0.10


# ---------------------------------------------------------
# 書類候補の面積
# ---------------------------------------------------------
MIN_AREA_RATIO = 0.15
MAX_AREA_RATIO = 0.95


# =========================================================
# 用紙サイズ
# 単位：mm
# =========================================================

PAPER_SIZES_MM = {
    "A4": (210, 297),
    "B5": (182, 257)
}


# =========================================================
# Tesseract設定
# =========================================================
def setup_tesseract():

    # PATHから検索
    path = shutil.which("tesseract")

    if path is not None:

        pytesseract.pytesseract.tesseract_cmd = path

        return True


    # Windowsの一般的な場所
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ]


    for path in common_paths:

        if os.path.exists(path):

            pytesseract.pytesseract.tesseract_cmd = path

            return True


    return False


# =========================================================
# OCR言語を決定
# =========================================================
def get_ocr_language():

    try:

        languages = pytesseract.get_languages(
            config=""
        )

    except Exception:

        languages = []


    print(
        "Tesseract languages :",
        languages
    )


    # 日本語 + 英語
    if (
        "jpn" in languages
        and
        "eng" in languages
    ):

        return "jpn+eng"


    # 日本語だけ
    if "jpn" in languages:

        return "jpn"


    # 英語だけ
    if "eng" in languages:

        print(
            "警告：日本語OCRデータ jpn がありません。"
        )

        print(
            "日本語書類の方向判定精度が低下する可能性があります。"
        )

        return "eng"


    return None


# =========================================================
# 4点の順番を整理
#
# 左上
# 右上
# 右下
# 左下
# =========================================================
def order_points(pts):

    pts = pts.reshape(
        4,
        2
    ).astype(
        "float32"
    )


    rect = np.zeros(
        (4, 2),
        dtype="float32"
    )


    # x + y
    s = pts.sum(
        axis=1
    )


    # 左上
    rect[0] = pts[
        np.argmin(s)
    ]


    # 右下
    rect[2] = pts[
        np.argmax(s)
    ]


    # y - x
    diff = np.diff(
        pts,
        axis=1
    ).reshape(-1)


    # 右上
    rect[1] = pts[
        np.argmin(diff)
    ]


    # 左下
    rect[3] = pts[
        np.argmax(diff)
    ]


    return rect


# =========================================================
# 仮透視変換
#
# この時点ではA4/B5の比率を強制しない。
# 撮影された書類を長方形にするだけ。
# =========================================================
def four_point_transform(
    image,
    pts
):

    rect = order_points(
        pts
    )


    tl, tr, br, bl = rect


    # -----------------------------------------------------
    # 幅
    # -----------------------------------------------------

    width_a = np.linalg.norm(
        br - bl
    )

    width_b = np.linalg.norm(
        tr - tl
    )


    max_width = max(
        int(width_a),
        int(width_b)
    )


    # -----------------------------------------------------
    # 高さ
    # -----------------------------------------------------

    height_a = np.linalg.norm(
        tr - br
    )

    height_b = np.linalg.norm(
        tl - bl
    )


    max_height = max(
        int(height_a),
        int(height_b)
    )


    if (
        max_width <= 0
        or
        max_height <= 0
    ):

        return None


    # -----------------------------------------------------
    # 変換先
    # -----------------------------------------------------

    dst = np.array(
        [
            [0, 0],

            [
                max_width - 1,
                0
            ],

            [
                max_width - 1,
                max_height - 1
            ],

            [
                0,
                max_height - 1
            ]
        ],
        dtype="float32"
    )


    matrix = cv2.getPerspectiveTransform(
        rect,
        dst
    )


    warped = cv2.warpPerspective(
        image,
        matrix,
        (
            max_width,
            max_height
        )
    )


    return warped


# =========================================================
# 輪郭 → 四角形
# =========================================================
def approximate_quadrilateral(
    contour
):

    peri = cv2.arcLength(
        contour,
        True
    )


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


        # 4点のみ
        if len(approx) != 4:
            continue


        # 凸四角形のみ
        if not cv2.isContourConvex(
            approx
        ):

            continue


        return approx


    return None


# =========================================================
# 書類候補の採点
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


    # -----------------------------------------------------
    # 中心
    # -----------------------------------------------------

    document_center = np.mean(
        rect,
        axis=0
    )


    center_distance = np.linalg.norm(
        document_center
        -
        guide_center
    )


    center_score = (
        1.0
        -
        min(
            center_distance
            /
            (
                image_diagonal
                *
                0.35
            ),
            1.0
        )
    )


    # -----------------------------------------------------
    # 四隅
    # -----------------------------------------------------

    corner_distances = []


    for document_point, guide_point in zip(
        rect,
        guide_corners
    ):

        distance = np.linalg.norm(
            document_point
            -
            guide_point
        )


        corner_distances.append(
            distance
        )


    mean_corner_distance = np.mean(
        corner_distances
    )


    corner_score = (
        1.0
        -
        min(
            mean_corner_distance
            /
            (
                image_diagonal
                *
                0.35
            ),
            1.0
        )
    )


    # -----------------------------------------------------
    # 面積
    # -----------------------------------------------------

    size_difference = abs(
        area
        -
        guide_area
    ) / guide_area


    size_score = (
        1.0
        -
        min(
            size_difference,
            1.0
        )
    )


    # -----------------------------------------------------
    # 総合点
    # -----------------------------------------------------

    score = (
        corner_score
        *
        0.45

        +

        size_score
        *
        0.35

        +

        center_score
        *
        0.20
    )


    return score


# =========================================================
# 画像を回転
# =========================================================
def rotate_image(
    image,
    rotation
):

    if rotation == 90:

        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )


    if rotation == 180:

        return cv2.rotate(
            image,
            cv2.ROTATE_180
        )


    if rotation == 270:

        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )


    return image.copy()


# =========================================================
# mm → px
# =========================================================
def mm_to_px(
    mm,
    dpi
):

    return int(
        round(
            mm
            /
            25.4
            *
            dpi
        )
    )


# =========================================================
# 用紙サイズ取得
# =========================================================
def get_paper_pixel_size(
    paper,
    dpi,
    orientation
):

    if paper not in PAPER_SIZES_MM:

        raise ValueError(
            "PAPER_SIZE は A4 または B5 を指定してください。"
        )


    width_mm, height_mm = (
        PAPER_SIZES_MM[
            paper
        ]
    )


    width = mm_to_px(
        width_mm,
        dpi
    )

    height = mm_to_px(
        height_mm,
        dpi
    )


    # 縦
    if orientation == "portrait":

        return (
            width,
            height
        )


    # 横
    return (
        height,
        width
    )


# =========================================================
# OCR判定用サイズ
#
# 200dpiのまま8回OCRすると重いため、
# 判定時は低解像度を使用する。
# =========================================================
def get_ocr_candidate_size(
    paper,
    orientation
):

    # 約110 DPI相当
    OCR_DPI = 110


    return get_paper_pixel_size(
        paper,
        OCR_DPI,
        orientation
    )


# =========================================================
# OCR用画像を作成
# =========================================================
def prepare_for_ocr(
    image
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # 軽いノイズ除去
    gray = cv2.GaussianBlur(
        gray,
        (
            3,
            3
        ),
        0
    )


    # Otsu二値化
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        +
        cv2.THRESH_OTSU
    )


    return binary


# =========================================================
# OCRスコア計算
#
# OCR confidenceだけでなく、
# 認識できた文字数も評価する。
# =========================================================
def calculate_ocr_score(
    image,
    language
):

    prepared = prepare_for_ocr(
        image
    )


    try:

        data = pytesseract.image_to_data(
            prepared,
            lang=language,
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT
        )

    except Exception as error:

        print(
            "OCRエラー :",
            error
        )

        return -1


    weighted_conf_sum = 0.0

    weight_sum = 0

    character_count = 0

    detected_items = 0


    for text, confidence in zip(
        data["text"],
        data["conf"]
    ):

        text = text.strip()


        if not text:
            continue


        try:

            confidence = float(
                confidence
            )

        except ValueError:

            continue


        if confidence < 0:
            continue


        # 文字数を重みとして使用
        char_length = max(
            len(text),
            1
        )


        weighted_conf_sum += (
            confidence
            *
            char_length
        )


        weight_sum += char_length

        character_count += char_length

        detected_items += 1


    if weight_sum == 0:

        return 0


    # -----------------------------------------------------
    # 平均confidence
    # -----------------------------------------------------

    average_confidence = (
        weighted_conf_sum
        /
        weight_sum
    )


    # -----------------------------------------------------
    # 文字数評価
    #
    # 1〜120文字まで加点
    # -----------------------------------------------------

    character_bonus = (
        min(
            character_count,
            120
        )
        /
        120
        *
        15
    )


    # -----------------------------------------------------
    # 認識された項目数
    # -----------------------------------------------------

    item_bonus = (
        min(
            detected_items,
            30
        )
        /
        30
        *
        5
    )


    # -----------------------------------------------------
    # 最終スコア
    # -----------------------------------------------------

    score = (
        average_confidence
        *
        0.80

        +

        character_bonus

        +

        item_bonus
    )


    return score


# =========================================================
# OCRで向き + 用紙方向を自動判定
#
# 4回転 × 縦横
# = 8候補
# =========================================================
def detect_best_orientation(
    warped,
    paper,
    language
):

    rotations = [
        0,
        90,
        180,
        270
    ]


    orientations = [
        "portrait",
        "landscape"
    ]


    best_score = -1

    best_rotation = 0

    best_orientation = "portrait"


    for rotation in rotations:

        rotated = rotate_image(
            warped,
            rotation
        )


        for orientation in orientations:

            candidate_width, candidate_height = (
                get_ocr_candidate_size(
                    paper,
                    orientation
                )
            )


            # -------------------------------------------------
            # この段階で用紙比率へ変換
            #
            # 正しい比率なら文字形状も自然になる。
            # 間違った比率なら文字が縦/横に潰れる。
            # -------------------------------------------------

            candidate = cv2.resize(
                rotated,
                (
                    candidate_width,
                    candidate_height
                ),
                interpolation=cv2.INTER_AREA
            )


            score = calculate_ocr_score(
                candidate,
                language
            )


            if SHOW_OCR_SCORES:

                print(
                    "OCR候補 : "
                    f"rotation={rotation:3d}°, "
                    f"orientation={orientation:9s}, "
                    f"score={score:.2f}"
                )


            if score > best_score:

                best_score = score

                best_rotation = rotation

                best_orientation = orientation


    return (
        best_rotation,
        best_orientation,
        best_score
    )


# =========================================================
# 画質補正
# =========================================================
def enhance_document(
    image
):

    # -----------------------------------------------------
    # グレースケール
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # -----------------------------------------------------
    # ノイズ除去
    # -----------------------------------------------------

    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=5,
        templateWindowSize=7,
        searchWindowSize=21
    )


    # -----------------------------------------------------
    # 背景・影を推定
    # -----------------------------------------------------

    background = cv2.GaussianBlur(
        denoised,
        (
            0,
            0
        ),
        25
    )


    # -----------------------------------------------------
    # 影除去
    # -----------------------------------------------------

    shadow_removed = cv2.divide(
        denoised,
        background,
        scale=255
    )


    # -----------------------------------------------------
    # コントラスト
    # -----------------------------------------------------

    scan = cv2.convertScaleAbs(
        shadow_removed,
        alpha=1.15,
        beta=-20
    )


    # -----------------------------------------------------
    # 軽くシャープ化
    # -----------------------------------------------------

    blur_for_sharp = cv2.GaussianBlur(
        scan,
        (
            0,
            0
        ),
        1.0
    )


    final_scan = cv2.addWeighted(
        scan,
        1.4,
        blur_for_sharp,
        -0.4,
        0
    )


    return final_scan


# =========================================================
# 表示用縮小
# =========================================================
def resize_for_display(
    image,
    max_width=1000,
    max_height=850
):

    height, width = image.shape[:2]


    scale = min(
        max_width / width,
        max_height / height,
        1.0
    )


    if scale >= 1.0:

        return image


    new_width = int(
        width
        *
        scale
    )


    new_height = int(
        height
        *
        scale
    )


    return cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )


# =========================================================
# Tesseract確認
# =========================================================

if not setup_tesseract():

    print(
        "Tesseract OCR本体が見つかりません。"
    )

    print(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    print(
        "などにTesseractをインストールしてください。"
    )

    sys.exit()


OCR_LANGUAGE = get_ocr_language()


if OCR_LANGUAGE is None:

    print(
        "OCR用言語データがありません。"
    )

    sys.exit()


print(
    f"OCR language : {OCR_LANGUAGE}"
)


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

    sys.exit()


# =========================================================
# 画像情報
# =========================================================

height, width = img.shape[:2]


image_area = (
    width
    *
    height
)


image_diagonal = np.sqrt(
    width ** 2
    +
    height ** 2
)


# =========================================================
# ガイド
# =========================================================

guide_x1 = int(
    width
    *
    GUIDE_MARGIN_X
)


guide_y1 = int(
    height
    *
    GUIDE_MARGIN_Y
)


guide_x2 = int(
    width
    *
    (
        1
        -
        GUIDE_MARGIN_X
    )
)


guide_y2 = int(
    height
    *
    (
        1
        -
        GUIDE_MARGIN_Y
    )
)


guide_area = (

    (
        guide_x2
        -
        guide_x1
    )

    *

    (
        guide_y2
        -
        guide_y1
    )
)


guide_center = np.array(
    [
        (
            guide_x1
            +
            guide_x2
        )
        /
        2,

        (
            guide_y1
            +
            guide_y2
        )
        /
        2
    ],
    dtype="float32"
)


guide_corners = np.array(
    [
        [
            guide_x1,
            guide_y1
        ],

        [
            guide_x2,
            guide_y1
        ],

        [
            guide_x2,
            guide_y2
        ],

        [
            guide_x1,
            guide_y2
        ]
    ],
    dtype="float32"
)


# =========================================================
# 書類検出
# =========================================================

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)


blur = cv2.GaussianBlur(
    gray,
    (
        5,
        5
    ),
    0
)


edge = cv2.Canny(
    blur,
    50,
    150
)


kernel = np.ones(
    (
        3,
        3
    ),
    np.uint8
)


edge_closed = cv2.morphologyEx(
    edge,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1
)


contours, _ = cv2.findContours(
    edge_closed,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)


# =========================================================
# 書類候補探索
# =========================================================

document = None

best_document_score = -1

best_area = 0


for contour in contours:

    area = cv2.contourArea(
        contour
    )


    area_ratio = (
        area
        /
        image_area
    )


    if area_ratio < MIN_AREA_RATIO:
        continue


    if area_ratio > MAX_AREA_RATIO:
        continue


    approx = approximate_quadrilateral(
        contour
    )


    if approx is None:
        continue


    rect = order_points(
        approx
    )


    center = np.mean(
        rect,
        axis=0
    )


    # 中心がガイド内にあるか
    if not (
        guide_x1 <= center[0] <= guide_x2
        and
        guide_y1 <= center[1] <= guide_y2
    ):

        continue


    score = calculate_document_score(
        approx,
        area,
        guide_corners,
        guide_area,
        guide_center,
        image_diagonal
    )


    if score > best_document_score:

        document = approx.copy()

        best_document_score = score

        best_area = area


# =========================================================
# 書類未検出
# =========================================================

if document is None:

    print(
        "書類を検出できませんでした。"
    )

    cv2.destroyAllWindows()

    sys.exit()


print(
    "書類を検出しました。"
)


print(
    f"Detection score : "
    f"{best_document_score:.3f}"
)


print(
    f"Area ratio : "
    f"{best_area / image_area:.3f}"
)


# =========================================================
# デバッグ表示
# =========================================================

if DEBUG_PREVIEW:

    debug_image = img.copy()


    # ガイド
    cv2.rectangle(
        debug_image,
        (
            guide_x1,
            guide_y1
        ),
        (
            guide_x2,
            guide_y2
        ),
        (
            255,
            0,
            0
        ),
        3
    )


    # 書類
    cv2.drawContours(
        debug_image,
        [
            document
        ],
        -1,
        (
            0,
            255,
            0
        ),
        4
    )


    # 四隅
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
            debug_image,
            (
                x,
                y
            ),
            8,
            (
                0,
                0,
                255
            ),
            -1
        )


    cv2.imshow(
        "Edge",
        resize_for_display(
            edge_closed
        )
    )


    cv2.imshow(
        "Document Detection",
        resize_for_display(
            debug_image
        )
    )


    cv2.waitKey(
        DEBUG_WAIT
    )


    # 途中のウィンドウを閉じる
    cv2.destroyAllWindows()


# =========================================================
# 仮透視変換
# =========================================================

warped = four_point_transform(
    img,
    document
)


if warped is None:

    print(
        "透視変換に失敗しました。"
    )

    sys.exit()


# =========================================================
# OCRによる向き自動判定
# =========================================================

print()
print(
    "文字方向と用紙方向を判定しています..."
)


(
    best_rotation,
    best_orientation,
    best_ocr_score
) = detect_best_orientation(
    warped,
    PAPER_SIZE,
    OCR_LANGUAGE
)


print()
print(
    "===== OCR判定結果 ====="
)


print(
    f"回転 : {best_rotation}度"
)


if best_orientation == "portrait":

    print(
        "用紙方向 : 縦"
    )

else:

    print(
        "用紙方向 : 横"
    )


print(
    f"OCR score : "
    f"{best_ocr_score:.2f}"
)


# =========================================================
# 正しい方向へ回転
# =========================================================

oriented = rotate_image(
    warped,
    best_rotation
)


# =========================================================
# 正しいA4/B5サイズへ整形
#
# OCRで決定した方向を使用
# =========================================================

target_width, target_height = (
    get_paper_pixel_size(
        PAPER_SIZE,
        DPI,
        best_orientation
    )
)


paper_corrected = cv2.resize(
    oriented,
    (
        target_width,
        target_height
    ),
    interpolation=cv2.INTER_CUBIC
)


# =========================================================
# 影除去・画質補正
# =========================================================

final_scan = enhance_document(
    paper_corrected
)


# =========================================================
# 保存
# =========================================================

save_success = cv2.imwrite(
    OUTPUT_FILE,
    final_scan,
    [
        cv2.IMWRITE_JPEG_QUALITY,
        95
    ]
)


if save_success:

    print()
    print(
        "===== 保存完了 ====="
    )


    print(
        f"保存先 : {OUTPUT_FILE}"
    )


    print(
        f"用紙 : {PAPER_SIZE}"
    )


    print(
        "方向 : "
        +
        (
            "縦"
            if best_orientation == "portrait"
            else "横"
        )
    )


    print(
        f"DPI設定 : {DPI}"
    )


    print(
        "画像サイズ : "
        f"{final_scan.shape[1]}"
        " x "
        f"{final_scan.shape[0]}"
        " px"
    )


else:

    print(
        f"{OUTPUT_FILE} の保存に失敗しました。"
    )


# =========================================================
# 途中のOpenCVウィンドウを閉じる
# =========================================================

cv2.destroyAllWindows()


# =========================================================
# 最終画像だけ表示
# =========================================================

if SHOW_FINAL_PREVIEW:

    preview = resize_for_display(
        final_scan
    )


    cv2.imshow(
        "Final Scan",
        preview
    )


    print()
    print(
        "完成画像を表示しています。"
    )


    print(
        "何かキーを押すと閉じます。"
    )


    cv2.waitKey(0)


    cv2.destroyAllWindows()