import cv2
import numpy as np
import os
import shutil

# =========================================================
# pytesseract
# =========================================================

try:
    import pytesseract
    from pytesseract import Output

    PYTESSERACT_AVAILABLE = True

except ImportError:

    PYTESSERACT_AVAILABLE = False


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
# 用紙の向きは文字から自動判定する
# ---------------------------------------------------------
PAPER_SIZE = "A4"


# ---------------------------------------------------------
# 出力解像度
#
# 150 : 軽量
# 200 : おすすめ
# 300 : 高画質
# ---------------------------------------------------------
DPI = 200


# ---------------------------------------------------------
# 最終画像を表示
# ---------------------------------------------------------
SHOW_FINAL_PREVIEW = True


# ---------------------------------------------------------
# デバッグ表示
#
# True:
#   書類認識結果などを一時表示
#
# False:
#   最終画像だけ表示
# ---------------------------------------------------------
DEBUG_PREVIEW = False

DEBUG_WAIT = 1200


# ---------------------------------------------------------
# 撮影ガイド
# ---------------------------------------------------------
GUIDE_MARGIN_X = 0.10
GUIDE_MARGIN_Y = 0.10


# ---------------------------------------------------------
# 書類候補の最低・最大面積
# ---------------------------------------------------------
MIN_AREA_RATIO = 0.15
MAX_AREA_RATIO = 0.95


# =========================================================
# 用紙サイズ
# 単位：mm
# =========================================================

PAPER_SIZES_MM = {

    "A4": (
        210,
        297
    ),

    "B5": (
        182,
        257
    )
}


# =========================================================
# Tesseractを探す
# =========================================================
def setup_tesseract():

    if not PYTESSERACT_AVAILABLE:

        print(
            "pytesseract がインストールされていません。"
        )

        return False


    # -----------------------------------------------------
    # PATHにTesseractがある場合
    # -----------------------------------------------------

    tesseract_path = shutil.which(
        "tesseract"
    )


    if tesseract_path is not None:

        pytesseract.pytesseract.tesseract_cmd = (
            tesseract_path
        )

        return True


    # -----------------------------------------------------
    # Windowsの一般的なインストール場所
    # -----------------------------------------------------

    common_paths = [

        r"C:\Program Files\Tesseract-OCR\tesseract.exe",

        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ]


    for path in common_paths:

        if os.path.exists(
            path
        ):

            pytesseract.pytesseract.tesseract_cmd = (
                path
            )

            return True


    print(
        "Tesseract OCR本体が見つかりませんでした。"
    )

    print(
        "文字方向判定はフォールバック処理を使用します。"
    )

    return False


# =========================================================
# 4点の順番
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
        (
            4,
            2
        ),
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
# 仮の透視変換
#
# この時点ではA4/B5の比率にはしない。
#
# 撮影された書類そのものの形を使う。
# =========================================================
def four_point_transform(
    image,
    pts
):

    rect = order_points(
        pts
    )


    tl, tr, br, bl = rect


    # =====================================================
    # 横幅
    # =====================================================

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


    # =====================================================
    # 高さ
    # =====================================================

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


    # =====================================================
    # 変換先
    # =====================================================

    dst = np.array(
        [
            [
                0,
                0
            ],

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


    # =====================================================
    # 透視変換
    # =====================================================

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
# 輪郭を四角形へ近似
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


        # 4点以外を除外
        if len(approx) != 4:
            continue


        # 凸四角形以外を除外
        if not cv2.isContourConvex(
            approx
        ):

            continue


        return approx


    return None


# =========================================================
# 書類候補を採点
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


    # =====================================================
    # 中心
    # =====================================================

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


    # =====================================================
    # 四隅
    # =====================================================

    corner_distances = []


    for (
        document_point,
        guide_point
    ) in zip(
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


    # =====================================================
    # 面積
    # =====================================================

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


    # =====================================================
    # 総合得点
    # =====================================================

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
# 画像を指定角度だけ回転
#
# Tesseractのrotate値を使用
# =========================================================
def rotate_document(
    image,
    rotation
):

    if rotation == 90:

        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )


    elif rotation == 180:

        return cv2.rotate(
            image,
            cv2.ROTATE_180
        )


    elif rotation == 270:

        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )


    return image


# =========================================================
# OCR用の画像を作成
# =========================================================
def prepare_orientation_image(
    image
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # -----------------------------------------------------
    # 少し拡大するとOSDが安定する場合がある
    # -----------------------------------------------------

    height, width = gray.shape[:2]


    if max(
        width,
        height
    ) < 1500:

        scale = (
            1500
            /
            max(
                width,
                height
            )
        )


        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )


    # -----------------------------------------------------
    # 軽くノイズ除去
    # -----------------------------------------------------

    gray = cv2.GaussianBlur(
        gray,
        (
            3,
            3
        ),
        0
    )


    # -----------------------------------------------------
    # Otsu二値化
    #
    # これは方向判定専用。
    # 最終成果物には使わない。
    # -----------------------------------------------------

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
# 文字方向を自動判定
# =========================================================
def detect_text_orientation(
    image,
    tesseract_available
):

    # =====================================================
    # Tesseractが使えない場合
    # =====================================================

    if not tesseract_available:

        print(
            "OCR方向判定を使用できません。"
        )

        print(
            "回転角度は0度として処理します。"
        )

        return 0


    orientation_image = prepare_orientation_image(
        image
    )


    try:

        # -------------------------------------------------
        # Orientation and Script Detection
        # -------------------------------------------------

        osd = pytesseract.image_to_osd(
            orientation_image,
            output_type=Output.DICT
        )


        rotation = int(
            osd.get(
                "rotate",
                0
            )
        )


        orientation_confidence = osd.get(
            "orientation_conf",
            0
        )


        script = osd.get(
            "script",
            "Unknown"
        )


        script_confidence = osd.get(
            "script_conf",
            0
        )


        print(
            f"文字回転補正 : {rotation}度"
        )

        print(
            "方向判定信頼度 : "
            f"{orientation_confidence}"
        )

        print(
            f"文字種 : {script}"
        )

        print(
            "文字種判定信頼度 : "
            f"{script_confidence}"
        )


        return rotation


    except Exception as error:

        print(
            "文字方向の自動判定に失敗しました。"
        )

        print(
            f"理由 : {error}"
        )

        print(
            "回転角度0度で処理を続行します。"
        )


        return 0


# =========================================================
# mm → pixel
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
# 最終用紙サイズ
# =========================================================
def get_paper_pixel_size(
    paper,
    dpi,
    landscape
):

    if paper not in PAPER_SIZES_MM:

        raise ValueError(
            "PAPER_SIZE は "
            "'A4' または 'B5' を指定してください。"
        )


    paper_width_mm, paper_height_mm = (
        PAPER_SIZES_MM[
            paper
        ]
    )


    width_px = mm_to_px(
        paper_width_mm,
        dpi
    )


    height_px = mm_to_px(
        paper_height_mm,
        dpi
    )


    if landscape:

        return (
            height_px,
            width_px
        )


    return (
        width_px,
        height_px
    )


# =========================================================
# A4 / B5 の最終比率へ整形
# =========================================================
def resize_to_paper(
    image,
    paper,
    dpi,
    landscape
):

    target_width, target_height = (
        get_paper_pixel_size(
            paper,
            dpi,
            landscape
        )
    )


    result = cv2.resize(
        image,
        (
            target_width,
            target_height
        ),
        interpolation=cv2.INTER_CUBIC
    )


    return result


# =========================================================
# 表示用に縮小
#
# 保存画像には影響しない
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
# Tesseract初期化
# =========================================================

tesseract_available = setup_tesseract()


if tesseract_available:

    print(
        "Tesseract OCR : 使用可能"
    )

else:

    print(
        "Tesseract OCR : 使用不可"
    )


# =========================================================
# 入力画像
# =========================================================

img = cv2.imread(
    INPUT_FILE
)


if img is None:

    print(
        f"{INPUT_FILE} を読み込めませんでした。"
    )

    exit()


# =========================================================
# 画像サイズ
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
# 撮影ガイド
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
        1.0
        -
        GUIDE_MARGIN_X
    )
)


guide_y2 = int(
    height
    *
    (
        1.0
        -
        GUIDE_MARGIN_Y
    )
)


# =========================================================
# ガイド面積
# =========================================================

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


# =========================================================
# ガイド中心
# =========================================================

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


# =========================================================
# ガイド四隅
# =========================================================

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
# 書類検出用前処理
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

    area = cv2.contourArea(
        contour
    )


    area_ratio = (
        area
        /
        image_area
    )


    # 小さすぎる
    if area_ratio < MIN_AREA_RATIO:
        continue


    # 大きすぎる
    if area_ratio > MAX_AREA_RATIO:
        continue


    # 四角形へ近似
    approx = approximate_quadrilateral(
        contour
    )


    if approx is None:
        continue


    # 四角形の中心
    rect = order_points(
        approx
    )


    center = np.mean(
        rect,
        axis=0
    )


    # 中心がガイド内か
    if not (

        guide_x1
        <=
        center[0]
        <=
        guide_x2

        and

        guide_y1
        <=
        center[1]
        <=
        guide_y2

    ):

        continue


    # 採点
    score = calculate_document_score(

        approx,

        area,

        guide_corners,

        guide_area,

        guide_center,

        image_diagonal
    )


    if score > best_score:

        document = approx.copy()

        best_score = score

        best_area = area


# =========================================================
# 書類未検出
# =========================================================

if document is None:

    print(
        "書類を検出できませんでした。"
    )

    cv2.destroyAllWindows()

    exit()


# =========================================================
# 書類検出成功
# =========================================================

print(
    "書類を検出しました。"
)


print(
    f"Score : {best_score:.3f}"
)


print(
    "Area ratio : "
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
#
# まだA4/B5比率にはしない
# =========================================================

warped = four_point_transform(
    img,
    document
)


if warped is None:

    print(
        "透視変換に失敗しました。"
    )

    exit()


# =========================================================
# 文字方向を認識
# =========================================================

rotation = detect_text_orientation(
    warped,
    tesseract_available
)


# =========================================================
# 文字が正立するよう回転
# =========================================================

oriented = rotate_document(
    warped,
    rotation
)


# =========================================================
# 用紙が縦か横か判定
#
# 文字を正しい方向へ戻した「後」で判断する。
# =========================================================

oriented_height, oriented_width = (
    oriented.shape[:2]
)


landscape = (
    oriented_width
    >
    oriented_height
)


if landscape:

    print(
        "用紙方向 : 横向き"
    )

else:

    print(
        "用紙方向 : 縦向き"
    )


# =========================================================
# 画像補正
# =========================================================

scan_gray = cv2.cvtColor(
    oriented,
    cv2.COLOR_BGR2GRAY
)


# =========================================================
# ノイズ除去
# =========================================================

denoised = cv2.fastNlMeansDenoising(
    scan_gray,
    None,
    h=5,
    templateWindowSize=7,
    searchWindowSize=21
)


# =========================================================
# 背景・影を推定
# =========================================================

background = cv2.GaussianBlur(
    denoised,
    (
        0,
        0
    ),
    25
)


# =========================================================
# 影除去
# =========================================================

shadow_removed = cv2.divide(
    denoised,
    background,
    scale=255
)


# =========================================================
# コントラスト補正
# =========================================================

scan = cv2.convertScaleAbs(
    shadow_removed,
    alpha=1.15,
    beta=-20
)


# =========================================================
# 軽いシャープ化
# =========================================================

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


# =========================================================
# A4/B5の正しい比率へ変換
#
# ここで初めて用紙サイズを固定する
# =========================================================

paper_scan = resize_to_paper(
    final_scan,
    paper=PAPER_SIZE,
    dpi=DPI,
    landscape=landscape
)


# =========================================================
# 保存
# =========================================================

save_success = cv2.imwrite(
    OUTPUT_FILE,
    paper_scan,
    [
        cv2.IMWRITE_JPEG_QUALITY,
        95
    ]
)


if save_success:

    print(
        f"{OUTPUT_FILE} として保存しました。"
    )


    print(
        f"用紙サイズ : {PAPER_SIZE}"
    )


    print(
        f"DPI設定 : {DPI}"
    )


    print(
        "最終画像サイズ : "
        f"{paper_scan.shape[1]}"
        " x "
        f"{paper_scan.shape[0]}"
        " px"
    )


else:

    print(
        f"{OUTPUT_FILE} の保存に失敗しました。"
    )


# =========================================================
# 途中のOpenCVウィンドウを完全に閉じる
# =========================================================

cv2.destroyAllWindows()


# =========================================================
# 最終画像だけ表示
# =========================================================

if SHOW_FINAL_PREVIEW:

    preview = resize_for_display(
        paper_scan
    )


    cv2.imshow(
        "Final Scan",
        preview
    )


    print(
        "完成画像を表示しています。"
    )


    print(
        "何かキーを押すとウィンドウを閉じます。"
    )


    cv2.waitKey(0)


    cv2.destroyAllWindows()