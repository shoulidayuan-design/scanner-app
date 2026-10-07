import cv2
import numpy as np


# =========================================================
# 設定
# =========================================================

# 入力画像
INPUT_FILE = "document.jpg"

# 出力画像
OUTPUT_FILE = "document_1.jpg"


# ---------------------------------------------------------
# 出力用紙
# "A4" または "B5"
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
# 最終画像を画面に表示するか
# ---------------------------------------------------------
SHOW_FINAL_PREVIEW = True


# ---------------------------------------------------------
# デバッグ用
#
# False:
#   途中のウィンドウを表示しない
#
# True:
#   エッジや検出結果を一時的に表示する
# ---------------------------------------------------------
DEBUG_PREVIEW = False


# デバッグ画像を表示する時間（ミリ秒）
DEBUG_WAIT = 1200


# ---------------------------------------------------------
# 撮影ガイド
# ---------------------------------------------------------
GUIDE_MARGIN_X = 0.10
GUIDE_MARGIN_Y = 0.10


# ---------------------------------------------------------
# 書類候補として認める面積
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
    ).astype("float32")

    rect = np.zeros(
        (4, 2),
        dtype="float32"
    )


    # -----------------------------------------------------
    # x + y
    #
    # 最小 → 左上
    # 最大 → 右下
    # -----------------------------------------------------

    s = pts.sum(
        axis=1
    )

    rect[0] = pts[
        np.argmin(s)
    ]

    rect[2] = pts[
        np.argmax(s)
    ]


    # -----------------------------------------------------
    # y - x
    #
    # 最小 → 右上
    # 最大 → 左下
    # -----------------------------------------------------

    diff = np.diff(
        pts,
        axis=1
    ).reshape(-1)

    rect[1] = pts[
        np.argmin(diff)
    ]

    rect[3] = pts[
        np.argmax(diff)
    ]


    return rect


# =========================================================
# mm → pixel
# =========================================================
def mm_to_px(mm, dpi):

    return int(
        round(
            mm / 25.4 * dpi
        )
    )


# =========================================================
# 用紙サイズをpixelで取得
# =========================================================
def get_paper_pixel_size(
    paper,
    dpi,
    landscape=False
):

    if paper not in PAPER_SIZES_MM:

        raise ValueError(
            "PAPER_SIZE は "
            "'A4' または 'B5' を指定してください。"
        )


    paper_width_mm, paper_height_mm = (
        PAPER_SIZES_MM[paper]
    )


    width_px = mm_to_px(
        paper_width_mm,
        dpi
    )

    height_px = mm_to_px(
        paper_height_mm,
        dpi
    )


    # 横向きの場合
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
# 透視変換
#
# ここでA4/B5本来の縦横比に補正する
# =========================================================
def four_point_transform(
    image,
    pts,
    paper="A4"
):

    rect = order_points(
        pts
    )

    tl, tr, br, bl = rect


    # =====================================================
    # 撮影画像上での書類の幅
    # =====================================================

    width_a = np.linalg.norm(
        br - bl
    )

    width_b = np.linalg.norm(
        tr - tl
    )

    detected_width = max(
        width_a,
        width_b
    )


    # =====================================================
    # 撮影画像上での書類の高さ
    # =====================================================

    height_a = np.linalg.norm(
        tr - br
    )

    height_b = np.linalg.norm(
        tl - bl
    )

    detected_height = max(
        height_a,
        height_b
    )


    if (
        detected_width <= 0
        or
        detected_height <= 0
    ):

        return None, False


    # =====================================================
    # 用紙サイズ取得
    # =====================================================

    if paper not in PAPER_SIZES_MM:

        raise ValueError(
            "paper は "
            "'A4' または 'B5' を指定してください。"
        )


    paper_width_mm, paper_height_mm = (
        PAPER_SIZES_MM[paper]
    )


    # =====================================================
    # 縦置き / 横置き判定
    # =====================================================

    landscape = (
        detected_width
        >
        detected_height
    )


    # =====================================================
    # 透視変換後のサイズ
    #
    # 撮影画像の解像度を極端に変えず、
    # 用紙本来の縦横比だけを復元する
    # =====================================================

    if landscape:

        # 横向き
        paper_ratio = (
            paper_height_mm
            /
            paper_width_mm
        )

        output_width = int(
            detected_width
        )

        output_height = int(
            output_width
            /
            paper_ratio
        )

    else:

        # 縦向き
        paper_ratio = (
            paper_height_mm
            /
            paper_width_mm
        )

        output_width = int(
            detected_width
        )

        output_height = int(
            output_width
            *
            paper_ratio
        )


    if (
        output_width <= 0
        or
        output_height <= 0
    ):

        return None, landscape


    # =====================================================
    # 変換後の4点
    # =====================================================

    dst = np.array(
        [
            [
                0,
                0
            ],

            [
                output_width - 1,
                0
            ],

            [
                output_width - 1,
                output_height - 1
            ],

            [
                0,
                output_height - 1
            ]
        ],
        dtype="float32"
    )


    # =====================================================
    # 透視変換行列
    # =====================================================

    matrix = cv2.getPerspectiveTransform(
        rect,
        dst
    )


    # =====================================================
    # 透視変換
    # =====================================================

    warped = cv2.warpPerspective(
        image,
        matrix,
        (
            output_width,
            output_height
        )
    )


    return warped, landscape


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


    # 1種類だけだと4点にならない場合があるため
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


        # 4点でなければ除外
        if len(approx) != 4:
            continue


        # 凸四角形でなければ除外
        if not cv2.isContourConvex(
            approx
        ):
            continue


        return approx


    return None


# =========================================================
# 書類候補の点数
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
    # 書類候補の中心
    # =====================================================

    document_center = np.mean(
        rect,
        axis=0
    )


    # =====================================================
    # ガイド中心との距離
    # =====================================================

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
    # 四隅とガイド四隅との距離
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
    # ガイドとの面積差
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
    # 総合点
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
# 最終画像を正確なA4/B5サイズへ変更
#
# この時点ですでに縦横比を補正済みなので、
# 白い余白を追加する必要はない
# =========================================================
def resize_to_paper(
    image,
    paper,
    dpi,
    landscape=False
):

    target_width, target_height = (
        get_paper_pixel_size(
            paper,
            dpi,
            landscape
        )
    )


    resized = cv2.resize(
        image,
        (
            target_width,
            target_height
        ),
        interpolation=cv2.INTER_CUBIC
    )


    return resized


# =========================================================
# 画面表示用縮小
#
# 保存画像そのものには影響しない
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
# ガイド中央
# =========================================================

guide_center = np.array(
    [
        (
            guide_x1
            +
            guide_x2
        ) / 2,

        (
            guide_y1
            +
            guide_y2
        ) / 2
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
# グレースケール化
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
    (
        5,
        5
    ),
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
# エッジを少し繋げる
# =========================================================

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
# 書類候補
# =========================================================

document = None

best_score = -1

best_area = 0


# =========================================================
# 書類候補探索
# =========================================================

for contour in contours:

    # -----------------------------------------------------
    # 面積
    # -----------------------------------------------------

    area = cv2.contourArea(
        contour
    )


    area_ratio = (
        area
        /
        image_area
    )


    # -----------------------------------------------------
    # 小さすぎる輪郭を除外
    # -----------------------------------------------------

    if area_ratio < MIN_AREA_RATIO:
        continue


    # -----------------------------------------------------
    # 大きすぎる輪郭を除外
    # -----------------------------------------------------

    if area_ratio > MAX_AREA_RATIO:
        continue


    # -----------------------------------------------------
    # 四角形へ近似
    # -----------------------------------------------------

    approx = approximate_quadrilateral(
        contour
    )


    if approx is None:
        continue


    # -----------------------------------------------------
    # 候補の中心
    # -----------------------------------------------------

    rect = order_points(
        approx
    )


    center = np.mean(
        rect,
        axis=0
    )


    # -----------------------------------------------------
    # 中心がガイド内に存在するか
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # 書類候補を採点
    # -----------------------------------------------------

    score = calculate_document_score(
        approx,
        area,
        guide_corners,
        guide_area,
        guide_center,
        image_diagonal
    )


    # -----------------------------------------------------
    # 最も高得点の候補
    # -----------------------------------------------------

    if score > best_score:

        document = approx.copy()

        best_score = score

        best_area = area


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
        "Area ratio : "
        f"{best_area / image_area:.3f}"
    )


    # =====================================================
    # デバッグ用画像
    # =====================================================

    if DEBUG_PREVIEW:

        debug_image = img.copy()


        # 撮影ガイド
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


        # 検出した書類
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


        # エッジ表示
        cv2.imshow(
            "Edge",
            resize_for_display(
                edge_closed
            )
        )


        # 書類認識結果
        cv2.imshow(
            "Document Detection",
            resize_for_display(
                debug_image
            )
        )


        # 一時表示
        cv2.waitKey(
            DEBUG_WAIT
        )


        # -------------------------------------------------
        # 途中で開いたウィンドウを閉じる
        # -------------------------------------------------

        cv2.destroyAllWindows()


    # =====================================================
    # A4/B5比率で透視変換
    # =====================================================

    warped, landscape = four_point_transform(
        img,
        document,
        paper=PAPER_SIZE
    )


    if warped is None:

        print(
            "透視変換に失敗しました。"
        )

        cv2.destroyAllWindows()

        exit()


    # =====================================================
    # グレースケール
    # =====================================================

    scan_gray = cv2.cvtColor(
        warped,
        cv2.COLOR_BGR2GRAY
    )


    # =====================================================
    # 軽いノイズ除去
    # =====================================================

    denoised = cv2.fastNlMeansDenoising(
        scan_gray,
        None,
        h=5,
        templateWindowSize=7,
        searchWindowSize=21
    )


    # =====================================================
    # 背景・照明ムラを推定
    # =====================================================

    background = cv2.GaussianBlur(
        denoised,
        (
            0,
            0
        ),
        25
    )


    # =====================================================
    # 影除去
    # =====================================================

    shadow_removed = cv2.divide(
        denoised,
        background,
        scale=255
    )


    # =====================================================
    # コントラスト調整
    #
    # 白い紙は白く、
    # 文字や罫線は少し濃くする
    # =====================================================

    scan = cv2.convertScaleAbs(
        shadow_removed,
        alpha=1.15,
        beta=-20
    )


    # =====================================================
    # 軽いシャープ化
    # =====================================================

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


    # =====================================================
    # 正確なA4/B5ピクセルサイズへ変更
    #
    # この時点ですでに用紙比率になっているため、
    # 白余白は追加されない
    # =====================================================

    paper_scan = resize_to_paper(
        final_scan,
        paper=PAPER_SIZE,
        dpi=DPI,
        landscape=landscape
    )


    # =====================================================
    # 保存
    # =====================================================

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
            "向き : "
            +
            (
                "横"
                if landscape
                else "縦"
            )
        )

        print(
            f"DPI : {DPI}"
        )

        print(
            "出力サイズ : "
            f"{paper_scan.shape[1]}"
            " x "
            f"{paper_scan.shape[0]}"
            " px"
        )

    else:

        print(
            f"{OUTPUT_FILE} の保存に失敗しました。"
        )


    # =====================================================
    # 念のため、途中のウィンドウを全て閉じる
    # =====================================================

    cv2.destroyAllWindows()


    # =====================================================
    # 最終画像だけ表示
    # =====================================================

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
            "何かキーを押すと終了します。"
        )


        # キー入力待ち
        cv2.waitKey(0)


        # 最終ウィンドウを閉じる
        cv2.destroyAllWindows()


# =========================================================
# 書類を検出できなかった場合
# =========================================================

else:

    print(
        "書類を検出できませんでした。"
    )

    cv2.destroyAllWindows()