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
# 解像度
#
# 150 : 軽量
# 200 : おすすめ
# 300 : 高画質
# ---------------------------------------------------------
DPI = 200


# ---------------------------------------------------------
# 用紙サイズへの合わせ方
#
# True
#   元画像の縦横比を維持
#   足りない部分を白い余白で埋める
#
# False
#   A4/B5サイズへ強制的に引き伸ばす
# ---------------------------------------------------------
KEEP_ASPECT_RATIO = True


# ---------------------------------------------------------
# デバッグ表示
#
# False
#   最終画像だけ表示
#
# True
#   エッジ・書類認識などを途中で一時表示
#   その後、自動的に閉じる
# ---------------------------------------------------------
DEBUG_PREVIEW = False


# デバッグ表示時間
# ミリ秒
DEBUG_WAIT = 1200


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

    # x + y
    s = pts.sum(
        axis=1
    )

    rect[0] = pts[
        np.argmin(s)
    ]  # 左上

    rect[2] = pts[
        np.argmax(s)
    ]  # 右下


    # y - x
    diff = np.diff(
        pts,
        axis=1
    ).reshape(-1)

    rect[1] = pts[
        np.argmin(diff)
    ]  # 右上

    rect[3] = pts[
        np.argmax(diff)
    ]  # 左下

    return rect


# =========================================================
# 透視変換
# =========================================================
def four_point_transform(image, pts):

    rect = order_points(
        pts
    )

    tl, tr, br, bl = rect


    # -----------------------------------------------------
    # 横幅
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


    # 異常なサイズなら終了
    if max_width <= 0 or max_height <= 0:
        return None


    # -----------------------------------------------------
    # 変換後の座標
    # -----------------------------------------------------

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ],
        dtype="float32"
    )


    # -----------------------------------------------------
    # 透視変換行列
    # -----------------------------------------------------

    matrix = cv2.getPerspectiveTransform(
        rect,
        dst
    )


    # -----------------------------------------------------
    # 透視変換
    # -----------------------------------------------------

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
def approximate_quadrilateral(contour):

    peri = cv2.arcLength(
        contour,
        True
    )


    # 一つの値だけだと4点にならない場合があるため
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


        # 4点以外は除外
        if len(approx) != 4:
            continue


        # 凹んでいる四角形は除外
        if not cv2.isContourConvex(
            approx
        ):
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


    # -----------------------------------------------------
    # 書類候補の中心
    # -----------------------------------------------------

    document_center = np.mean(
        rect,
        axis=0
    )


    # -----------------------------------------------------
    # ガイド中心との近さ
    # -----------------------------------------------------

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
            (image_diagonal * 0.35),
            1.0
        )
    )


    # -----------------------------------------------------
    # 四隅とガイドの近さ
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
            (image_diagonal * 0.35),
            1.0
        )
    )


    # -----------------------------------------------------
    # ガイドとの面積差
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
        corner_score * 0.45
        +
        size_score * 0.35
        +
        center_score * 0.20
    )

    return score


# =========================================================
# mm → pixel
# =========================================================
def mm_to_px(mm, dpi):

    return int(
        mm / 25.4 * dpi
    )


# =========================================================
# A4 / B5 サイズへ整形
# =========================================================
def fit_to_paper(
    image,
    paper="A4",
    dpi=200,
    keep_aspect=True
):

    # -----------------------------------------------------
    # 用紙サイズ
    # 単位 mm
    # -----------------------------------------------------

    paper_sizes = {

        "A4": (
            210,
            297
        ),

        "B5": (
            182,
            257
        )
    }


    if paper not in paper_sizes:

        raise ValueError(
            "PAPER_SIZE は "
            "'A4' または 'B5' を指定してください。"
        )


    paper_width_mm, paper_height_mm = (
        paper_sizes[paper]
    )


    # -----------------------------------------------------
    # pixelへ変換
    # -----------------------------------------------------

    target_width = mm_to_px(
        paper_width_mm,
        dpi
    )

    target_height = mm_to_px(
        paper_height_mm,
        dpi
    )


    # =====================================================
    # 強制的に用紙サイズへ変更する場合
    # =====================================================

    if not keep_aspect:

        resized = cv2.resize(
            image,
            (
                target_width,
                target_height
            ),
            interpolation=cv2.INTER_CUBIC
        )

        return resized


    # =====================================================
    # 比率を維持する場合
    # =====================================================

    height, width = image.shape[:2]


    scale = min(
        target_width / width,
        target_height / height
    )


    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )


    resized = cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_CUBIC
    )


    # -----------------------------------------------------
    # 白背景の用紙を作成
    # -----------------------------------------------------

    if len(image.shape) == 2:

        # グレースケール
        canvas = np.full(
            (
                target_height,
                target_width
            ),
            255,
            dtype=np.uint8
        )

    else:

        # カラー画像
        canvas = np.full(
            (
                target_height,
                target_width,
                3
            ),
            255,
            dtype=np.uint8
        )


    # -----------------------------------------------------
    # 中央に配置
    # -----------------------------------------------------

    x = (
        target_width
        -
        new_width
    ) // 2

    y = (
        target_height
        -
        new_height
    ) // 2


    canvas[
        y:y + new_height,
        x:x + new_width
    ] = resized


    return canvas


# =========================================================
# 画面表示用に縮小
#
# 保存画像そのものは縮小しない
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


    if scale == 1.0:
        return image


    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
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


# ガイド面積
guide_area = (
    (guide_x2 - guide_x1)
    *
    (guide_y2 - guide_y1)
)


# ガイド中央
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


# ガイド四隅
guide_corners = np.array(
    [
        [guide_x1, guide_y1],
        [guide_x2, guide_y1],
        [guide_x2, guide_y2],
        [guide_x1, guide_y2]
    ],
    dtype="float32"
)


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
# エッジを少し繋げる
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


    # 小さすぎる
    if area_ratio < MIN_AREA_RATIO:
        continue


    # 大きすぎる
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
    # 中心を取得
    # -----------------------------------------------------

    rect = order_points(
        approx
    )


    center = np.mean(
        rect,
        axis=0
    )


    # -----------------------------------------------------
    # 中心がガイド内か
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
    # 一番得点の高いもの
    # -----------------------------------------------------

    if score > best_score:

        document = approx.copy()

        best_score = score

        best_area = area


# =========================================================
# 確認用画像
# =========================================================

result = img.copy()


# ガイド表示
cv2.rectangle(
    result,
    (
        guide_x1,
        guide_y1
    ),
    (
        guide_x2,
        guide_y2
    ),
    (255, 0, 0),
    3
)


text_y = (
    guide_y1
    -
    15
)


if text_y < 30:

    text_y = (
        guide_y1
        +
        30
    )


cv2.putText(
    result,
    "Place document inside guide",
    (
        guide_x1,
        text_y
    ),
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
        "Area ratio : "
        f"{best_area / image_area:.3f}"
    )


    # -----------------------------------------------------
    # 検出した輪郭
    # -----------------------------------------------------

    cv2.drawContours(
        result,
        [document],
        -1,
        (0, 255, 0),
        4
    )


    # -----------------------------------------------------
    # 四隅
    # -----------------------------------------------------

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
            (
                x,
                y
            ),
            8,
            (0, 0, 255),
            -1
        )


    # =====================================================
    # デバッグ表示
    # =====================================================

    if DEBUG_PREVIEW:

        cv2.imshow(
            "Edge",
            resize_for_display(
                edge_closed
            )
        )

        cv2.imshow(
            "Document Detection",
            resize_for_display(
                result
            )
        )


        # 指定時間だけ表示
        cv2.waitKey(
            DEBUG_WAIT
        )


        # -------------------------------------------------
        # 中間ウィンドウをすべて閉じる
        # -------------------------------------------------

        cv2.destroyAllWindows()


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
        # 背景・照明ムラを推定
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
        # 紙を白くしつつ
        # 文字を少し濃くする
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
        # A4 / B5 サイズへ整形
        # =================================================

        paper_scan = fit_to_paper(
            final_scan,
            paper=PAPER_SIZE,
            dpi=DPI,
            keep_aspect=KEEP_ASPECT_RATIO
        )


        # =================================================
        # 保存
        # =================================================

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


        # =================================================
        # 念のため中間ウィンドウを閉じる
        # =================================================

        cv2.destroyAllWindows()


        # =================================================
        # 最終結果だけ表示
        # =================================================

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


        # -------------------------------------------------
        # 最終ウィンドウも閉じる
        # -------------------------------------------------

        cv2.destroyAllWindows()


    else:

        print(
            "透視変換に失敗しました。"
        )

        cv2.destroyAllWindows()


# =========================================================
# 書類検出失敗
# =========================================================

else:

    print(
        "書類を検出できませんでした。"
    )

    cv2.destroyAllWindows()