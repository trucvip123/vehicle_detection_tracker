"""
PaddleOCR Wrapper cho License Plate Recognition
Thay thế yolo_license_plate bằng PaddleOCR
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import collections
import logging

# Try to import logging utility
try:
    from VehicleDetectionTracker.logging_utils import log as _log_ocr
except ImportError:

    def _log_ocr(message, category="ocr"):
        print(
            f"[{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        )


# Module logger
logger = logging.getLogger(__name__)


class PaddleOCRWrapper:
    """
    Wrapper class để sử dụng PaddleOCR cho license plate recognition
    """

    def __init__(
        self,
        lang="en",
        use_angle_cls=False,
        show_log=False,
        lite=False,
        det_model_dir=None,
        rec_model_dir=None,
        use_gpu=None,  # None = auto-detect, True = force GPU, False = force CPU
    ):
        """
        Khởi tạo PaddleOCR

        Args:
            lang (str): Ngôn ngữ ('en', 'ch', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka', 'hi', 'mr', 'ne', 'ur', 'fa', 'bn', 'bg', 'uk', 'be', 'te', 'ab', 'ru', 'rs_cyrillic', 'oc', 'bg', 'uk', 'be', 'te', 'kn', 'ch_tra', 'hi', 'mr', 'ne', 'ur', 'fa', 'bn', 'arabic', 'cyrillic', 'devanagari')
            use_angle_cls (bool): Có sử dụng angle classification không
            show_log (bool): Có hiển thị log không (không sử dụng trong phiên bản mới)
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.show_log = show_log
        self.lite = lite
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.use_gpu = use_gpu  # Lưu use_gpu parameter

        # Khởi tạo PaddleOCR
        self._init_ocr()

    def _init_ocr(self):
        """Khởi tạo PaddleOCR"""
        try:
            # Chuẩn bị tham số khởi tạo
            ocr_kwargs = dict(
                lang=self.lang,
                use_angle_cls=self.use_angle_cls,
            )

            # Phiên bản PaddleOCR mới không hỗ trợ tham số use_gpu trực tiếp
            # PaddleOCR sẽ tự động phát hiện GPU dựa trên PaddlePaddle được cài đặt
            # Kiểm tra và log GPU status (chỉ để thông báo)
            device_info = "auto (PaddleOCR tự phát hiện)"
            try:
                import paddle

                # Kiểm tra xem PaddlePaddle có được compile với CUDA không
                if paddle.device.is_compiled_with_cuda():
                    _log_ocr(
                        "✓ PaddleOCR: PaddlePaddle hỗ trợ GPU, sẽ tự động sử dụng GPU nếu có",
                        "ocr",
                    )
                    device_info = "GPU (auto-detected)"
                else:
                    _log_ocr(
                        "✓ PaddleOCR: PaddlePaddle được compile cho CPU, sử dụng CPU",
                        "ocr",
                    )
                    device_info = "CPU"
            except Exception:
                # Nếu không import được paddle, để PaddleOCR tự phát hiện
                _log_ocr(
                    "✓ PaddleOCR: PaddleOCR sẽ tự động chọn device (CPU hoặc GPU)",
                    "ocr",
                )

            # Lưu ý: use_gpu parameter chỉ để thông báo, không ảnh hưởng đến PaddleOCR
            # PaddleOCR mới tự động phát hiện dựa trên PaddlePaddle được cài đặt
            if self.use_gpu is False:
                _log_ocr(
                    "⚠ PaddleOCR: Lưu ý - use_gpu=False được chỉ định nhưng PaddleOCR mới tự động phát hiện device",
                    "ocr",
                )
                device_info = "CPU (requested, nhưng PaddleOCR tự quyết định)"
            elif self.use_gpu is True:
                _log_ocr(
                    "✓ PaddleOCR: Cố gắng sử dụng GPU (nếu PaddlePaddle-GPU được cài đặt)",
                    "ocr",
                )
                device_info = "GPU (requested)"

            # Bật cấu hình "lite" nếu người dùng yêu cầu
            # - Ưu tiên dùng model dir nhỏ nếu được cung cấp và tồn tại
            # - Nếu không, vẫn dùng mobile models mặc định của PaddleOCR
            if self.lite:
                import os

                if self.det_model_dir and os.path.isdir(self.det_model_dir):
                    ocr_kwargs["det_model_dir"] = self.det_model_dir
                if self.rec_model_dir and os.path.isdir(self.rec_model_dir):
                    ocr_kwargs["rec_model_dir"] = self.rec_model_dir
                # Sử dụng phiên bản mobile của PP-OCR (nhanh/nhẹ)
                ocr_kwargs["ocr_version"] = "PP-OCRv3"
                # Set cpu_threads cho CPU mode (không ảnh hưởng nếu dùng GPU)
                ocr_kwargs["cpu_threads"] = 4

            # Khởi tạo PaddleOCR với tham số đã cấu hình
            # Lưu ý: KHÔNG truyền use_gpu vào ocr_kwargs vì PaddleOCR mới không hỗ trợ tham số này
            # PaddleOCR sẽ tự động sử dụng GPU nếu PaddlePaddle-GPU được cài đặt
            self.ocr = PaddleOCR(**ocr_kwargs)
            _log_ocr(
                f"OK: PaddleOCR initialized successfully (lang={self.lang}, device={device_info})",
                "ocr",
            )

        except Exception as e:
            _log_ocr(f"ERROR: Error initializing PaddleOCR: {e}", "ocr")
            raise

    def read_license_plate(self, image):
        """
        Đọc biển số xe từ ảnh với xử lý tối ưu

        Args:
            image: Input image (numpy array hoặc PIL Image)

        Returns:
            str: Biển số xe được nhận dạng hoặc "unknown"
        """
        try:
            # Chuyển đổi input thành numpy array
            if hasattr(image, "shape"):
                # Nếu là numpy array
                img_array = image.copy()
            else:
                # Nếu là PIL Image hoặc path
                from PIL import Image

                if isinstance(image, str):
                    img_array = np.array(Image.open(image))
                else:
                    img_array = np.array(image)

            # Chuyển từ RGB sang BGR nếu cần
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Xử lý ảnh để cải thiện OCR
            # processed_images = self._preprocess_image_for_ocr(img_array)
            processed_images = [img_array]

            all_text_results = []

            # Thử OCR trên các ảnh đã xử lý
            for processed_img in processed_images:
                try:
                    # OCR với PaddleOCR
                    results = self.ocr.ocr(processed_img)

                    if results and len(results) > 0:
                        # Xử lý kết quả - phiên bản mới có cấu trúc khác
                        result = results[0] if isinstance(results, list) else results

                        # Kiểm tra cấu trúc mới
                        if isinstance(result, dict):
                            # Cấu trúc mới: dict với rec_texts và rec_scores
                            if "rec_texts" in result and "rec_scores" in result:
                                texts = result["rec_texts"]
                                scores = result["rec_scores"]

                                for text, confidence in zip(texts, scores):
                                    if confidence > 0.3:  # Threshold thấp hơn
                                        cleaned_text = self._clean_license_plate_text(
                                            text
                                        )
                                        if cleaned_text and len(cleaned_text) >= 3:
                                            all_text_results.append(
                                                (cleaned_text, confidence)
                                            )

                        elif isinstance(result, list):
                            # Cấu trúc cũ: list of lines
                            for line in result:
                                if line and len(line) >= 2:
                                    try:
                                        # Kiểm tra cấu trúc kết quả
                                        if (
                                            isinstance(line[1], (list, tuple))
                                            and len(line[1]) >= 2
                                        ):
                                            text = line[1][0]  # Text được nhận dạng
                                            confidence = line[1][1]  # Confidence score
                                        else:
                                            continue

                                        # Giảm confidence threshold cho biển số
                                        if confidence > 0.3:  # Threshold thấp hơn
                                            cleaned_text = (
                                                self._clean_license_plate_text(text)
                                            )
                                            if (
                                                cleaned_text and len(cleaned_text) >= 6
                                            ):  # Độ dài tối thiểu thấp hơn
                                                all_text_results.append(
                                                    (cleaned_text, confidence)
                                                )
                                    except (IndexError, TypeError):
                                        continue
                except Exception as e:
                    continue

            if all_text_results:
                # Sắp xếp theo confidence và độ dài
                all_text_results.sort(key=lambda x: (x[1], len(x[0])), reverse=True)

                # Thử kết hợp các kết quả để tạo biển số hoàn chỉnh
                combined_text = self._combine_license_plate_results(all_text_results)
                if combined_text:
                    _log_ocr(f"DEBUG: Combined OCR result: {combined_text}", "ocr")
                    return combined_text.upper()
            return "unknown"

        except Exception as e:
            _log_ocr(f"ERROR: Error reading license plate: {e}", "ocr")
            return "unknown"

    def _preprocess_image_for_ocr(self, img_array):
        """
        Xử lý ảnh để cải thiện OCR cho biển số xe

        Args:
            img_array: Ảnh gốc (numpy array)

        Returns:
            list: Danh sách các ảnh đã xử lý
        """
        processed_images = []

        try:
            # 1. Ảnh gốc
            processed_images.append(img_array.copy())

            # 2. Resize ảnh để tăng độ phân giải
            height, width = img_array.shape[:2]
            if height < 100 or width < 200:
                scale_factor = max(200 / width, 100 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                resized = cv2.resize(
                    img_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC
                )
                processed_images.append(resized)

            # 3. Chuyển sang grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array.copy()

            # 4. Tăng contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            processed_images.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

            # 5. Gaussian blur để làm mịn
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            processed_images.append(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR))

            # 6. Threshold binary
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))

            # 7. Adaptive threshold
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            processed_images.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))

            # 8. Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            processed_images.append(cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR))

        except Exception as e:
            _log_ocr(f"DEBUG: Error in preprocessing: {e}", "ocr")
            # Trả về ảnh gốc nếu có lỗi
            processed_images = [img_array.copy()]

        return processed_images

    def normalize_plate_text(self, text):
        """Chuẩn hóa: bỏ khoảng trắng, viết hoa, thay ký tự dễ nhầm."""
        if not text:
            return ""
        text = text.strip().upper()
        text = (
            text.replace(" ", "")
            .replace("O", "0")
            .replace("I", "1")
            .replace("T", "7")
            .replace("D", "0")
        )
        # print(f"DEBUG: text after replace: {text}")
        text = re.sub(r"[^A-Z0-9\.-]", "", text)  # chỉ giữ ký tự hợp lệ
        # print(f"DEBUG: text after sub: {text}")
        return text

    def merge_ocr_results(self, ocr_results):
        # print(f"DEBUG: ocr_results: {ocr_results}")
        """Hợp nhất danh sách kết quả OCR thành biển số hợp lý nhất."""
        normalized = [self.normalize_plate_text(t) for t in ocr_results if t]
        if not normalized:
            return None
        # print(f"DEBUG: normalized: {normalized}")
        # Đếm tần suất chuỗi
        counts = collections.Counter(normalized)
        most_common = counts.most_common(1)[0][0]

        # Tìm chuỗi dạng mã tỉnh: 1–3 số + 1–2 chữ cái (ví dụ: 77A, 51G, 47F1,...)
        province_part = next(
            (t for t in normalized if re.match(r"^\d{2,3}[A-Z]{1,2}$", t.upper())), ""
        )
        # number_candidates chỉ chứa chuỗi số (không còn chữ), lấy từ normalized, mỗi phần tử là chuỗi số có độ dài 4–5 ký tự.
        number_candidates = [
            re.sub(r"\D", "", t)
            for t in normalized
            if re.match(r"^\d{4,5}$", re.sub(r"\D", "", t))
        ]
        number_part = max(number_candidates, key=len, default="")
        # print(f"DEBUG: province_part: {province_part}")
        # print(f"DEBUG: number_part: {number_part}")
        # Ghép hai phần lại
        if province_part and number_part:
            plate = f"{province_part}-{number_part}"
        else:
            plate = most_common
        # print(f"DEBUG: plate before format: {plate}")
        # Làm sạch định dạng kiểu 77A33151 -> 77A-331.51
        plate = re.sub(r"^(\d{2}[A-Z])[- ]?(\d{3})(\d{2})$", r"\1-\2.\3", plate)
        # print(f"DEBUG: plate after format: {plate}")
        if len(plate) < 8:
            _log_ocr(
                f"merge_ocr_results returning None due to short length: {plate}", "ocr"
            )
            return None
        if "-" not in plate:
            return None
        return plate

    def _combine_license_plate_results(self, text_results):
        """
        Kết hợp các kết quả OCR để tạo biển số hoàn chỉnh theo format XX XXX.XX

        Args:
            text_results: List các tuple (text, confidence)

        Returns:
            str: Biển số kết hợp theo format XX XXX.XX hoặc None
        """
        if not text_results:
            return None

        # Lấy tất cả text unique với confidence
        unique_results = {}
        _log_ocr(f"text_results: {text_results}", "ocr")
        for text, confidence in text_results:
            if text not in unique_results and confidence > 0.5:
                unique_results[text] = confidence

        unique_texts = list(unique_results.keys())
        _log_ocr(f"DEBUG: All OCR results: {unique_texts}", "ocr")
        if len(unique_texts) == 0:
            return None
        elif len(unique_texts) == 1:
            license_plate = unique_texts[0]
            if len(license_plate) < 7:
                return None
            # Check if the first character is a digit (biển số xe VN phải bắt đầu bằng số)
            if (
                not license_plate[0].isdigit()
                or not license_plate[1].isdigit()
                or (license_plate[2].isdigit() and license_plate[2] != "1")
            ):
                _log_ocr(
                    f"DEBUG: License plate does not format license plate: {license_plate}",
                    "ocr",
                )
                return None
        return self.merge_ocr_results(unique_texts)

    def _clean_license_plate_text(self, text):
        """
        Làm sạch text biển số xe cho biển số Việt Nam

        Args:
            text (str): Text thô từ OCR

        Returns:
            str: Text đã được làm sạch
        """
        if not text:
            return ""

        # Loại bỏ khoảng trắng và ký tự đặc biệt không cần thiết
        text = re.sub(r"[^\w\-\.]", "", text)

        # Loại bỏ các từ không phù hợp
        unwanted_words = [
            "license",
            "plate",
            "number",
            "biển",
            "số",
            "xe",
            "car",
            "vehicle",
        ]
        for word in unwanted_words:
            text = text.replace(word.lower(), "")

        # Chỉ giữ lại chữ cái, số, dấu gạch ngang và dấu chấm
        text = re.sub(r"[^A-Za-z0-9\-\.]", "", text)

        # Xử lý các ký tự dễ nhầm lẫn
        text = text.replace("O", "0")  # O thành 0
        text = text.replace("I", "1")  # I thành 1
        text = text.replace("S", "5")  # S thành 5 (trong một số trường hợp)

        # Kiểm tra độ dài hợp lệ cho biển số VN
        if len(text) < 3 or len(text) > 15:
            return ""

        # Kiểm tra pattern biển số VN
        # Pattern: XX-XXXX hoặc XX-XXXXX hoặc XX-XXX.XX
        if re.match(r"^[A-Z0-9]{2,3}[-\.]?[A-Z0-9]{3,5}$", text.upper()):
            return text.upper()

        # Nếu không match pattern, vẫn trả về text đã làm sạch
        return text.upper()

    def __call__(self, image):
        """
        Alias cho read_license_plate để tương thích với API cũ
        """
        return self.read_license_plate(image)


def create_paddleocr_reader(
    lang="en",
    use_angle_cls=False,
    show_log=False,
    lite=False,
    det_model_dir=None,
    rec_model_dir=None,
    use_gpu=None,
):
    """
    Tạo PaddleOCR reader

    Args:
        lang (str): Ngôn ngữ
        use_angle_cls (bool): Sử dụng angle classification
        show_log (bool): Hiển thị log

    Returns:
        PaddleOCRWrapper: OCR reader
    """
    return PaddleOCRWrapper(
        lang=lang,
        use_angle_cls=use_angle_cls,
        show_log=show_log,
        lite=lite,
        det_model_dir=det_model_dir,
        rec_model_dir=rec_model_dir,
        use_gpu=use_gpu,
    )


# Alias để tương thích với code cũ
def load_paddleocr_model(
    lang="en",
    use_angle_cls=False,
    show_log=False,
    lite=False,
    det_model_dir=None,
    rec_model_dir=None,
    use_gpu=None,
):
    """
    Load PaddleOCR model (alias cho create_paddleocr_reader)
    """
    return create_paddleocr_reader(
        lang=lang,
        use_angle_cls=use_angle_cls,
        show_log=show_log,
        lite=lite,
        det_model_dir=det_model_dir,
        rec_model_dir=rec_model_dir,
        use_gpu=use_gpu,
    )
