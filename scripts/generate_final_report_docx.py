"""Generate the final Vietnamese report DOCX without external dependencies."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape


OUT = Path("docs/final_report/final_report.docx")


def t(text: str) -> str:
    return escape(str(text), {'"': "&quot;"})


def run(text: str, bold=False, italic=False, size=None, color=None):
    props = []
    if bold:
        props.append("<w:b/>")
    if italic:
        props.append("<w:i/>")
    if size:
        props.append(f'<w:sz w:val="{int(size * 2)}"/>')
    if color:
        props.append(f'<w:color w:val="{color}"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f"<w:r>{rpr}<w:t xml:space=\"preserve\">{t(text)}</w:t></w:r>"


def para(text="", style=None, align=None, bold=False, italic=False, size=None, color=None, before=None, after=None):
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    if before is not None or after is not None:
        before_xml = f' w:before="{before}"' if before is not None else ""
        after_xml = f' w:after="{after}"' if after is not None else ""
        ppr.append(f"<w:spacing{before_xml}{after_xml}/>")
    ppr_xml = f"<w:pPr>{''.join(ppr)}</w:pPr>" if ppr else ""
    return f"<w:p>{ppr_xml}{run(text, bold=bold, italic=italic, size=size, color=color)}</w:p>"


def page_break():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def cell(text: str, width: int, header=False):
    fill = '<w:shd w:fill="F2F4F7"/>' if header else ""
    bold = header
    return (
        "<w:tc>"
        f'<w:tcPr><w:tcW w:w="{width}" w:type="dxa"/><w:vAlign w:val="center"/>'
        '<w:tcMar><w:top w:w="80" w:type="dxa"/><w:bottom w:w="80" w:type="dxa"/>'
        '<w:start w:w="120" w:type="dxa"/><w:end w:w="120" w:type="dxa"/></w:tcMar>'
        f"{fill}</w:tcPr>"
        f"{para(text, style='TableText', bold=bold)}"
        "</w:tc>"
    )


def table(rows, widths):
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    trs = []
    for r_idx, row in enumerate(rows):
        cells = "".join(cell(value, widths[i], header=(r_idx == 0)) for i, value in enumerate(row))
        trs.append(f"<w:tr>{cells}</w:tr>")
    return (
        "<w:tbl>"
        '<w:tblPr><w:tblW w:w="9360" w:type="dxa"/><w:tblInd w:w="120" w:type="dxa"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:color="D0D5DD"/>'
        '<w:left w:val="single" w:sz="4" w:color="D0D5DD"/><w:bottom w:val="single" w:sz="4" w:color="D0D5DD"/>'
        '<w:right w:val="single" w:sz="4" w:color="D0D5DD"/><w:insideH w:val="single" w:sz="4" w:color="D0D5DD"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="D0D5DD"/></w:tblBorders>'
        '<w:tblLayout w:type="fixed"/></w:tblPr>'
        f"<w:tblGrid>{grid}</w:tblGrid>"
        f"{''.join(trs)}</w:tbl>"
        + para("", after=120)
    )


def section_title(text):
    return para(text, style="Heading1")


def subsection(text):
    return para(text, style="Heading2")


def build_document_xml():
    body = []
    body.append(para("BÁO CÁO CUỐI KÌ", style="Title", align="center"))
    body.append(para("Hệ thống phát hiện và phân đoạn ảnh y tế MedSeg", style="Subtitle", align="center"))
    body.append(para("", after=360))
    body.append(table([
        ["Thông tin", "Nội dung"],
        ["Môn học", ".........................................................."],
        ["Giảng viên hướng dẫn", ".........................................................."],
        ["Sinh viên thực hiện", ".........................................................."],
        ["Mã số sinh viên", ".........................................................."],
        ["Lớp", ".........................................................."],
        ["Ngày hoàn thành", "25/06/2026"],
    ], [2600, 6760]))
    body.append(page_break())

    body.append(section_title("Tóm tắt"))
    body.append(para("Đề tài xây dựng hệ thống MedSeg nhằm hỗ trợ phân tích ảnh y tế bằng học sâu. Hệ thống tập trung vào phát hiện tổn thương da trên ISIC 2018 bằng Faster R-CNN và phân đoạn vùng tổn thương hoặc vùng phổi bằng Attention U-Net. Ngoài phần huấn luyện và đánh giá mô hình, đề tài triển khai backend FastAPI và frontend React để người dùng tải ảnh, chọn workflow phân tích, xem overlay và theo dõi chỉ số đánh giá."))
    body.append(para("Kết quả thực nghiệm: phân đoạn phổi X-quang đạt Dice 0.9811 và IoU 0.9628; phân đoạn tổn thương da ISIC đạt Dice 0.8847 và IoU 0.7932; phát hiện tổn thương da đạt mAP@0.5 là 0.7142."))

    body.append(section_title("1. Giới thiệu đề tài"))
    body.append(para("Ảnh y tế là nguồn dữ liệu quan trọng trong chẩn đoán và theo dõi bệnh. Việc đọc ảnh và khoanh vùng tổn thương thủ công thường tốn thời gian, phụ thuộc kinh nghiệm chuyên môn và khó mở rộng khi số lượng ảnh lớn. Các mô hình học sâu có khả năng học đặc trưng hình ảnh và hỗ trợ tự động hóa một phần quy trình phân tích."))
    body.append(para("MedSeg được xây dựng như một prototype end-to-end gồm dữ liệu, mô hình, pipeline inference, API backend và frontend. Hệ thống không thay thế bác sĩ mà đóng vai trò công cụ hỗ trợ định vị và trực quan hóa vùng quan tâm."))

    body.append(section_title("2. Mục tiêu và phạm vi"))
    body.append(table([
        ["Nhóm mục tiêu", "Nội dung"],
        ["Dữ liệu", "Xử lý ISIC 2018 và Montgomery/Shenzhen Chest X-ray."],
        ["Mô hình", "Faster R-CNN cho detection; Attention U-Net cho segmentation."],
        ["Phần mềm", "Backend FastAPI, frontend React, trang chỉ số và xuất PDF."],
        ["Đánh giá", "Test-level và per-image/trial-level trên tập test độc lập."],
    ], [2200, 7160]))
    body.append(para("Hệ thống hiện chưa có database. Các checkpoint, dữ liệu processed, split và file metrics được lưu trên filesystem local."))

    body.append(section_title("3. Cơ sở lý thuyết"))
    body.append(subsection("3.1 Object Detection"))
    body.append(para("Object detection xác định vị trí đối tượng bằng bounding box. Trong đề tài, Faster R-CNN trả về boxes, scores và labels để phát hiện tổn thương da."))
    body.append(subsection("3.2 Image Segmentation"))
    body.append(para("Image segmentation gán nhãn từng pixel. Segmentation cho biết chính xác pixel nào thuộc vùng tổn thương hoặc vùng phổi, chi tiết hơn bounding box."))
    body.append(subsection("3.3 Các chỉ số đánh giá"))
    body.append(table([
        ["Chỉ số", "Công thức", "Ý nghĩa"],
        ["Dice", "2TP / (2TP + FP + FN)", "Độ trùng khớp giữa mask dự đoán và mask thật."],
        ["IoU", "TP / (TP + FP + FN)", "Tỉ lệ giao nhau trên hợp nhất."],
        ["Sensitivity", "TP / (TP + FN)", "Khả năng tìm đúng vùng thật."],
        ["Specificity", "TN / (TN + FP)", "Khả năng nhận đúng vùng nền."],
        ["Pixel Accuracy", "(TP + TN) / tổng pixel", "Tỉ lệ pixel đúng tổng thể."],
        ["mAP@0.5", "AP với IoU >= 0.5", "Chỉ số chính cho detection."],
    ], [1700, 2600, 5060]))

    body.append(section_title("4. Dữ liệu và tiền xử lý"))
    body.append(table([
        ["Dataset", "Tác vụ", "Train", "Val", "Test"],
        ["ISIC 2018", "Detection + segmentation tổn thương da", "1815", "389", "390"],
        ["Montgomery CXR", "Segmentation vùng phổi", "96", "20", "22"],
    ], [2200, 3860, 1100, 1100, 1100]))
    body.append(para("data/raw chứa dữ liệu gốc. data/processed chứa ảnh và mask đã tiền xử lý. data/splits chứa CSV train/val/test. Với Chest X-ray, mask trái và phải được gộp thành mask phổi hoàn chỉnh. Với ISIC, annotation detection được suy ra từ mask tổn thương."))

    body.append(section_title("5. Kiến trúc hệ thống"))
    body.append(para("Kiến trúc tổng quát: React Frontend -> FastAPI Backend -> PyTorch Models. Frontend upload ảnh và hiển thị kết quả. Backend load checkpoint, chạy inference và trả kết quả JSON/base64."))
    body.append(table([
        ["Endpoint", "Chức năng"],
        ["/api/health", "Kiểm tra trạng thái API, GPU và model đã load."],
        ["/api/detect", "Phát hiện tổn thương da bằng Faster R-CNN."],
        ["/api/segment", "Phân đoạn phổi X-quang."],
        ["/api/pipeline", "Pipeline ISIC đầy đủ: detection + segmentation."],
    ], [2400, 6960]))

    body.append(section_title("6. Huấn luyện mô hình"))
    body.append(table([
        ["Checkpoint", "Ý nghĩa"],
        ["outputs/checkpoints/best_detection.pth", "Faster R-CNN phát hiện tổn thương da ISIC."],
        ["outputs/checkpoints/best_segmentation.pth", "Attention U-Net phân đoạn tổn thương da ISIC."],
        ["outputs/checkpoints/best_chest_xray_segmentation.pth", "Attention U-Net phân đoạn phổi X-quang."],
    ], [4300, 5060]))

    body.append(section_title("7. Đánh giá thực nghiệm"))
    body.append(subsection("7.1 Phân đoạn phổi X-quang"))
    body.append(table([
        ["Chỉ số", "Toàn test", "Mean per-image", "Std per-image"],
        ["Dice", "0.9811", "0.9807", "0.0115"],
        ["IoU", "0.9628", "0.9623", "0.0215"],
        ["Sensitivity", "0.9804", "0.9807", "0.0227"],
        ["Specificity", "0.9932", "0.9931", "0.0027"],
        ["Pixel Accuracy", "0.9897", "0.9897", "0.0064"],
    ], [2600, 2200, 2300, 2260]))
    body.append(subsection("7.2 Phân đoạn tổn thương da ISIC"))
    body.append(table([
        ["Chỉ số", "Toàn test", "Mean per-image", "Std per-image"],
        ["Dice", "0.8847", "0.8707", "0.1443"],
        ["IoU", "0.7932", "0.7927", "0.1728"],
        ["Sensitivity", "0.8611", "0.9135", "0.1495"],
        ["Specificity", "0.9774", "0.9757", "0.0377"],
        ["Pixel Accuracy", "0.9532", "0.9532", "0.0741"],
    ], [2600, 2200, 2300, 2260]))
    body.append(subsection("7.3 Phát hiện tổn thương da ISIC"))
    body.append(table([
        ["Chỉ số", "Giá trị"],
        ["mAP@0.5", "0.7142"],
        ["Precision@0.5", "0.8768"],
        ["Recall@0.5", "0.7846"],
        ["TP / FP / FN", "306 / 43 / 84"],
        ["Best IoU mean/std", "0.6879 / 0.3365"],
    ], [3300, 6060]))

    body.append(section_title("8. Giao diện và chức năng phần mềm"))
    body.append(table([
        ["Trang", "Chức năng"],
        ["Tổng quan", "Hiển thị trạng thái API, GPU, model đã load và lịch sử gần đây."],
        ["Phân tích", "Upload ảnh và chọn workflow inference."],
        ["So sánh", "So sánh detection ISIC với pipeline ISIC đầy đủ trên cùng ảnh."],
        ["Chỉ số", "Hiển thị số liệu test, giải thích metric và xuất PDF."],
        ["Lịch sử", "Hiển thị lượt phân tích lưu trong localStorage."],
    ], [2200, 7160]))

    body.append(section_title("9. Hạn chế và hướng phát triển"))
    body.append(table([
        ["Hạn chế", "Hướng phát triển"],
        ["Chưa có database lưu lịch sử dài hạn.", "Thêm database lưu user, ảnh upload, kết quả và metadata."],
        ["Detection chưa huấn luyện riêng cho X-quang.", "Train thêm detection nếu cần phát hiện bất thường trên X-quang."],
        ["Chưa hỗ trợ DICOM trực tiếp trên frontend.", "Bổ sung parser DICOM và preview ảnh y tế chuyên dụng."],
        ["Chưa triển khai production.", "Đóng gói Docker và cấu hình deploy."],
    ], [4680, 4680]))

    body.append(section_title("10. Kết luận"))
    body.append(para("MedSeg đã xây dựng được hệ thống phân tích ảnh y tế hoàn chỉnh ở mức prototype, bao gồm xử lý dữ liệu, huấn luyện mô hình, đánh giá định lượng, backend API và frontend trực quan. Kết quả cho thấy mô hình phân đoạn phổi đạt độ chính xác cao, trong khi mô hình ISIC đạt kết quả khả quan cho cả segmentation và detection."))
    body.append(para("Hệ thống có thể tiếp tục phát triển theo hướng quản lý dữ liệu, lưu lịch sử phân tích, hỗ trợ nhiều định dạng ảnh y tế hơn và cải thiện khả năng triển khai thực tế."))

    body.append(section_title("11. Phụ lục"))
    body.append(table([
        ["Lệnh", "Mục đích"],
        ["uvicorn api.main:app --reload --port 8000", "Chạy backend."],
        ["npm run dev", "Chạy frontend tại http://localhost:5173."],
        ["python scripts/evaluate.py --model segmentation ...", "Đánh giá segmentation và xuất CSV per-image."],
        ["python scripts/evaluate.py --model detection ...", "Đánh giá detection và xuất CSV per-image."],
    ], [4300, 5060]))

    sect = (
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body)}{sect}</w:body></w:document>"
    )


STYLES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:after="160"/></w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="46"/><w:color w:val="000000"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:after="320"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="28"/><w:color w:val="555555"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:keepNext/><w:spacing w:before="320" w:after="160"/></w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="32"/><w:color w:val="2E74B5"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr><w:b/><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="26"/><w:color w:val="2E74B5"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="TableText">
    <w:name w:val="Table Text"/><w:basedOn w:val="Normal"/>
    <w:pPr><w:spacing w:after="0" w:line="264" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="20"/></w:rPr>
  </w:style>
</w:styles>
"""


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""


CORE = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Báo cáo cuối kì MedSeg</dc:title>
  <dc:creator>MedSeg</dc:creator>
  <cp:lastModifiedBy>MedSeg</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-06-25T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-06-25T00:00:00Z</dcterms:modified>
</cp:coreProperties>
"""


APP = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Application>MedSeg Report Generator</Application>
</Properties>
"""


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/_rels/document.xml.rels", DOC_RELS)
        z.writestr("word/document.xml", build_document_xml())
        z.writestr("word/styles.xml", STYLES_XML)
        z.writestr("docProps/core.xml", CORE)
        z.writestr("docProps/app.xml", APP)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
