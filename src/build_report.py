"""把 Markdown 报告转换为 Word，再通过本机 Word 导出 PDF。

运行方式（在“作业三”目录下）：
    python src/build_report.py
输出：
    report/作业三_实验报告.docx
    report/2453-孟珂瑶-1120243260-作业三.pdf
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_DIR = os.path.join(ROOT, "report")
MD_PATH = os.path.join(REPORT_DIR, "作业三_实验报告.md")
DOCX_PATH = os.path.join(REPORT_DIR, "作业三_实验报告.docx")
PDF_PATH = os.path.join(REPORT_DIR, "2453-孟珂瑶-1120243260-作业三.pdf")
REF_PATH = os.path.join(REPORT_DIR, "reference.docx")

# 复用作业二目录下已经安装的 pypandoc（避免重复拷贝 200MB 的 pandoc）
PYPANDOC_DIR = os.path.abspath(os.path.join(ROOT, "..", "作业二", "tools", "pypandoc"))
if os.path.isdir(PYPANDOC_DIR):
    sys.path.insert(0, PYPANDOC_DIR)

import pypandoc  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Inches, Mm, Pt, RGBColor, Twips  # noqa: E402
from docx.text.paragraph import Paragraph  # noqa: E402


def make_reference_docx() -> None:
    """如果参考样式不存在，则用 pandoc 默认模板生成一份。"""
    if os.path.exists(REF_PATH):
        return
    pandoc = pypandoc.get_pandoc_path()
    with open(REF_PATH, "wb") as f:
        subprocess.run(
            [pandoc, "--print-default-data-file", "reference.docx"],
            stdout=f,
            check=True,
        )

    doc = Document(REF_PATH)

    def set_east_asia(style, font_name: str) -> None:
        rpr = style.element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = rpr.makeelement(qn("w:rFonts"), {})
            rpr.append(rfonts)
        rfonts.set(qn("w:eastAsia"), font_name)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.paragraph_format.line_spacing = 1.35
    normal.paragraph_format.space_after = Pt(4)
    set_east_asia(normal, "宋体")

    heading_specs = [
        ("Title", 22, "黑体"),
        ("Subtitle", 14, "楷体"),
        ("Heading 1", 16, "黑体"),
        ("Heading 2", 14, "黑体"),
        ("Heading 3", 12, "黑体"),
    ]
    for name, size, east in heading_specs:
        if name not in [s.name for s in doc.styles]:
            continue
        style = doc.styles[name]
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.name = "Times New Roman"
        set_east_asia(style, east)
        style.paragraph_format.space_before = Pt(10)
        style.paragraph_format.space_after = Pt(6)

    doc.save(REF_PATH)


def _display_len(text: str) -> int:
    return sum(2 if ord(ch) > 127 else 1 for ch in text)


def _add_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "8C8C8C")
        borders.append(el)
    tbl_pr.append(borders)


def _add_cell_margins(table) -> None:
    tbl_pr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for tag, value in (("top", 30), ("left", 70), ("bottom", 30), ("right", 70)):
        el = OxmlElement(f"w:{tag}")
        el.set(qn("w:w"), str(value))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tbl_pr.append(mar)


def style_tables(docx_path: str) -> None:
    doc = Document(docx_path)
    section = doc.sections[0]
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    page_width = section.page_width or Inches(8.5)
    left_margin = section.left_margin or Inches(1)
    right_margin = section.right_margin or Inches(1)
    available = int((page_width - left_margin - right_margin) / 635)

    for table in doc.tables:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        _add_borders(table)
        _add_cell_margins(table)

        ncols = len(table.columns)
        col_weights = []
        min_widths = []
        for col in range(ncols):
            longest = 0
            longest_token = 0
            for cell in table.columns[col].cells:
                for line in cell.text.splitlines() or [""]:
                    longest = max(longest, _display_len(line.strip()))
                for token in cell.text.split():
                    if token.isascii() and len(token) >= 10:
                        longest_token = max(longest_token, len(token))
            col_weights.append(max(6, longest))
            header = table.rows[0].cells[col].text.strip() if table.rows else ""
            if header and " " not in header:
                min_width = max(620, _display_len(header) * 95 + 160)
            else:
                min_width = 620
            if longest_token:
                min_width = max(min_width, longest_token * 120 + 160)
            min_widths.append(min_width)

        usable = available - ncols * 140
        weights_sum = sum(col_weights)
        widths = [
            max(min_widths[i], int(usable * w / weights_sum))
            for i, w in enumerate(col_weights)
        ]
        if sum(widths) > usable:
            scale = usable / sum(widths)
            widths = [int(w * scale) for w in widths]

        grid_cols = table._tbl.tblGrid.findall(qn("w:gridCol"))
        for grid_col, width in zip(grid_cols, widths):
            grid_col.set(qn("w:w"), str(width))
        for row in table.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Twips(width)

        tbl_pr = table._tbl.tblPr
        tbl_w = tbl_pr.find(qn("w:tblW"))
        if tbl_w is None:
            tbl_w = OxmlElement("w:tblW")
            tbl_pr.append(tbl_w)
        tbl_w.set(qn("w:type"), "pct")
        tbl_w.set(qn("w:w"), "5000")

        for row_index, row in enumerate(table.rows):
            tr_pr = row._tr.get_or_add_trPr()
            tr_pr.append(OxmlElement("w:cantSplit"))
            if row_index == 0:
                tr_pr.append(OxmlElement("w:tblHeader"))

            for cell in row.cells:
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                if row_index == 0:
                    shading = OxmlElement("w:shd")
                    shading.set(qn("w:val"), "clear")
                    shading.set(qn("w:fill"), "E9EEF7")
                    cell._tc.get_or_add_tcPr().append(shading)
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_before = Pt(1)
                    paragraph.paragraph_format.space_after = Pt(1)
                    paragraph.paragraph_format.line_spacing = 1.0
                    for run in paragraph.runs:
                        run.font.size = Pt(9)
                        if row_index == 0:
                            run.font.bold = True

    for table in doc.tables:
        prev = table._tbl.getprevious()
        if prev is not None and prev.tag == qn("w:p"):
            paragraph = Paragraph(prev, doc)
            if paragraph.text.strip():
                paragraph.paragraph_format.keep_with_next = True
                if paragraph.style is not None and paragraph.style.name == "Table Caption":
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        nxt = table._tbl.getnext()
        if nxt is not None and nxt.tag == qn("w:p"):
            Paragraph(nxt, doc).paragraph_format.space_before = Pt(10)
        if len(table.rows) <= 12:
            for row in table.rows[:-1]:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        paragraph.paragraph_format.keep_with_next = True

    for paragraph in doc.paragraphs:
        if next(paragraph._p.iter(qn("w:drawing")), None) is not None:
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.space_before = Pt(4)
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif paragraph.style is not None and paragraph.style.name == "Image Caption":
            paragraph.paragraph_format.space_before = Pt(3)
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    section.different_first_page_header_footer = True
    footer = section.footer
    footer.is_linked_to_previous = False
    fpara = footer.paragraphs[0]
    fpara.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fpara.add_run()
    run.font.size = Pt(9)
    run.font.name = "Times New Roman"
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)

    doc.save(docx_path)


def export_pdf(docx_path: str, pdf_path: str) -> None:
    """用本机 Microsoft Word 导出 PDF（公式、中文字体都能保留）。"""
    import win32com.client

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    doc = None
    try:
        doc = word.Documents.Open(os.path.abspath(docx_path))
        doc.SaveAs(os.path.abspath(pdf_path), FileFormat=17)  # wdFormatPDF
    finally:
        if doc is not None:
            doc.Close(False)
        word.Quit()
    print(f"[完成] {pdf_path}")


def main() -> None:
    if not os.path.exists(MD_PATH):
        raise FileNotFoundError(f"找不到报告源文件：{MD_PATH}")
    make_reference_docx()

    # 把报告引用的图片复制到 report/figures，保证相对路径可用
    fig_src = os.path.join(ROOT, "results", "figures")
    fig_dst = os.path.join(REPORT_DIR, "figures")
    if os.path.isdir(fig_src):
        os.makedirs(fig_dst, exist_ok=True)
        for name in os.listdir(fig_src):
            if name.lower().endswith(".png"):
                shutil.copy2(os.path.join(fig_src, name), os.path.join(fig_dst, name))

    pypandoc.convert_file(
        MD_PATH,
        "docx",
        outputfile=DOCX_PATH,
        extra_args=[
            f"--reference-doc={REF_PATH}",
            f"--resource-path={REPORT_DIR}",
        ],
    )
    style_tables(DOCX_PATH)
    print(f"[完成] {DOCX_PATH}")
    export_pdf(DOCX_PATH, PDF_PATH)


if __name__ == "__main__":
    main()
