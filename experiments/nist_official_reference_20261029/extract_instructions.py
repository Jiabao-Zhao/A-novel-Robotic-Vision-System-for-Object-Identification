"""Read official ATB1 DOCX text/tables/figures without modifying the source.

This is document inspection, not an assembly/grasp experiment. Paragraph indices
are OOXML paragraph positions, not rendered page numbers. All nine figures were
visually inspected separately when producing instructions_findings.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'outputs/simulation/nist_task_board_1/assets/assembly_instructions.docx'
OUTPUT = ROOT/'outputs/nist_official_reference_20261029/instructions'
NS = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def extract():
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE) as z:
        relationships = {r.attrib['Id']:r.attrib['Target']
                         for r in ET.fromstring(z.read('word/_rels/document.xml.rels'))}
        root = ET.fromstring(z.read('word/document.xml'))
        paragraphs = []
        for index, paragraph in enumerate(root.findall('.//w:p', NS)):
            text = ''.join(t.text or '' for t in paragraph.findall('.//w:t', NS))
            images = [relationships.get(t.attrib.get('{'+NS['r']+'}embed'))
                      for t in paragraph.findall('.//a:blip', NS)]
            if text or images:
                paragraphs.append(dict(paragraph_index=index, text=text, images=images))
        for member in z.namelist():
            if member.startswith('word/media/'):
                (OUTPUT/Path(member).name).write_bytes(z.read(member))
        tables = []
        for index, table in enumerate(root.findall('.//w:tbl', NS)):
            rows = [[''.join(t.text or '' for t in cell.findall('.//w:t', NS))
                     for cell in row.findall('./w:tc', NS)]
                    for row in table.findall('./w:tr', NS)]
            tables.append(dict(table_index=index, rows=rows))
        result = dict(source_path=SOURCE.relative_to(ROOT).as_posix(),
                      source_sha256=before, paragraphs=paragraphs, tables=tables)
        (OUTPUT/'extraction.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        (OUTPUT/'text.txt').write_text('\n'.join(
            f"{p['paragraph_index']:03d}: {p['text']} IMAGES={p['images']}" for p in paragraphs)+'\n',
            encoding='utf-8')
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before
    return result


if __name__ == '__main__':
    result = extract()
    print(json.dumps(dict(paragraphs=len(result['paragraphs']), tables=len(result['tables']),
                          original_DOCX_unchanged=True)))
