import sys
import xml.etree.ElementTree as et
import os
from collections import defaultdict
from typing import List, Mapping

WORD_NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def is_subject_bookmark(paragraph: et.Element):
    bookmark_p = paragraph.find('w:bookmarkStart', namespaces=WORD_NS)
    if bookmark_p is None or '_GoBack' in bookmark_p.attrib.values():
        return False

    return True


def is_speaker(paragraph: et.Element):
    speaker_val = paragraph.find(
        'w:pPr//w:pStyle[@{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val="a4"]', namespaces=WORD_NS)
    return speaker_val is not None


def extract_text(paragraph: et.Element, delimiter=" "):
    return delimiter.join([t for t in paragraph.itertext()])


def extract(tree: et.ElementTree):
    # Fetch all paragraph ids, we will later use these as anchors
    body = tree.getroot()[0]
    speakers = defaultdict(list)
    speaker = None
    for paragraph in body:
        if is_subject_bookmark(paragraph):
            speaker = None
            continue

        if is_speaker(paragraph):
            speaker = extract_text(paragraph)
            continue

        if speaker is None:
            continue

        if len(extract_text(paragraph)) == 0:
            continue

        speakers[speaker].append(paragraph)

    return speakers


def transform_paragraphs(paragraphs: List[et.Element]) -> str:
    # TODO: Tokenization should come here
    return "\n**\n".join([extract_text(paragraph) for paragraph in paragraphs])


def transform(extracted_by_speaker: Mapping[str, List[et.Element]]) -> Mapping[str, str]:
    return {speaker: transform_paragraphs(paragraphs) for (speaker, paragraphs) in extracted_by_speaker.items()}


def load(filename_output, transformed_by_speaker):
    root = et.Element('root')
    for speaker, quoted in transformed_by_speaker.items():
        doc = et.Element('doc')

        name = et.Element('name')
        name.text = speaker

        text = et.Element('text')
        text.text = quoted

        doc.append(name)
        doc.append(text)

        root.append(doc)

    et.ElementTree(root).write(filename_output, encoding='unicode')
    # with open(filename_output, 'w') as fd:
    #     xml_data = et.tostring(root, encoding='unicode', method='text')
    #     fd.write(xml_data)

    return True


def main(argv):
    dirname_input = argv[1]
    filename_output = argv[2]

    source = os.path.join(dirname_input, 'word', 'document.xml')
    tree = et.parse(source)
    result = extract(tree)
    result = transform(result)
    result = load(filename_output, result)


if __name__ == "__main__":
    main(sys.argv)
