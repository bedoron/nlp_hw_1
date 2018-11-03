import sys
import xml.etree.ElementTree as et
import os
from collections import defaultdict

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


def extract_text(paragraph: et.Element):
    return " ".join([t for t in paragraph.itertext()])

def handle_frequencies(tree: et.ElementTree):
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

    for speaker in speakers.keys():
        print(speaker)

    return speakers


def main(argv):
    dirname_input = argv[1]
    filename_output = argv[2]

    source = os.path.join(dirname_input, 'word', 'document.xml')
    tree = et.parse(source)
    result = handle_frequencies(tree)


if __name__ == "__main__":
    main(sys.argv)
