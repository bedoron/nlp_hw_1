import itertools
import re
import sys
import xml.etree as et
from xml.etree import ElementTree as et
from xml.dom import minidom
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


def is_styled(paragraph: et.Element):
    style_val = paragraph.find(
        'w:pPr//w:pStyle', namespaces=WORD_NS)

    return style_val is not None


def extract_text(paragraph: et.Element, delimiter=" "):
    return delimiter.join([t for t in paragraph.itertext()]).strip()


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

        if is_styled(paragraph):
            speaker = None
            continue

        if len(extract_text(paragraph)) == 0:
            continue

        speakers[speaker].append(paragraph)

    return speakers


def transform_paragraphs(speaker: str, paragraphs: List[et.Element]) -> str:
    text_paragraphs = [extract_text(paragraph) for paragraph in paragraphs]

    # Normalize - replace weird hypes with regular ones
    # clean_paragraphs = map(lambda paragraph: paragraph.replace("\u2013", chr(45)), text_paragraphs)
    # Remove '- - -' (Abrupt stop)
    clean_paragraphs = map(lambda paragraph: re.sub('\u2013\s+\u2013\s+\u2013', '', paragraph, flags=re.UNICODE),
                           text_paragraphs)
    # Remove extra spaces
    clean_paragraphs = map(lambda paragraph: re.sub('\s{2,}', ' ', paragraph), clean_paragraphs)

    # Split sentences by '<letter>. ' and flatten the list
    # The followind exepciton are made:
    #  enumeration, ie. 1. <some text>
    clean_sentences = itertools.chain.from_iterable(
        map(lambda paragraph: re.split('(?<=\S\D\.)\s', paragraph), clean_paragraphs))

    # Now, split <letter>;<space>
    clean_sentences = itertools.chain.from_iterable(
        map(lambda sentence: re.split('(?=\S;)\s', sentence), clean_sentences))

    # Get rid of (-\s+){2,} at the end and the beginning
    clean_sentences = map(
        lambda sentence: re.sub('((\s+\u2013){2,}$|^(\u2013\s+){2,})', '', sentence, flags=re.UNICODE),
        clean_sentences)

    # trim spaces
    clean_sentences = map(lambda sentence: sentence.strip(), clean_sentences)

    # Get rid of empty sentences
    clean_sentences = filter(lambda paragraph: len(paragraph) != 0, clean_sentences)

    clean_sentences = list(clean_sentences)  # Unwind filters
    return "\n".join(clean_sentences)


def transform(extracted_by_speaker: Mapping[str, List[et.Element]]) -> Mapping[str, str]:
    return {speaker: transform_paragraphs(speaker, paragraphs) for (speaker, paragraphs) in
            extracted_by_speaker.items()}


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

    xml_str = minidom.parseString(et.tostring(root, encoding='unicode', method='xml')).toprettyxml(indent='    ')
    xml_str = xml_str[xml_str.find("?>") + 3:]  # Get rid of xml header
    with open(filename_output, 'wb') as f:
        f.write(xml_str.encode('utf-16'))

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
