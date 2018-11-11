import itertools
import re
import sys
import xml.etree as et
from xml.etree import ElementTree as et
import os
from collections import defaultdict
from typing import List, Mapping

WORD_NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def is_subject_bookmark(paragraph: et.Element):
    bookmark_p = paragraph.find('w:bookmarkStart', namespaces=WORD_NS)
    if bookmark_p is None or '_GoBack' in bookmark_p.attrib.values():
        return False

    return True


def has_style(paragraph: et.Element, style=None):
    x_path = 'w:pPr//w:pStyle' if style is None else 'w:pPr//w:pStyle[@{http://schemas.openxmlformats.org' \
                                                     '/wordprocessingml/2006/main}val="' + \
                                                     str(style) + '"]'

    style_val = paragraph.find(x_path, namespaces=WORD_NS)
    return style_val is not None


def is_speaker(paragraph: et.Element):
    return has_style(paragraph, 'a4')


def is_speaker_remark(paragraph: et.Element):
    return has_style(paragraph, 'a5')


def is_styled(paragraph: et.Element):
    return has_style(paragraph)


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

        if is_speaker_remark(paragraph):
            continue

        if is_styled(paragraph):
            speaker = None
            continue

        if len(extract_text(paragraph)) == 0:
            continue

        speakers[speaker].append(paragraph)

    return speakers


def tokenize_word(current_word: str, state='middle') -> List[str]:
    tokens = tokenize_word_reversed(current_word, state)
    tokens.reverse()
    return tokens


def tokenize_word_reversed(current_word: str, state='middle') -> List[str]:
    if len(current_word) == 1:
        return [current_word]

    tokens = []

    # End of sentence tokens should be tokenized away
    if state == 'end' and re.match('^.*?[?!.;]$', current_word) is not None:
        tokens.append(current_word[-1])
        current_word = current_word[:-1]

    if len(current_word) == 0:
        return tokens

    # Handle comma or : - we assume that the following constructs are invalid: ,:  :,
    if current_word.endswith(',') or current_word.endswith(':'):
        tokens.append(current_word[-1])
        current_word = current_word[:-1]

    if len(current_word) == 0:
        return tokens

    # Get rid of ) or " from end of word if exists
    token_matcher = re.match('^.*?([")]+)$', current_word)
    if token_matcher is not None:
        symbols_str = token_matcher.group(1)
        symbols = list(symbols_str)
        symbols.reverse()
        tokens += symbols

        current_word = current_word[:-len(symbols_str)]

    if len(current_word) == 0:
        return tokens

    # Tokenize forms of ha"bla or be(ble away
    token_matcher = re.match('^(\w)?([("]+)(.*)$', current_word)
    if token_matcher is not None:
        tokens.append(token_matcher.group(3))
        symbols = list(token_matcher.group(2))
        symbols.reverse()
        tokens += symbols

        if token_matcher.group(1):
            tokens.append(token_matcher.group(1))

        current_word = ''

    if len(current_word) == 0:
        return tokens

    tokens.append(current_word)
    return tokens


def tokenize_sentence(sentence: str) -> str:
    tokens = []
    state = 'start'
    words = sentence.split(' ')
    for idx, word in enumerate(words):
        state = 'end' if idx == (len(words) - 1) else state
        tokens += tokenize_word(word, state)
        state = 'middle'

    return " ".join(tokens)


def transform_paragraphs(speaker: str, text_paragraphs: List[str]) -> str:
    # Normalize/pre-process - replace weird hypes with regular ones
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
        map(lambda paragraph: re.split('(?<=(?!^\d)\w[.?!])\s', paragraph, flags=re.UNICODE), clean_paragraphs))

    # Now, split <letter>;<space>
    clean_sentences = itertools.chain.from_iterable(
        map(lambda sentence: re.split('(?<=\w;)\s', sentence), clean_sentences))

    # Get rid of (-\s+){2,} at the end and the beginning, also - some file has _ _ _ in it, a mistake ?
    clean_sentences = map(
        lambda sentence: re.sub('((\s+\u2013){2,}$|^(\u2013\s+){2,}|(_\s+){2}_)', '', sentence, flags=re.UNICODE),
        clean_sentences)

    # trim spaces
    clean_sentences = map(lambda sentence: sentence.strip(), clean_sentences)

    # Get rid of empty sentences
    clean_sentences = filter(lambda paragraph: len(paragraph) != 0, clean_sentences)

    # Hypens of all types got seperated, we need to "reconnect" words with regular 45 hypens
    clean_sentences = map(lambda sentence: re.sub(' - ', '-', sentence), clean_sentences)

    # number relation letters should be tokenized me-15 -> me - 15
    clean_sentences = map(lambda sentence: re.sub(r'(?<=\b\D)-(\d+)', r' - \1', sentence, flags=re.UNICODE),
                          clean_sentences)

    clean_sentences = list(clean_sentences)  # Unwind filters
    tokenized_sentences = []
    for sentence in clean_sentences:
        tokenized_sentence = tokenize_sentence(sentence)
        tokenized_sentences.append(tokenized_sentence)

    return "\n".join(tokenized_sentences)


def transform(extracted_by_speaker: Mapping[str, List[et.Element]]) -> Mapping[str, str]:
    def speaker_paragraphs_handler(s, ps):
        texts = [extract_text(paragraph) for paragraph in ps]
        return transform_paragraphs(s, texts)

    return {re.sub('\s{2,}', ' ', speaker): speaker_paragraphs_handler(speaker, paragraphs) for (speaker, paragraphs) in
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

    et.ElementTree(root).write(filename_output, encoding='unicode')
    return True


def main(argv):
    dirname_input = argv[1]
    filename_output = argv[2]

    source = os.path.join(dirname_input, 'word', 'document.xml')
    tree = et.parse(os.path.abspath(source))
    result = extract(tree)
    result = transform(result)
    result = load(filename_output, result)


if __name__ == "__main__":
    main(sys.argv)
