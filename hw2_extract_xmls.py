import os
from sys import argv
import glob
import zipfile
import tempfile
import hw1
import xml.etree as et
from xml.etree import ElementTree as et
from collections import defaultdict


def extract_docs(docs_source):
    documents = []
    target_base = os.path.join(tempfile.gettempdir(), 'nlp')
    if os.path.exists(target_base):
        return list(glob.glob(os.path.join(target_base, '*')))

    for docx in glob.glob(os.path.join(docs_source, '*docx')):
        zfh = zipfile.ZipFile(docx, 'r')
        fname = os.path.basename(docx)
        target_folder = os.path.join(target_base, fname)
        zfh.extractall(target_folder)
        zfh.close()
        documents.append(target_folder)

    return documents


def build_speakers_files(documents_xmls):
    already_handles = glob.glob(os.path.join('resources', '*.docx.xml'))
    if len(already_handles) == len(documents_xmls):
        return already_handles

    handled_outputs = []
    for xml in documents_xmls:
        print("Handling ", xml)
        output_xml = os.path.join('resources', '{}.xml'.format(os.path.basename(xml)))
        hw1.main([None, xml, output_xml])
        handled_outputs.append(output_xml)

    return handled_outputs


def extract_to_map(speaker_file, speaker_to_speeches):
    root = et.parse(os.path.abspath(speaker_file)).getroot()
    for doc in root:
        speaker_to_speeches[doc[0].text].append(doc[1].text if doc[1].text is not None else '')


def store_results(speaker_to_speeches, target_file):
    root = et.Element('root')
    for speaker, quoted in speaker_to_speeches.items():
        doc = et.Element('doc')

        name = et.Element('name')
        name.text = speaker

        text = et.Element('text')
        text.text = "\n".join(quoted)

        doc.append(name)
        doc.append(text)

        root.append(doc)

    et.ElementTree(root).write(target_file, encoding='unicode')
    return True


def merge_speaker_files(speakers_files, target_file):
    speaker_to_speeches = defaultdict(list)
    for speaker_file in speakers_files:
        extract_to_map(speaker_file, speaker_to_speeches)

    # Now store it as merged format
    store_results(speaker_to_speeches, target_file)


def main(docs_source):
    documents_xmls = extract_docs(docs_source)
    speakers_files = build_speakers_files(documents_xmls)
    merge_speaker_files(speakers_files, os.path.join('resources', 'merged.xml'))


if __name__ == "__main__":
    main(*argv[1:])
