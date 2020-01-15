import mimetypes
import os

import magic
import textract
from chardet.universaldetector import UniversalDetector

__all__ = ["get_content_type", "get_file_extensions", "get_file_encoding", "extract_text_from_file"]

ALLOWED_EXTENSIONS = ("csv", "doc", "docx", "eml", "epub", "gif", "jpg", "jpeg", "json", "html", "htm",
                      "msg", "odt", "pdf", "png", "pptx", "ps", "rtf", "tiff", "txt", "xls", "xlsx")


def get_content_type(file_path):
    """
    This function return the content type of a file
    :param file_path: address where file is
    :return: a content type string
    """

    try:
        magic_obj = magic.Magic(mime=True)
        magic_obj.file = magic_obj.from_file
    except AttributeError as e:
        magic_obj = magic.open(magic.MAGIC_MIME_TYPE)
        magic_obj.load()

    content_type = magic_obj.file(file_path)
    return content_type


def content_type_to_extensions(content_type):
    mime_type = content_type.split("; ")[0]
    return mimetypes.guess_all_extensions(mime_type)


def get_file_extensions(file_path):
    content_type = get_content_type(file_path)
    return content_type_to_extensions(content_type)


def get_file_encoding(file_path):
    detector = UniversalDetector()
    # for line in open(file_path, 'rb'):
    #     detector.feed(line)
    #     if detector.done:
    #         break
    with open(file_path, 'rb') as reader:
        while not detector.done:
            chunk = reader.read(1024000)
            if not chunk:
                break
            detector.feed(chunk)
    detector.close()
    return detector.result["encoding"]


def create_directory(directory_path):
    """
    This function validate that exist a directory, if not, then try to create
    :param directory_path:
    :return:
    """
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        os.makedirs(directory_path)
    return True


def extract_text_from_file(file_path):
    extension = None
    file_name = os.path.basename(file_path)
    extension_list = file_name.split(".")

    if len(extension_list) > 1:
        if extension_list[-1] not in ALLOWED_EXTENSIONS:
            raise TypeError(".{} extension can't be processed".format(extension_list[-1]))
        extension = extension_list[-1]

    else:
        candidates = get_file_extensions(file_path)
        for ext in candidates:
            ext = ext.strip(".")
            if ext not in ALLOWED_EXTENSIONS:
                continue
            extension = ext
        if extension is None:
            raise TypeError("Cannot define a valid extension for file '{}'".format(file_name))

    # encoding = get_file_encoding(file_path) or 'utf-8'

    # Because some .txt files fails in textract because being decoded only with UTF-8, we do it manually
    if extension == "txt":
        try:
            with open(file_path) as reader:
                text = reader.read()
        except UnicodeDecodeError:
            encoding = get_file_encoding(file_path)
            if encoding is None:
                raise UnicodeError("Cannot define encoding for file {}".format(file_name))
            with open(file_path, encoding=encoding, errors='replace') as reader:
                text = reader.read()
    else:
        try:
            text = textract.process(file_path, extension=extension)
        except UnicodeDecodeError:
            raise UnicodeError("Cannot define encoding for file {}".format(file_name))

    if not isinstance(text, str):
        text = text.decode('utf-8', 'replace')
    return text
