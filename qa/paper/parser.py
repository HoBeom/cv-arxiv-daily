"""
Copyright 2024 - Chansung Park

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# modified by HoBeom
import os

import fitz
import PyPDF2


def extract_text_and_figures(pdf_path):
    """
    Extracts text and figures from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        tuple: A tuple containing two lists:
            * A list of extracted text blocks.
            * A list of extracted figures (as bytes).
    """

    texts = []
    figures = []

    # Open the PDF using PyMuPDF (fitz) for image extraction
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        text = page.get_text('text')  # Extract text as plain text
        texts.append(text)

        # Process images on the page
        image_list = page.get_images()
        for image_index, img in enumerate(image_list):
            xref = img[0]  # Image XREF
            pix = fitz.Pixmap(doc, xref)  # Create Pixmap image

            # Save image in desired format (here, PNG)
            if pix.colorspace.name not in (fitz.csGRAY.name, fitz.csRGB.name):
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes('png')

            figures.append(img_bytes)

    # Extract additional text using PyPDF2 (in case fitz didn't get everything)
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            texts.append(text)

    try:
        os.remove(pdf_path)
    except FileNotFoundError:
        print(f"File '{pdf_path}' not found.")
    except PermissionError:
        print(f"Unable to remove '{pdf_path}'. Check permissions.")

    return texts, figures
