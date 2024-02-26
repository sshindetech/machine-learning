import logging
import os
import pypdfium2 as pdfium

from machine_learning.utils.chroma_client import DocumentEmbeddingsClient
import machine_learning.utils.constants as CONST;

class PDFToImageConverter:
    def __init__(
        self, 
        doc_path = CONST.DOCUMENT_SOURCE):
        self.doc_path = doc_path 
            
    def __get_images_from_pdf(self, pdf_path, img_dump_path):
        """
        Extract images from each page of a PDF document and save as JPEG files.

        :param pdf_path: A string representing the path to the PDF file.
        :param img_dump_path: A string representing the path to dummp images.
        """
        pdf = pdfium.PdfDocument(pdf_path)
        
        images = []
        n_pages = len(pdf)
        for page_number in range(n_pages):
            page = pdf.get_page(page_number)
            bitmap = page.render(scale=1, rotation=0, crop=(0, 0, 0, 0))
            pil_image = bitmap.to_pil()
            image_path = f"{img_dump_path}/img_{page_number + 1}.jpg"
            pil_image.save(image_path, format="JPEG")
            images.append(image_path)
            logging.info(f"Saving Image {image_path}")
            
        return images
    
    def convert(self):
        # Load PDF
        pdf_doc_path = os.path.join(self.doc_path, 'sample_deck.pdf')               
        rel_img_dump_path = os.path.join(self.doc_path, 'images')
        
        image_uris = self.__get_images_from_pdf(pdf_doc_path, rel_img_dump_path)

        # Get image URIs
        # image_uris = sorted(
        #     [
        #         os.path.join(rel_img_dump_path, image_name)
        #         for image_name in os.listdir(rel_img_dump_path)
        #         if image_name.endswith(".jpg")
        #     ]
        # )
        
        return sorted(image_uris)
           

