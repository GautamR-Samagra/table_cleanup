from transformers import TableTransformerModel, TableTransformerConfig
from transformers import AutoImageProcessor, TableTransformerModel
from transformers import DetrFeatureExtractor
from transformers import TableTransformerForObjectDetection
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
from PIL import Image
import pdf2image
import matplotlib.pyplot as plt
import torch
import fitz
import pytesseract
from pytesseract import Output
import cv2
from google.colab.patches import cv2_imshow
import tempfile

class TableToMD:
    def __init__(self):
        self.configuration = TableTransformerConfig()
        self.feature_extractor = DetrFeatureExtractor()
        self.table_detect_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

    def convert_pdf_to_image(self, pdf_path, page_number):
        images = pdf2image.convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
        return images[0]

    def crop_and_save_detected_table(self, image, box, output_path):
        # Convert the tensor to a list of Python floats and then to integers
        box = [int(coord) for coord in box.tolist()]

        # Crop format: (left, upper, right, lower)
        left, upper, right, lower = box
        cropped_image = image.crop((left, upper, right, lower))
        cropped_image.save(output_path)
        return(output_path)

    def replace_pipe(self, df):
        for column in df.columns: 
            if df[column].dtype  == 'object': 
                df[column]=  df[column].str.replace("|", "।")
        df.columns = df.columns.str.replace("|", "।")
        return(df)

    # Add other functions here...

    def extract_table(self, pdf, page_num,tess_lang):
        page = pdf.load_page(page_num)
        pix = page.get_pixmap(dpi=180)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Extract features from the image
        encoding = self.feature_extractor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.table_detect_model(**encoding)

        # Rescale bounding boxes
        width, height = image.size
        target_size = [(height, width)]
        results = self.feature_extractor.post_process_object_detection(outputs, threshold=0.7,
                                                                       target_sizes=target_size)[0]
        if len(results['boxes']) > 0:
            result_bbox = list(np.ceil(results['boxes'][0].numpy()).astype(int))
            # Save the cropped table images and print confidence scores
            for index, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    image_path = temp_file.name  # Get the path of the temporary file

                    self.crop_and_save_detected_table(image, box, image_path)
                    image = Image.open(image_path)
                    tesseract_config = '-l '+tess_lang+' --oem 3 --psm 6'
                    data = pytesseract.image_to_data(image, config=tesseract_config, output_type=pytesseract.Output.DICT)
                    horizontal_lines_bbox, vertical_line_bboxes = self.detect_borderlines(image_path)
                    text_line_bbox, text_line_height = self.get_horizontal_textlines(data, image)
                    text_line_bbox = [bbox for bbox in text_line_bbox if bbox[2] > bbox[3]]
                    _, _, _, new_horizontal_lines_bbox = self.filter_non_intersecting_lines(text_line_bbox,
                                                                                            horizontal_lines_bbox,
                                                                                            horizontal_lines_height=text_line_height / 4)
                    horizontal_lines_bbox_clustered = self.cluster_horizontal_lines(new_horizontal_lines_bbox,
                                                                                     text_line_height)
                    vertical_line_bboxes_clustered = self.cluster_vertical_lines(vertical_line_bboxes,
                                                                                 text_line_height)
                    horizontal_lines_bbox_clustered_filt, _, _, _ = self.filter_non_intersecting_lines(
                        horizontal_lines_bbox_clustered, vertical_line_bboxes_clustered,
                        vertical_lines_width=text_line_height / 8)
                    both_lines = vertical_line_bboxes_clustered + horizontal_lines_bbox_clustered_filt
                    # draw_bboxes_on_image(image_path , both_lines)
                    if (len(vertical_line_bboxes_clustered) > 1) & (
                            len(horizontal_lines_bbox_clustered_filt) > 1):
                        df1 = self.get_df_from_lines(horizontal_lines_bbox_clustered_filt,
                                                     vertical_line_bboxes_clustered, data)
                        df1.columns = df1.iloc[0].to_list()
                        df1 = df1.drop(df1.index[0])
                        df1 = df1.reset_index(drop=True)
                        df2 = self.replace_pipe(df1)
                        md_format_string = df2.to_markdown(index=False)
                        return result_bbox, md_format_string
        else:
            return ()

    # Add other functions here...

    def detect_borderlines(self, file_path, angle_threshold=1, vertical_slope_threshold=10):
        image = cv2.imread(file_path)
        if image is None:
            return None, None, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=300, minLineLength=200, maxLineGap=10)

        horizontal_line_bboxes = []
        vertical_line_bboxes = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Horizontal line check
                if x2 - x1 != 0:  # To avoid division by zero
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) < np.tan(np.radians(angle_threshold)):  # Check for horizontal lines
                        min_x, max_x = min(x1, x2), max(x1, x2)
                        min_y, max_y = min(y1, y2), max(y1, y2)
                        horizontal_line_bboxes.append([min_x, min_y, max_x - min_x, max_y - min_y])

                # Vertical line check
                if abs(x2 - x1) < np.tan(np.radians(90 - angle_threshold)) or abs(slope) > vertical_slope_threshold:  # Check for vertical lines
                    min_x, max_x = min(x1, x2), max(x1, x2)
                    min_y, max_y = min(y1, y2), max(y1, y2)
                    vertical_line_bboxes.append([min_x, min_y, max_x - min_x, max_y - min_y])

        return horizontal_line_bboxes, vertical_line_bboxes


    def filter_non_intersecting_rectangles(self,horizontal_rects, vertical_rects):
        # Keep a copy of the original lines for final output
        original_horizontal_rects = horizontal_rects.copy()
        original_vertical_rects = vertical_rects.copy()

        # Use sets to keep track of indexes of intersecting lines
        intersecting_horizontal_indexes = set()
        intersecting_vertical_indexes = set()

        # Check for intersection between horizontal and vertical lines in one pass
        for i, h_rect in enumerate(horizontal_rects):
            for j, v_rect in enumerate(vertical_rects):
                if self.overlap(h_rect, v_rect):
                    intersecting_horizontal_indexes.add(i)
                    intersecting_vertical_indexes.add(j)

        # Prepare the output lists using original line coordinates
        intersecting_horizontal_lines = [original_horizontal_rects[i] for i in intersecting_horizontal_indexes]
        non_intersecting_horizontal_lines = [original_horizontal_rects[i] for i in range(len(original_horizontal_rects)) if i not in intersecting_horizontal_indexes]
        intersecting_vertical_lines = [original_vertical_rects[j] for j in intersecting_vertical_indexes]
        non_intersecting_vertical_lines = [original_vertical_rects[j] for j in range(len(original_vertical_rects)) if j not in intersecting_vertical_indexes]

        return intersecting_horizontal_lines, non_intersecting_horizontal_lines, intersecting_vertical_lines, non_intersecting_vertical_lines

    def get_horizontal_textlines(self, data, image):
        heights = [data['height'][i] for i in range(len(data['text'])) if int(data['conf'][i]) > 0]
        average_height = np.mean(heights)
        lines = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:
                word_center = data['top'][i] + data['height'][i] / 2
                found_line = False
                for line in lines:
                    if any(abs((data['top'][idx] + data['height'][idx] / 2) - word_center) <= average_height / 2 for idx in line):
                        line.append(i)
                        found_line = True
                        break
                if not found_line:
                    lines.append([i])

        opencv_image = np.array(image)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        line_bboxes = []
        for line in lines:
            centers = [(data['left'][i] + data['width'][i] // 2, data['top'][i] + data['height'][i] // 2) for i in line]
            avg_y = int(np.mean([c[1] for c in centers]))
            min_x = int(min(c[0] for c in centers)) - 10
            max_x = int(max(c[0] for c in centers)) + 10
            line_bboxes.append([min_x, avg_y - int(average_height / 2), max_x - min_x, int(average_height)])

        return line_bboxes ,average_height
	
    def draw_bboxes_on_image(self, image_path, bboxes):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image at {image_path}")
            return None
        image_with_bboxes = image.copy()
        for bbox in bboxes:
            x, y, w, h = bbox  
            cv2.rectangle(image_with_bboxes, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the rectangle
        image_with_bboxes_rgb = cv2.cvtColor(image_with_bboxes, cv2.COLOR_BGR2RGB)

        cv2_imshow( image_with_bboxes_rgb)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()

        return image_with_bboxes_rgb

    def cluster_horizontal_lines(self, horizontal_lines_bbox, h):
        horizontal_lines_bbox.sort(key=lambda x: x[1])
        clusters = []

        for line in horizontal_lines_bbox:
            x, y, w, _ = line
            added_to_cluster = False
            for cluster in clusters:
                avg_y = sum([line[1] for line in cluster]) / len(cluster)
                if abs(y - avg_y) <= h:
                    cluster.append(line)
                    added_to_cluster = True
                    break

            if not added_to_cluster:
                clusters.append([line])
        merged_lines = []
        for cluster in clusters:
            min_x = min(line[0] for line in cluster)
            max_x = max(line[0] + line[2] for line in cluster)
            #avg_y = sum(line[1] for line in cluster) // len(cluster)
            avg_y = int(np.median([line[1] for line in cluster]))
            merged_line = [min_x, avg_y, max_x - min_x, 1]
            merged_lines.append(merged_line)

        return merged_lines

    def cluster_vertical_lines(self, vertical_lines_bbox, v):
        vertical_lines_bbox.sort(key=lambda x: x[0])
        clusters = []
        for line in vertical_lines_bbox:
            x, y, _, h = line
            added_to_cluster = False

            for cluster in clusters:
                avg_x = sum([line[0] for line in cluster]) / len(cluster)
                if abs(x - avg_x) <= v:
                    cluster.append(line)
                    added_to_cluster = True
                    break

            if not added_to_cluster:
                clusters.append([line])

        merged_lines = []
        for cluster in clusters:
            min_y = min(line[1] for line in cluster)
            max_y = max(line[1] + line[3] for line in cluster)
            avg_x = sum(line[0] for line in cluster) // len(cluster)
            merged_line = [avg_x, min_y, 1, max_y - min_y]  # Using 1 as the width for the new line
            merged_lines.append(merged_line)

        return merged_lines

    def find_row(self, y_coord, horizontal_lines):
        for i, line in enumerate(horizontal_lines):
            if y_coord < line[1]: 
                return i
        return len(horizontal_lines)  
    
    def find_column(self, x_coord, vertical_lines):
        for i, line in enumerate(vertical_lines):
            if x_coord < line[0]:  
                return i
        return len(vertical_lines) 

    def get_df_from_lines(self, horizontal_lines_bbox_clustered_filt, vertical_line_bboxes_clustered, data):
        num_rows = len(horizontal_lines_bbox_clustered_filt) + 1 
        num_columns = len(vertical_line_bboxes_clustered) + 1  
        df = pd.DataFrame(index=range(num_rows), columns=range(num_columns))

        # Assign words to DataFrame cells
        for i, word in enumerate(data['text']):
            if data['conf'][i] > 0:  # Check for valid words
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                word_row = self.find_row(y, horizontal_lines_bbox_clustered_filt)
                word_col = self.find_column(x, vertical_line_bboxes_clustered)

                # If the cell already contains text, append the new word
                existing_text = df.at[word_row, word_col]
                if pd.isnull(existing_text) :
                    existing_text = ''
                df.at[word_row, word_col] = (existing_text + " " if existing_text else "") + word


        df.columns = [f"Column {col}" for col in df.columns]
        df.index = [f"Row {row}" for row in df.index]
        return(df)



    class Rect_simple:
	  def __init__(self, x, y, w, h):
	        self.left = x
	        self.right = w
	        self.top = y
	        self.bottom = h
		  
    class Rectangle:
        def __init__(self, x, y, width, height):
            self.left = x
            self.right = x + width
            self.top = y
            self.bottom = y + height

    @staticmethod
    def range_overlap(a_min, a_max, b_min, b_max):
        return (a_min <= b_max) and (b_min <= a_max)

    @staticmethod
    def overlap(rect1, rect2):
        return TableToMD.range_overlap(rect1.left, rect1.right, rect2.left, rect2.right) and \
               TableToMD.range_overlap(rect1.top, rect1.bottom, rect2.top, rect2.bottom)

    
    def calculate_overlap_percentage(self, rect1, rect2):
        # Determine the (x, y) coordinates of the overlap rectangle's bottom-left and top-right corners
        overlap_left = max(rect1.left, rect2.left)
        overlap_right = min(rect1.right, rect2.right)
        overlap_top = min(rect1.top, rect2.top)
        overlap_bottom = max(rect1.bottom, rect2.bottom)

        overlap_width = overlap_right - overlap_left
        overlap_height = overlap_top - overlap_bottom

        if overlap_width <= 0 or overlap_height <= 0:
            return 0, 0  # No overlap

        overlap_area = overlap_width * overlap_height
        overlap_percentage_rect1 = (overlap_area / rect1.area) * 100
        overlap_percentage_rect2 = (overlap_area / rect2.area) * 100

        return overlap_percentage_rect1, overlap_percentage_rect2


    
    def filter_non_intersecting_lines(self,horizontal_lines, vertical_lines, vertical_lines_width=None, horizontal_lines_height=None):
        # Keep a copy of the original lines for final output
        original_horizontal_lines = horizontal_lines.copy()
        original_vertical_lines = vertical_lines.copy()

        # Modify the dimensions for intersection checking if specified
        if not pd.isnull(vertical_lines_width):
            vertical_lines_width = int(np.ceil(vertical_lines_width))
            vertical_lines = [(x, y, vertical_lines_width, h) for x, y, w, h in vertical_lines]

        if not pd.isnull(horizontal_lines_height):
            horizontal_lines_height = int(np.ceil(horizontal_lines_height))
            horizontal_lines = [(x, y, w, horizontal_lines_height) for x, y, w, h in horizontal_lines]

        horizontal_rects = [self.Rectangle(x, y, w, h) for x, y, w, h in horizontal_lines]
        vertical_rects = [self.Rectangle(x, y, w, h) for x, y, w, h in vertical_lines]

        # Use sets to keep track of indexes of intersecting lines
        intersecting_horizontal_indexes = set()
        intersecting_vertical_indexes = set()

        # Check for intersection between horizontal and vertical lines in one pass
        for i, h_rect in enumerate(horizontal_rects):
            for j, v_rect in enumerate(vertical_rects):
                if self.overlap(h_rect, v_rect):
                    intersecting_horizontal_indexes.add(i)
                    intersecting_vertical_indexes.add(j)

        # Prepare the output lists using original line coordinates
        intersecting_horizontal_lines = [original_horizontal_lines[i] for i in intersecting_horizontal_indexes]
        non_intersecting_horizontal_lines = [original_horizontal_lines[i] for i in range(len(original_horizontal_lines)) if i not in intersecting_horizontal_indexes]
        intersecting_vertical_lines = [original_vertical_lines[j] for j in intersecting_vertical_indexes]
        non_intersecting_vertical_lines = [original_vertical_lines[j] for j in range(len(original_vertical_lines)) if j not in intersecting_vertical_indexes]

        return intersecting_horizontal_lines, non_intersecting_horizontal_lines, intersecting_vertical_lines, non_intersecting_vertical_lines

