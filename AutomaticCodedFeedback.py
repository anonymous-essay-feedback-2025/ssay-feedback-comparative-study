"""
Automatic Essay Annotator - Python GUI Version with Azure Computer Vision API

A full-featured essay annotation tool that:
1. Uploads student essay images (handwritten/printed)
2. Runs OCR using Azure Computer Vision API (excellent accuracy!)
3. Uses Google Gemini API to find writing issues
4. Overlays color-coded annotations on the original image

Requirements:
pip install pillow google-generativeai pydantic requests

Setup:
1. Create Azure Computer Vision resource: https://portal.azure.com
2. Get Gemini API key: https://aistudio.google.com/

Environment variables needed:
- GEMINI_API_KEY=your_gemini_api_key_here
- AZURE_COMPUTER_VISION_KEY=your_azure_api_key_here
- AZURE_COMPUTER_VISION_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import os
import json
import threading
import io
import time
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
from pydantic import BaseModel, Field

# Configure Gemini API
# Option 1: Use environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Option 2: Hardcode your API key (less secure but easier)
# Uncomment the line below and replace with your actual API key
# GEMINI_API_KEY = "your_actual_api_key_here"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Check for Azure API credentials
AZURE_API_KEY = os.getenv('AZURE_COMPUTER_VISION_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_COMPUTER_VISION_ENDPOINT')

# Data structures
class IssueType(str, Enum):
    SPELLING = "spelling"
    GRAMMAR = "grammar"
    STYLE = "style"
    PUNCTUATION = "punctuation"

@dataclass
class BoundingBox:
    x0: int
    y0: int
    x1: int
    y1: int

@dataclass
class OCRToken:
    text: str
    bbox: BoundingBox

class Issue(BaseModel):
    type: IssueType
    tokenIndices: List[int] = Field(min_items=1)
    span: Optional[Dict[str, int]] = None
    suggestion: Optional[str] = None
    message: str

class AnalysisResponse(BaseModel):
    issues: List[Issue]

class AzureOCRProcessor:
    """Handles OCR using Azure Computer Vision API"""
    
    def __init__(self):
        if not AZURE_API_KEY or not AZURE_ENDPOINT:
            raise Exception("AZURE_COMPUTER_VISION_KEY and AZURE_COMPUTER_VISION_ENDPOINT environment variables must be set!")
        
        self.api_key = AZURE_API_KEY
        self.endpoint = AZURE_ENDPOINT.rstrip('/')  # Remove trailing slash if present
        print("Using Azure Computer Vision API for OCR")
    
    def extract_text_with_positions(self, image: Image.Image) -> List[OCRToken]:
        """Extract text with bounding box positions using Azure Computer Vision API"""
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_data = img_byte_arr.getvalue()
        
        # Step 1: Submit image for analysis
        analyze_url = f"{self.endpoint}/vision/v3.2/read/analyze"
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Content-Type': 'application/octet-stream'
        }
        
        try:
            print("=== Azure Computer Vision API Debug ===")
            print(f"Endpoint: {analyze_url}")
            print(f"Image size: {len(img_data)} bytes")
            
            # Submit for analysis
            response = requests.post(analyze_url, headers=headers, data=img_data, timeout=30)
            print(f"Submit Status Code: {response.status_code}")
            
            if response.status_code != 202:
                error_detail = response.text
                print(f"Submit Error Response: {error_detail}")
                raise Exception(f"Azure API submission failed: {response.status_code} - {error_detail}")
            
            # Get operation location from response headers
            if 'Operation-Location' not in response.headers:
                raise Exception("No Operation-Location header in Azure response")
            
            operation_url = response.headers["Operation-Location"]
            print(f"Operation URL: {operation_url}")
            
            # Step 2: Poll for results
            poll_headers = {'Ocp-Apim-Subscription-Key': self.api_key}
            max_attempts = 30  # Wait up to 30 seconds
            
            for attempt in range(max_attempts):
                print(f"Polling attempt {attempt + 1}/{max_attempts}")
                time.sleep(1)
                
                result_response = requests.get(operation_url, headers=poll_headers, timeout=10)
                print(f"Poll Status Code: {result_response.status_code}")
                
                if result_response.status_code != 200:
                    print(f"Poll Error: {result_response.text}")
                    continue
                
                result_json = result_response.json()
                status = result_json.get("status", "unknown")
                print(f"Analysis Status: {status}")
                
                if status == "succeeded":
                    print("Analysis completed successfully!")
                    return self._parse_azure_results(result_json)
                elif status == "failed":
                    error_msg = result_json.get("message", "Unknown failure")
                    raise Exception(f"Azure analysis failed: {error_msg}")
                elif status in ["notStarted", "running"]:
                    continue  # Keep polling
                else:
                    print(f"Unknown status: {status}")
            
            raise Exception("Azure API polling timeout - analysis took too long")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error calling Azure API: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response from Azure API: {str(e)}")
        except Exception as e:
            raise Exception(f"Azure API error: {str(e)}")
    
    def _parse_azure_results(self, result_json: dict) -> List[OCRToken]:
        """Parse Azure Computer Vision API results into OCRToken list"""
        tokens = []
        
        try:
            analyze_result = result_json.get("analyzeResult", {})
            read_results = analyze_result.get("readResults", [])
            
            print(f"Found {len(read_results)} pages")
            
            for page_idx, page in enumerate(read_results):
                lines = page.get("lines", [])
                print(f"Page {page_idx} has {len(lines)} lines")
                
                for line_idx, line in enumerate(lines):
                    words = line.get("words", [])
                    print(f"Line {line_idx} has {len(words)} words")
                    
                    for word in words:
                        word_text = word.get("text", "").strip()
                        bbox_coords = word.get("boundingBox", [])
                        
                        if word_text and len(bbox_coords) >= 8:
                            # Azure returns bounding box as [x1,y1,x2,y2,x3,y3,x4,y4]
                            # We need to find min/max coordinates
                            x_coords = [bbox_coords[i] for i in range(0, 8, 2)]
                            y_coords = [bbox_coords[i] for i in range(1, 8, 2)]
                            
                            bbox = BoundingBox(
                                x0=int(min(x_coords)),
                                y0=int(min(y_coords)),
                                x1=int(max(x_coords)),
                                y1=int(max(y_coords))
                            )
                            
                            tokens.append(OCRToken(text=word_text, bbox=bbox))
                            print(f"Added word: '{word_text}' at ({bbox.x0},{bbox.y0},{bbox.x1},{bbox.y1})")
            
            print(f"Total tokens extracted: {len(tokens)}")
            print("=== End Debug ===")
            return tokens
            
        except Exception as e:
            print(f"Error parsing Azure results: {e}")
            print(f"Result JSON: {json.dumps(result_json, indent=2)}")
            raise Exception(f"Failed to parse Azure API results: {str(e)}")

class EssayAnnotatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gemini Essay Annotator")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f8f9fa')
        
        # Initialize OCR processor
        try:
            self.ocr_processor = AzureOCRProcessor()
        except Exception as e:
            messagebox.showerror("OCR Error", str(e))
            root.destroy()
            return
        
        # State variables
        self.original_image = None
        self.annotated_image = None
        self.display_image = None
        self.ocr_tokens: List[OCRToken] = []
        self.issues: List[Issue] = []
        self.scale_factor = 1.0
        
        self.setup_ui()
        self.update_status("Ready (OCR: Azure Computer Vision) - Upload an essay photo")

    def setup_ui(self):
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="Gemini Essay Annotator", 
                               font=('Arial', 18, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(header_frame, text="Ready", 
                                     font=('Arial', 10))
        self.status_label.pack(side=tk.RIGHT)
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (image and controls)
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Controls
        controls_frame = ttk.Frame(left_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(controls_frame, text="Upload Image", 
                  command=self.upload_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Test OCR", 
                  command=self.test_ocr).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Annotate", 
                  command=self.start_annotation).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Clear", 
                  command=self.clear_all).pack(side=tk.LEFT, padx=(0, 5))
        
        # Progress bar
        self.progress = ttk.Progressbar(controls_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Image display area
        image_frame = ttk.LabelFrame(left_frame, text="Essay Image", padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbars
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='white', scrollregion=(0, 0, 1000, 1000))
        
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Right panel (legend and issues)
        right_frame = ttk.Frame(content_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        # Legend
        legend_frame = ttk.LabelFrame(right_frame, text="Legend", padding=10)
        legend_frame.pack(fill=tk.X, pady=(0, 10))
        
        legend_items = [
            ("Spelling error", "red", "underline"),
            ("Grammar error", "gold", "underline"),
            ("Style / word choice", "lightblue", "highlight"),
            ("Missing punctuation", "red", "cross")
        ]
        
        for text, color, style in legend_items:
            item_frame = ttk.Frame(legend_frame)
            item_frame.pack(fill=tk.X, pady=2)
            
            # Visual indicator
            indicator = tk.Label(item_frame, width=3, height=1)
            if style == "underline":
                indicator.configure(bg=color, text="___")
            elif style == "highlight":
                indicator.configure(bg=color, text="   ")
            else:  # cross
                indicator.configure(bg="white", text="✕", fg=color)
            indicator.pack(side=tk.LEFT, padx=(0, 10))
            
            ttk.Label(item_frame, text=text, font=('Arial', 9)).pack(side=tk.LEFT)
        
        # Issues list
        issues_frame = ttk.LabelFrame(right_frame, text="Issues", padding=10)
        issues_frame.pack(fill=tk.BOTH, expand=True)
        
        self.issues_text = scrolledtext.ScrolledText(issues_frame, height=15, 
                                                    font=('Arial', 9), wrap=tk.WORD)
        self.issues_text.pack(fill=tk.BOTH, expand=True)
        
        # Footer
        footer_frame = ttk.Frame(main_frame)
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(footer_frame, 
                 text="Tip: Azure Computer Vision provides excellent accuracy. Create your resource at portal.azure.com",
                 font=('Arial', 8), foreground='gray').pack()

    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def test_ocr(self):
        """Test OCR on current image and show raw results"""
        if not self.original_image:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
        
        try:
            # Test the OCR directly
            tokens = self.ocr_processor.extract_text_with_positions(self.original_image)
            
            # Show results in a popup
            test_window = tk.Toplevel(self.root)
            test_window.title("OCR Test Results")
            test_window.geometry("600x400")
            
            text_widget = scrolledtext.ScrolledText(test_window, wrap=tk.WORD, font=('Courier', 10))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            text_widget.insert(tk.END, f"Total words found: {len(tokens)}\n")
            text_widget.insert(tk.END, "="*50 + "\n\n")
            
            if tokens:
                text_widget.insert(tk.END, "Extracted words:\n")
                for i, token in enumerate(tokens[:50]):  # Show first 50 words
                    text_widget.insert(tk.END, f"{i}: '{token.text}' at ({token.bbox.x0},{token.bbox.y0})\n")
                
                if len(tokens) > 50:
                    text_widget.insert(tk.END, f"\n... and {len(tokens) - 50} more words")
                
                # Show full text
                full_text = " ".join([token.text for token in tokens])
                text_widget.insert(tk.END, f"\n\nFull extracted text:\n{full_text}")
            else:
                text_widget.insert(tk.END, "No words found. Check the console output for debug information.")
            
        except Exception as e:
            messagebox.showerror("OCR Test Error", f"OCR test failed: {str(e)}")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Essay Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                self.original_image = Image.open(file_path)
                self.display_image_on_canvas()
                self.ocr_tokens = []
                self.issues = []
                self.update_issues_display()
                
                # Start OCR in background
                self.start_ocr()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def display_image_on_canvas(self):
        if not self.original_image:
            return
            
        # Calculate scale factor to fit canvas
        canvas_width = self.canvas.winfo_width() or 800
        canvas_height = self.canvas.winfo_height() or 600
        
        img_width, img_height = self.original_image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.scale_factor = min(scale_x, scale_y, 1.0)  # Don't scale up
        
        # Resize image for display
        display_width = int(img_width * self.scale_factor)
        display_height = int(img_height * self.scale_factor)
        
        self.display_image = self.original_image.resize((display_width, display_height), 
                                                       Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage and display
        self.photo_image = ImageTk.PhotoImage(self.display_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def start_ocr(self):
        self.update_status("Running OCR (Azure Computer Vision)...")
        self.progress.start()
        
        def ocr_worker():
            try:
                # Use the OCR processor
                tokens = self.ocr_processor.extract_text_with_positions(self.original_image)
                self.ocr_tokens = tokens
                
                # Update UI in main thread
                self.root.after(0, self.ocr_complete)
                
            except Exception as e:
                self.root.after(0, lambda: self.ocr_error(str(e)))
        
        threading.Thread(target=ocr_worker, daemon=True).start()

    def ocr_complete(self):
        self.progress.stop()
        self.update_status(f"OCR complete: {len(self.ocr_tokens)} words found. Click 'Annotate' to analyze.")

    def ocr_error(self, error_msg):
        self.progress.stop()
        self.update_status("OCR failed")
        messagebox.showerror("OCR Error", f"Failed to process image: {error_msg}")

    def start_annotation(self):
        if not self.ocr_tokens:
            messagebox.showwarning("Warning", "Please upload an image and wait for OCR to complete first.")
            return
            
        if not GEMINI_API_KEY:
            messagebox.showerror("Error", "GEMINI_API_KEY environment variable not set!")
            return
        
        self.update_status("Calling Gemini API...")
        self.progress.start()
        
        def annotation_worker():
            try:
                # Prepare data for API
                plain_text = " ".join([token.text for token in self.ocr_tokens])
                tokens_data = [{"text": token.text} for token in self.ocr_tokens]
                
                # Call Gemini API
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                system_prompt = """You are an English writing assessor. Given a student's text and a list of word tokens (in order), find writing issues and return STRICT JSON matching the responseSchema.

Categories:
- spelling: misspelled words.
- grammar: subject-verb agreement, tense, articles, prepositions, etc.
- style: awkward phrasing, word choice, redundancy; use sparingly.
- punctuation: missing commas/periods/apostrophes/quotation marks etc.

Rules:
- Refer to tokens by their zero-based indices in the provided token list.
- For multi-word issues, either list all tokenIndices OR provide a span {start,end}.
- Keep messages short and actionable.
- Provide 'suggestion' when a specific fix is obvious.
- Return only valid JSON matching the schema."""

                user_prompt = f"""Tokens (zero-based):
{' '.join([f"{i}:{token['text']}" for i, token in enumerate(tokens_data)])}

Plain text (for context):
{plain_text}

Return JSON with this exact structure:
{{
  "issues": [
    {{
      "type": "spelling|grammar|style|punctuation",
      "tokenIndices": [0, 1, 2],
      "span": {{"start": 0, "end": 2}},
      "suggestion": "optional suggestion",
      "message": "brief description"
    }}
  ]
}}"""

                response = model.generate_content(
                    [system_prompt, user_prompt],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                    )
                )
                
                # Parse response
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3].strip()
                
                try:
                    result_data = json.loads(response_text)
                    analysis = AnalysisResponse(**result_data)
                    self.issues = analysis.issues
                except Exception as parse_error:
                    raise Exception(f"Failed to parse API response: {parse_error}\nResponse: {response_text}")
                
                # Update UI in main thread
                self.root.after(0, self.annotation_complete)
                
            except Exception as e:
                self.root.after(0, lambda: self.annotation_error(str(e)))
        
        threading.Thread(target=annotation_worker, daemon=True).start()

    def annotation_complete(self):
        self.progress.stop()
        self.update_status(f"Annotated: {len(self.issues)} issues found")
        self.update_issues_display()
        self.draw_annotations()

    def annotation_error(self, error_msg):
        self.progress.stop()
        self.update_status("Annotation failed")
        messagebox.showerror("Annotation Error", f"Failed to analyze text: {error_msg}")

    def update_issues_display(self):
        self.issues_text.delete(1.0, tk.END)
        
        if not self.issues:
            self.issues_text.insert(tk.END, "(No issues found yet)")
            return
        
        for i, issue in enumerate(self.issues):
            self.issues_text.insert(tk.END, f"{i+1}. {issue.type.upper()}\n")
            self.issues_text.insert(tk.END, f"   {issue.message}")
            if issue.suggestion:
                self.issues_text.insert(tk.END, f" → {issue.suggestion}")
            self.issues_text.insert(tk.END, "\n\n")

    def draw_annotations(self):
        if not self.display_image or not self.issues:
            return
        
        # Create a copy of the display image for drawing
        annotated = self.display_image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Color mapping
        colors = {
            IssueType.SPELLING: "red",
            IssueType.GRAMMAR: "gold",
            IssueType.STYLE: "lightblue",
            IssueType.PUNCTUATION: "red"
        }
        
        for issue in self.issues:
            try:
                # Get token boxes for this issue
                token_boxes = []
                for idx in issue.tokenIndices:
                    if 0 <= idx < len(self.ocr_tokens):
                        bbox = self.ocr_tokens[idx].bbox
                        # Scale coordinates
                        scaled_bbox = BoundingBox(
                            x0=int(bbox.x0 * self.scale_factor),
                            y0=int(bbox.y0 * self.scale_factor),
                            x1=int(bbox.x1 * self.scale_factor),
                            y1=int(bbox.y1 * self.scale_factor)
                        )
                        token_boxes.append(scaled_bbox)
                
                # Add span tokens if specified
                if issue.span:
                    for idx in range(issue.span["start"], issue.span["end"] + 1):
                        if 0 <= idx < len(self.ocr_tokens) and idx not in issue.tokenIndices:
                            bbox = self.ocr_tokens[idx].bbox
                            scaled_bbox = BoundingBox(
                                x0=int(bbox.x0 * self.scale_factor),
                                y0=int(bbox.y0 * self.scale_factor),
                                x1=int(bbox.x1 * self.scale_factor),
                                y1=int(bbox.y1 * self.scale_factor)
                            )
                            token_boxes.append(scaled_bbox)
                
                if not token_boxes:
                    continue
                
                color = colors.get(issue.type, "black")
                
                if issue.type in [IssueType.SPELLING, IssueType.GRAMMAR]:
                    # Draw underlines
                    for bbox in token_boxes:
                        y = bbox.y1 + 2
                        draw.line([(bbox.x0, y), (bbox.x1, y)], fill=color, width=3)
                        
                elif issue.type == IssueType.STYLE:
                    # Draw highlight
                    if token_boxes:
                        # Draw semi-transparent rectangle
                        for bbox in token_boxes:
                            draw.rectangle([bbox.x0, bbox.y0, bbox.x1, bbox.y1], 
                                         fill=color, outline=color)
                        
                elif issue.type == IssueType.PUNCTUATION:
                    # Draw crosses
                    for bbox in token_boxes:
                        x = bbox.x1 + 5
                        y = (bbox.y0 + bbox.y1) // 2
                        size = 6
                        # Draw X
                        draw.line([(x-size, y-size), (x+size, y+size)], fill=color, width=2)
                        draw.line([(x+size, y-size), (x-size, y+size)], fill=color, width=2)
                        
            except Exception as e:
                print(f"Error drawing annotation: {e}")
        
        # Update display
        self.annotated_image = annotated
        self.photo_image = ImageTk.PhotoImage(annotated)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def clear_all(self):
        self.original_image = None
        self.annotated_image = None
        self.display_image = None
        self.ocr_tokens = []
        self.issues = []
        self.canvas.delete("all")
        self.update_issues_display()
        self.update_status("Ready (OCR: Azure Computer Vision) - Upload an essay photo")

def main():
    # Check for required dependencies
    if not AZURE_API_KEY or not AZURE_ENDPOINT:
        messagebox.showerror("Error", 
                           "Azure Computer Vision credentials not set!\n\n"
                           "Required environment variables:\n"
                           "- AZURE_COMPUTER_VISION_KEY=your_api_key\n"
                           "- AZURE_COMPUTER_VISION_ENDPOINT=https://your-resource.cognitiveservices.azure.com/\n\n"
                           "Create your resource at: https://portal.azure.com")
        return
    
    # Check for Gemini API key
    if not GEMINI_API_KEY:
        messagebox.showwarning("Warning", 
                             "GEMINI_API_KEY environment variable not set!\n"
                             "The annotation feature will not work without it.")
    
    root = tk.Tk()
    app = EssayAnnotatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
