import json
import re
import subprocess
import os
from pathlib import Path
from typing import Literal
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info

# Set up azcopy authentication
os.environ["AZCOPY_AUTO_LOGIN_TYPE"] = "AZCLI"

# ============ CONFIGURATION ============
CIRCLE_RADIUS = 10  # Radius of red circles in pixels
CIRCLE_COLOR = "red"
CIRCLE_WIDTH = 3  # Width of circle outline (only used if CIRCLE_FILLED is False)
CIRCLE_FILLED = True  # Whether to draw filled circles (True) or just outlines (False)
SHOW_COORDINATES = True  # Whether to display coordinate labels next to each point
LABEL_COLOR = "white"  # Color for coordinate labels
LABEL_OFFSET = 10  # Pixel offset for label from the circle center
COORDINATE_ORIGIN: Literal["top-left", "bottom-left"] = "top-left"  # Options: "top-left" or "bottom-left"
BASE_BLOB_URL = "https://magmardata.blob.core.windows.net/public"
OUTPUT_DIR = Path("./annotated_images")
IMAGES_DIR = OUTPUT_DIR / "images"
LATEX_DIR = OUTPUT_DIR / "latex"

# Image resizing parameters (similar to Qwen processor)
RESIZE_IMAGES = True  # Whether to resize images before annotating (using Qwen processor logic)
MIN_PIXELS = 3136  # Minimum number of pixels (e.g., 56x56)
MAX_PIXELS = 262144  # Maximum number of pixels (e.g., 512x512)

# ============ INPUT DATA ============
input_data = [
    '{"data_type": "image", "path": [{"type": "image", "image": "video-r1/data/Video-R1-data/General/A-OKVQA/images/508985.jpg"}], "conversations": [{"role": "user", "content": [{"type": "image", "image": "video-r1/data/Video-R1-data/General/A-OKVQA/images/508985.jpg"}, {"type": "text", "text": "Is there a vehicle with a spare tire visible on its exterior? A. No. B. Yes."}]}, {"role": "assistant", "content": [{"type": "text", "text": "<think>Let me analyze the image carefully while identifying the vehicle with a spare tire.\\n\\nScanning the scene, I need to look for vehicles that show a spare tire on their exterior. I\'ll move systematically through the image looking at each vehicle.\\n\\nI notice a dark-colored jeep or utility vehicle on the right side of the image, which appears to have a spare tire mounted on its exterior. Let me focus on the exact location of this vehicle.\\n\\nLooking at the right side of the image, I can see a vehicle that appears to be a jeep or utility vehicle. Upon closer inspection, I can confirm there\'s a spare tire mounted on the back of this vehicle, which is clearly visible. This appears to be the vehicle with the spare tire on its exterior.\\n\\nI observe a jeep type vehicle at approximately <points x1=\\"408\\" y1=\\"179\\" alt=\\"jeep or utility vehicle\\">jeep or utility vehicle</points> with a spare tire visible at its rear. The spare tire appears to be mounted on the exterior of this vehicle, making it clearly visible.\\n\\nTherefore, the answer should be B. Yes. There is a vehicle with a spare tire visible on its exterior.</think><answer>I examine the scene systematically from left to right. As I scan the right side of the image toward the middle-right area, I notice a dark-colored utility vehicle that appears to be a jeep type. On its exterior, I can clearly see a spare tire mounted on the back of the vehicle. This spare tire is visible and prominently displayed on the side of the vehicle\'s exterior structure. The vehicle is positioned near the right edge of the scene, traveling on the right side of the road. I confirm this is indeed a vehicle with a spare tire visible on its exterior.\\n\\n<|begin_of_box|>B. Yes.<|end_of_box|>"}]}], "uuid": "a2b63f5f-7497-5d63-bcf0-184e412b4120", "dataset_name": "video-r1", "correct_answer": "B"}',
    '{"data_type": "image", "path": [{"type": "image", "image": "video-r1/data/Video-R1-data/Math/CLEVR-Math/images/CLEVR_train_068973.png"}], "conversations": [{"role": "user", "content": [{"type": "image", "image": "video-r1/data/Video-R1-data/Math/CLEVR-Math/images/CLEVR_train_068973.png"}, {"type": "text", "text": "How many objects are there in total?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "<think>Looking at the original image, I need to count and identify all objects. Starting from the left side, I see a large brown cylinder positioned at around the left-center area. Moving rightward, there is a small brown sphere near the center of the image. Continuing further right, there is a gray metallic cube. At the top right area, I see a small yellow cube. \\n\\nTo confirm with spatial awareness, I scan from left to right: the large brown cylinder is at <points x1=\\"172\\" y1=\\"111\\" alt=\\"large brown cylinder\\">large brown cylinder</points> area, the small brown sphere is positioned closer to the center around <points x2=\\"247\\" y2=\\"153\\" alt=\\"small brown sphere\\">small brown sphere</points>, the large gray metallic cube is at <points x3=\\"335\\" y3=\\"182\\" alt=\\"large gray metallic cube\\">large gray metallic cube</points>, and the small yellow cube is positioned at the top right area at about <points x4=\\"279\\" y4=\\"101\\" alt=\\"small yellow cube\\">small yellow cube</points>. \\n\\nCounting these objects, there are 4 distinct objects total. I observe a large brown cylinder at <points x1=\\"172\\" y1=\\"111\\" alt=\\"large brown cylinder\\">large brown cylinder</points>, a small brown sphere at <points x2=\\"247\\" y2=\\"153\\" alt=\\"small brown sphere\\">small brown sphere</points>, a large gray metallic cube at <points x3=\\"335\\" y3=\\"182\\" alt=\\"large gray metallic cube\\">large gray metallic cube</points>, and a small yellow cube at <points x4=\\"279\\" y4=\\"101\\" alt=\\"small yellow cube\\">small yellow cube</points>. Therefore, there are 4 objects in total.</think><answer>Looking at the scene, I observe a large brown cylinder at position <points x1=\\"172\\" y1=\\"111\\" alt=\\"large brown cylinder\\">large brown cylinder</points> on the left side. Moving rightward toward the center, I notice a small brown sphere at <points x2=\\"247\\" y2=\\"153\\" alt=\\"small brown sphere\\">small brown sphere</points>. Continuing further right, there is a large gray metallic cube at position <points x3=\\"335\\" y3=\\"182\\" alt=\\"large gray metallic cube\\">large gray metallic cube</points>. In the upper right area of the image, I see a small yellow cube at position <points x4=\\"279\\" y4=\\"101\\" alt=\\"small yellow cube\\">small yellow cube</points>. Counting all these distinct objects, there are 4 objects in total.\\n<|begin_of_box|>4<|end_of_box|>"}]}], "uuid": "ee16c240-ac4d-554d-9ad5-607de4b981d4", "dataset_name": "video-r1", "correct_answer": "4"}',

    # these two don't have nice alignment of points
    # r"""{"data_type": "image", "path": [{"type": "image", "image": "video-r1/data/Video-R1-data/Knowledge/ScienceQA/images/17878.png"}], "conversations": [{"role": "user", "content": [{"type": "image", "image": "video-r1/data/Video-R1-data/Knowledge/ScienceQA/images/17878.png"}, {"type": "text", "text": "Is there a toy in the table that involves water play? A. No. B. Yes."}]}, {"role": "assistant", "content": [{"type": "text", "text": "<think>Looking at the image, I see a table with two columns - \"Inventor\" and \"Toy\". Let me scan through the toys listed:\n\n1. At <points x1=\"198\" y1=\"40\" alt=\"Rubik's Cube\">Rubik's Cube</points>, I can see \"Rubik's Cube\" as the toy by Ernisse Rubik\n2. At <points x2=\"230\" y2=\"100\" alt=\"Barbie doll\">Barbie doll</points>, I see \"Barbie doll\" by Ruth Handler\n3. At <points x3=\"242\" y3=\"133\" alt=\"Super Soaker\">Super Soaker</points>, I see \"Super Soaker\" by Lonnie Johnson\n\nI need to determine if any of these toys involves water play. The Rubik's Cube is a puzzle toy, the Barbie doll is a fashion toy, and the Super Soaker appears to be related to water since \"Soaker\" suggests water play equipment.\n\nLooking more carefully at the Super Soaker entry, it's clearly listed as a water-related toy. The name \"Super Soaker\" suggests it's designed for water play, possibly being a water gun or similar water-dispensing toy.\n\nBased on the table, there is indeed a toy that involves water play, which is the Super Soaker invented by Lonnie Johnson.</think><answer>I examine the table showing inventors and their toys. Looking at the toy column, I see Rubik's Cube, Barbie doll, and Super Soaker. Analyzing each entry's function, I notice the Super Soaker stands out as likely related to water play given its name. Scanning from left to right through the toy entries, specifically at position <points x3=\"242\" y3=\"133\" alt=\"Super Soaker\">Super Soaker</points> where I find \"Super Soaker\" associated with Lonnie Johnson, this appears designed for water-related activities. Therefore, there is a toy in the table that involves water play.\n<|begin_of_box|>B<|end_of_box|>"}]}], "uuid": "6b1fc4bf-2b31-56a6-b9eb-6019cba616bb", "dataset_name": "video-r1", "correct_answer": "B"}""",
    # r"""{"data_type": "image", "path": [{"type": "image", "image": "video-r1/data/Video-R1-data/OCR/Rendered_Text/cauldron/rendered_text/images/rendered_text_00008898.png"}], "conversations": [{"role": "user", "content": [{"type": "image", "image": "video-r1/data/Video-R1-data/OCR/Rendered_Text/cauldron/rendered_text/images/rendered_text_00008898.png"}, {"type": "text", "text": "Extract all text content from the photo."}]}, {"role": "assistant", "content": [{"type": "text", "text": "<think>Looking at the original image, I need to identify and report all text content with their coordinate positions. The text appears to be on a piece of paper that is on a textured surface.\n\nStarting from the upper portion of the paper, I can first identify \"I love\" written at the top. As I move down, I see \"learning\" next, followed by \"new\", then \"dance\", and finally \"steps.\" at the bottom.\n\nI'll track each word with precise coordinates based on where I observe them to be in the image.\n\nFor \"I love\", I observe it positioned near the top-left area of the paper.\nFor \"learning\", I see it below \"I love\" on the left side.\nFor \"new\", I note its position roughly in the center area of the paper.\nFor \"dance\", I observe it positioned below \"new\".\nFor \"steps.\" at the bottom right area of the paper.\n\nI'll structure my response naturally as if I'm analyzing the image without mentioning the second reference image or overlays.\n\nLooking at the scene, I observe \"I love\" written near the top portion of the paper around coordinate <points x1=\"126\" y1=\"101\" alt=\"I love\">I love</points>. Scanning downward, I notice \"learning\" positioned near coordinate <points x2=\"129\" y2=\"134\" alt=\"learning\">learning</points>. Continuing downward, I observe \"new\" near the center area around coordinate <points x3=\"241\" y3=\"219\" alt=\"new\">new</points>. Moving further down, I see \"dance\" near coordinate <points x4=\"249\" y4=\"291\" alt=\"dance\">dance</points>. Finally, I identify \"steps.\" written at the bottom right area around coordinate <points x5=\"291\" y5=\"365\" alt=\"steps.\">steps.</points>.\n\nI need to extract all the text content exactly as I observe it, not mentioning the second image.</think><answer>I observe \"I love\" written near the top left portion of the paper at approximately <points x1=\"126\" y1=\"101\" alt=\"I love\">I love</points>. Scanning downward, I notice \"learning\" positioned below it roughly at <points x2=\"129\" y2=\"134\" alt=\"learning\">learning</points>. Continuing my analysis, I identify \"new\" written centrally on the paper near coordinate <points x3=\"241\" y3=\"219\" alt=\"new\">new</points>. Moving further down, I see \"dance\" positioned at approximately <points x4=\"249\" y4=\"291\" alt=\"dance\">dance</points>. Finally, I observe \"steps.\" written at the bottom right area of the paper around coordinate <points x5=\"291\" y5=\"365\" alt=\"steps.\">steps.</points>. All text is written in purple pen on a white piece of paper that appears to be on a rough textured surface. The complete text sequence is: \"I love learning new dance steps.\"\nAll text extracted is: <|begin_of_box|>I love learning new dance steps.<|end_of_box|>"}]}], "uuid": "f4935429-eade-5f18-8efc-cd98f54bed32", "dataset_name": "video-r1", "correct_answer": "I love\nlearning\nnew\ndance\nsteps."}""",

    r"""{"data_type": "image", "path": [{"type": "image", "image": "video-r1/data/Video-R1-data/General/A-OKVQA/images/558011.jpg"}], "conversations": [{"role": "user", "content": [{"type": "image", "image": "video-r1/data/Video-R1-data/General/A-OKVQA/images/558011.jpg"}, {"type": "text", "text": "What is at the top of this structure in the middle of the city square? A. church tower. B. advertisement. C. latitude. D. optiplex."}]}, {"role": "assistant", "content": [{"type": "text", "text": "<think>Let me analyze the image focusing on the structure in the middle of the city square. I need to identify what's at the top of this structure.\n\nI observe a prominent building in the center of the image which appears to be the structure in question. Looking at the top of this building:\n- There's a spire-like structure at the top\n- There's a clock visible on what appears to be a tower\n- The building has architectural features typical of a church tower or bell tower\n\nLooking more carefully at the top of this central building specifically, I can see that there's a tall spire or tower element which has a distinctive shape typical of church architecture. The building appears to be historic with ornate architectural details around what would typically be a church tower or clock tower.\n\nNow let me consider the answer choices:\nA. church tower - This seems most likely given the architectural style with the spire, the historic nature of the building, and the prominent tower\nB. advertisement - There doesn't appear to be an advertisement at the top, rather architectural elements\nC. latitude - This is a directional coordinate measurement, not something you'd find at the top of a structure\nD. optiplex - This appears to be related to computer equipment, not something that would be at the top of a building\n\nBased on the visual evidence of the spire/tower structure with architectural details like what appears to be a clock, this fits the description of a church tower rather than an advertisement, latitude (which is not applicable), or optiplex (which is irrelevant).\n\nI observe a distinctive spire and tower at the top of the central building around <points x1=\"165\" y1=\"156\" alt=\"church tower\">church tower</points> area, which has architectural characteristics typical of a church tower.</think><answer>Looking at the central building in the city square, I can see it has a distinctive tower structure. Examining the top portion of this central structure, I observe a prominent spire-like element at the very top, approximately at the middle-upper section of the beige building with the clock. The architectural style includes ornate details and the tower appears to be an elevated structure with a pointed or tapered top that's characteristic of religious or civic architecture. Given the building's historic appearance with its detailed facade, corner tower, and clock, the structure at the top is most likely a church tower. None of the other options (advertisement, latitude, or optiplex) would naturally be found as a prominent feature at the top of this architectural structure.\n\nMy answer is <|begin_of_box|>A<|end_of_box|>."}]}], "uuid": "1f82389b-dc45-5176-8278-3167dad5016e", "dataset_name": "video-r1", "correct_answer": "A"}""",
    r"""{"data_type": "image", "path": [{"type": "image", "image": "video-r1/data/Video-R1-data/Knowledge/ScienceQA/images/4923.png"}], "conversations": [{"role": "user", "content": [{"type": "image", "image": "video-r1/data/Video-R1-data/Knowledge/ScienceQA/images/4923.png"}, {"type": "text", "text": "Which continent is being indicated in the highlighted area? A. Australia. B. Antarctica. C. Africa."}]}, {"role": "assistant", "content": [{"type": "text", "text": "<think>Let me analyze the original image carefully. I'm looking for the highlighted area that indicates a specific continent.\n\nLooking at the original image (first image), I can see a world map with blue oceans and green land masses. There's a distinct green landmass at the bottom of the image that appears to be completely surrounded by water. This is typically Antarctica on a world map.\n\nNow I need to focus on that highlighted area specifically. Looking at the bottom portion of the original image, there is indeed a green landmass at the southernmost part of the map. It appears completely isolated by water, which is characteristic of Antarctica.\n\nLooking at the second image (with coordinate overlays), I can see a coordinate at <points x1=\"427\" y1=\"346\" alt=\"green landmass at the southernmost part of the map\">green landmass at the southernmost part of the map</points> which appears to be pointing to the green area at the very bottom of the map, confirming that it is the southernmost continent.\n\nBased on the position at the bottom of the world map, completely separated by water from all other continents, I can identify this as Antarctica. This matches with the characteristics of Antarctica being located at the South Pole region.\n\nThe answer should be B. Antarctica.</think><answer>I examine the original world map, focusing on the lower portion where there is a distinct green landmass completely surrounded by water at the southernmost region of the map. Looking at this highlighted area, I note that it's positioned at the bottom of the map, separated by ocean from all other continents. This corresponds to the continent located at the South Pole. This geographic characteristic - isolated at the southernmost point of the map, completely surrounded by water - clearly indicates Antarctica. The distinctive positioning and separation from other landmasses confirms this is the proper identification.\n<|begin_of_box|>B. Antarctica.<|end_of_box|>"}]}], "uuid": "5df51f14-2f5e-56ee-a3bf-39aaa555d791", "dataset_name": "video-r1", "correct_answer": "B"}""",
]


def smart_resize(width, height, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Calculate resize dimensions using qwen_vl_utils process_vision_info."""
    # Use qwen_vl_utils to handle image resizing the same way as Qwen processor
    # Create a dummy message to use process_vision_info
    # We'll create a PIL image of the specified size to get the resizing logic
    from PIL import Image
    dummy_img = Image.new('RGB', (width, height))
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": dummy_img, "min_pixels": min_pixels, "max_pixels": max_pixels}
        ]
    }]
    
    # Process the image to get the resized dimensions
    images, _ = process_vision_info(messages)
    
    if images and len(images) > 0:
        resized_img = images[0]
        return resized_img.size
    
    # Fallback to original size if something goes wrong
    return width, height


def extract_points(text):
    """Extract all <points> annotations from text."""
    # Pattern to match <points x1="..." y1="..." ...> or <points x2="..." y2="..." ...> etc.
    # This will match x followed by any number and y followed by any number
    pattern = r'<points\s+x\d+="(\d+)"\s+y\d+="(\d+)"(?:\s+[^>]*)?>'
    matches = re.findall(pattern, text)
    return [(int(x), int(y)) for x, y in matches]


def download_image(partial_path, output_path):
    """Download image using azcopy."""
    full_url = f"{BASE_BLOB_URL}/{partial_path}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "azcopy", "copy", full_url, str(output_path),
        "--overwrite=false",
        "--check-md5", "FailIfDifferent",
        "--from-to=BlobLocal",
        "--log-level=INFO"
    ]
    
    print(f"Downloading: {full_url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error downloading {full_url}: {result.stderr}")
        return False
    
    return True


def annotate_image(image_path, points, output_path):
    """Create annotated version of image with red circles at specified points."""
    img = Image.open(image_path)
    original_width, original_height = img.size
    
    print(f"Original image size: {original_width}x{original_height}")
    print(f"Original points: {points}")
    
    if RESIZE_IMAGES:
        # Use qwen_vl_utils to resize the image the same way as the processor
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
            ]
        }]
        
        # Process the image to get the resized version
        result = process_vision_info(messages)
        if len(result) == 2:
            images, videos = result
        else:
            images, videos, _ = result
        
        if images and len(images) > 0:
            resized_img = images[0]
            new_width, new_height = resized_img.size
            
            # Calculate scaling factors
            scale_x = new_width / original_width
            scale_y = new_height / original_height
            
            print(f"Resized image size: {new_width}x{new_height}")
            print(f"Scaling factors: x={scale_x:.4f}, y={scale_y:.4f}")
            
            if (new_width, new_height) != (original_width, original_height):
                print(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
            
            img = resized_img
        else:
            # Fallback if processing fails
            print("Warning: Image processing failed, using original image")
    else:
        print("Image resizing disabled, using original image")
    
    draw = ImageDraw.Draw(img)
    
    # Try to load a font for labels, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    for x, y in points:
        # Use original coordinates without scaling
        print(f"Drawing circle at original point ({x}, {y})")
        
        # Draw circle at each point (origin is top-left)
        left_up_point = (x - CIRCLE_RADIUS, y - CIRCLE_RADIUS)
        right_down_point = (x + CIRCLE_RADIUS, y + CIRCLE_RADIUS)
        
        if CIRCLE_FILLED:
            draw.ellipse(
                [left_up_point, right_down_point],
                fill=CIRCLE_COLOR,
                outline=CIRCLE_COLOR
            )
        else:
            draw.ellipse(
                [left_up_point, right_down_point],
                outline=CIRCLE_COLOR,
                width=CIRCLE_WIDTH
            )
        
        # Add coordinate label if enabled
        if SHOW_COORDINATES:
            label = f"({x},{y})"
            # Position label offset from the circle
            label_x = x + LABEL_OFFSET
            label_y = y - LABEL_OFFSET 
            
            # Draw text with a background for better visibility
            bbox = draw.textbbox((label_x, label_y), label, font=font)
            draw.rectangle(bbox, fill="black")
            draw.text((label_x, label_y), label, fill=LABEL_COLOR, font=font)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"Saved annotated image: {output_path}")


def generate_latex(uuid, points, user_text, reasoning_text, output_path):
    """Generate LaTeX file for the figure with original and annotated images side by side."""
    
    # Function to highlight <points> tags in red
    def highlight_points(text):
        """Wrap entire <points>...</points> blocks with red color in LaTeX."""
        # Pattern to match complete <points ...>content</points> blocks
        text = re.sub(r'(<points[^>]*>.*?</points>)', r'@\\textcolor{red}{\1}@', text, flags=re.DOTALL)
        return text
    
    # Highlight points in both user and reasoning text
    user_text_highlighted = highlight_points(user_text)
    reasoning_text_highlighted = highlight_points(reasoning_text)
    
    latex_content = r"""\begin{figure*}[t]
\centering
\begin{AIBox}{Image Example - """ + uuid + r"""}
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{sec/figures/im_visuals/""" + uuid + r""".jpg}
    \caption*{Original Image}
\end{minipage}
\hfill
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{sec/figures/im_visuals/""" + uuid + r"""_annotated.jpg}
    \caption*{Annotated Image}
\end{minipage}

\vspace{0.5em}

\small
\textbf{Task:}

\begin{lstlisting}[breaklines=true, basicstyle=\small\ttfamily, breakindent=0pt, escapechar=@]
""" + user_text_highlighted + r"""
\end{lstlisting}

\vspace{0.3em}
\textbf{Agent Response:}

\begin{lstlisting}[breaklines=true, basicstyle=\small\ttfamily, breakindent=0pt, escapechar=@]
""" + reasoning_text_highlighted + r"""
\end{lstlisting}
\end{AIBox}
\caption{Example}
\label{fig:annotated_""" + uuid + r"""}
\end{figure*}
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"Saved LaTeX file: {output_path}")


def process_sample(sample_json):
    """Process a single sample: download image, extract points, create annotated version."""
    # Handle multi-line JSON strings with control characters
    sample_json = sample_json.strip()
    # Replace literal \n in strings with actual newlines for JSON parsing
    sample = json.loads(sample_json)
    
    uuid = sample["uuid"]
    partial_path = sample["path"][0]["image"]
    
    # Extract user text and assistant text separately
    user_text = ""
    assistant_text = ""
    
    for conv in sample["conversations"]:
        if "content" in conv:
            role = conv.get("role", "")
            for content_item in conv["content"]:
                if content_item.get("type") == "text":
                    text = content_item.get("text", "")
                    if role == "user":
                        user_text += text
                    elif role == "assistant":
                        assistant_text += text
    
    # Extract points from all text
    all_text = user_text + assistant_text
    points = extract_points(all_text)
    
    if not points:
        print(f"Warning: No points found for sample {uuid}")
        return
    
    print(f"\nProcessing sample {uuid}")
    print(f"Found {len(points)} point(s): {points}")
    
    # Download original image
    original_path = IMAGES_DIR / f"{uuid}.jpg"
    if not original_path.exists():
        if not download_image(partial_path, original_path):
            print(f"Failed to download image for {uuid}")
            return
    
    # Create annotated version
    annotated_path = IMAGES_DIR / f"{uuid}_annotated.jpg"
    annotate_image(original_path, points, annotated_path)
    
    # Generate LaTeX file
    latex_path = LATEX_DIR / f"{uuid}.tex"
    generate_latex(uuid, points, user_text, assistant_text, latex_path)
    
    print(f"✓ Completed processing {uuid}")


def main():
    """Process all samples in input_data."""
    print(f"Configuration:")
    print(f"  Circle radius: {CIRCLE_RADIUS}px")
    print(f"  Circle color: {CIRCLE_COLOR}")
    print(f"  Resize images: {RESIZE_IMAGES}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"\nProcessing {len(input_data)} sample(s)...\n")
    
    for sample_json in input_data:
        try:
            process_sample(sample_json)
        except Exception as e:
            print(f"Error processing sample: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ All done! Check {OUTPUT_DIR} for results.")


if __name__ == "__main__":
    main()
