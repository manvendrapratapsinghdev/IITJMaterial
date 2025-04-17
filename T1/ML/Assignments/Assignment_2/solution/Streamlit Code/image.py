import pillow_avif
from PIL import Image, ImageDraw, UnidentifiedImageError

# Image dimensions for the banner
width, height = 1200, 300

# Create a blue-themed background
background = Image.new("RGB", (width, height), "#d0e7ff")

# Load the left and right images
try:
    left_image = Image.open("left.avif").convert("RGBA")
    right_image = Image.open("right.avif").convert("RGBA")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except UnidentifiedImageError as e:
    print(f"Error: {e}. Ensure the AVIF plugin is installed and the file is valid.")
    exit()

# Resize the images to fit the banner height while maintaining aspect ratio
left_image = left_image.resize((int(height * left_image.width / left_image.height), height))
right_image = right_image.resize((int(height * right_image.width / right_image.height), height))

# Paste the left image on the left side of the banner
background.paste(left_image, (0, 0), left_image)

# Paste the right image on the right side of the banner
background.paste(right_image, (width - right_image.width, 0), right_image)

# Draw some decorative elements (e.g., heartbeat line)
draw = ImageDraw.Draw(background)
heartbeat_color = "#004080"
line_y = height // 2

# Draw a simple heartbeat line in the center
points = [
    (width // 4, line_y),
    (width // 4 + 50, line_y - 30),
    (width // 4 + 100, line_y + 30),
    (width // 4 + 150, line_y - 30),
    (width // 4 + 200, line_y)
]
draw.line(points, fill=heartbeat_color, width=3)

# Save the final banner image
try:
    background.save("diabetes_prediction_app_banner.png", format="PNG", dpi=(300, 300))
    print("Banner image saved as 'diabetes_prediction_app_banner.png'")
    background.show()
except Exception as e:
    print(f"Error saving or displaying the image: {e}")
