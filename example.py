# Simulate prediction on a new article using the trained model
example_article = "Government announces new health policies to improve public care."

# Transform and predict
example_vec = vectorizer.transform([example_article])
prediction = model.predict(example_vec)[0]
label = "Real" if prediction == 1 else "Fake"

# Create image showing input and prediction
from PIL import Image, ImageDraw, ImageFont

output_text = f"""
Fake News Detector - Example Prediction

Input Article:
"{example_article}"

Predicted Label:
{label}
"""

# Create image from text
img = Image.new('RGB', (800, 300), color=(255, 255, 255))
draw = ImageDraw.Draw(img)
font = ImageFont.load_default()
draw.multiline_text((10, 10), output_text, fill=(0, 0, 0), font=font)

example_output_path = "/mnt/data/Fake_News_Detector_Example_Output.png"
img.save(example_output_path)

example_output_path
