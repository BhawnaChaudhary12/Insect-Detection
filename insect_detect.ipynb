import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import io
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output

model = MobileNetV2(weights='imagenet')
upload = widgets.FileUpload(accept='image/tt.jppeg', multiple=False)
display(upload)

def on_upload_change(change):
    clear_output(wait=True)
    display(upload)

    for name, file_info in upload.value.items():
        img_data = file_info['content']
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        preds = model.predict(img_array)
        decoded = decode_predictions(preds, top=5)[0]
        plt.imshow(img)
        plt.axis('off')
        plt.title("Predictions:")
        plt.show()

        for (_, label, prob) in decoded:
            print(f"{label}: {prob:.4f}")

        insect_keywords = ['insect', 'bug', 'ant', 'bee', 'fly', 'wasp', 'mosquito', 'grasshopper', 'beetle', 'caterpillar']
        if any(keyword in label.lower() for (_, label, _) in decoded for keyword in insect_keywords):
            print("\n Insect detected in the image.")
        else:
            print("\n No insect detected.")

upload.observe(on_upload_change, names='value')
