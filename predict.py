import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your pre-trained model (replace with your model's path)
model = tf.keras.models.load_model("C:/Users/prem5/Desktop/app/finalmodel.h5")

# Class labels (replace with your own class labels)
class_labels = ["Melasma", "Eczema","acne cystic","Rosacea","acne pustular", "Herpes", "Lentigo", "acne comedo"]

# Remedies for each disease
remedies = {
    "Melasma": "Use sunscreen daily and consider topical treatments.",
    "Eczema": "Keep your skin moisturized and avoid triggers.",
    "acne cystic": "Consult a dermatologist for proper treatment options.",
    "Rosacea": "Avoid triggers like spicy foods and use gentle skincare products.",
    "acne pustular": "Use over-the-counter acne treatments and maintain a clean skincare routine.",
    "Herpes": "Consult a healthcare professional for antiviral medication.",
    "Lentigo": "Protect your skin from the sun and consider laser treatments.",
    "acne comedo": "Use gentle cleansers and over-the-counter acne treatments."
}

# Preprocess image for prediction
def preprocess_image(image):
    image = image.resize((293, 192))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Main Streamlit app
def main():
    st.title("Skin Disease Prediction")
    st.write("Upload an image to predict the skin disease.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                preprocessed_image = preprocess_image(image)
                predictions = model.predict(preprocessed_image)
                st.success("Prediction completed!")

                predicted_class_index = np.argmax(predictions)
                predicted_class = class_labels[predicted_class_index]
                accuracy = predictions[0][predicted_class_index] * 100
                remedy = remedies.get(predicted_class, "No remedy information available.")

                st.write(f"Predicted class: {predicted_class}")
                st.write(f"Accuracy: {accuracy:.2f}%")
                st.write(f"Remedy: {remedy}")

if __name__ == "__main__":
    main()
