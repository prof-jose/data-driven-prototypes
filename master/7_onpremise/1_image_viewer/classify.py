"""
Streamlit app to classify images using a pretrained model.
"""


import streamlit as st
from PIL import Image
from transformers import pipeline
from annotated_text import annotated_text


@st.cache_resource
def load_model():
    """Load and return the model (once)"""
    pipe = pipeline(
        "image-classification",
        model="microsoft/resnet-50"
        )
    return pipe


def main():
    """Run the main app."""

    # Load model
    pipe = load_model()

    # App title
    st.title("Upload image")

    # File uploader
    file = st.file_uploader("Choose an image...", type="jpg")

    if file is not None:

        # Load and display the image
        image = Image.open(file)
        st.image(image, caption="Uploaded image.", use_column_width=True)

        # Classify the image
        result = pipe(image)

        # Display the classification results
        for classification in result:
            st.write(f"Label: {classification['label']} ")
            cols = st.columns([0.75, 0.25])
            cols[0].progress(classification["score"])
            with cols[1]:
                annotated_text(
                    (f"{classification['score']*100:.2f}%", "PROB")
                )


if __name__ == "__main__":
    main()
