from .torch_serve import TorchServeManager
from .vllm import VLLMManager
import streamlit as st
from PIL import Image
import requests
import os
import tempfile

class ModelInvoker:
    def __init__(self, servingengine):
        self.serving_engine = servingengine
        self.manager = None

    #Access Images
    def access_images(self, images):
        st.write("Accessing uploaded images:")
        for image in images:
            img = Image.open(image)
            st.image(img, caption=image.name)
        return images

    # Invoke ResNet
    def invoke_resnet(self, images):
        st.write("Invoking ResNet model and displaying predicted images...")
        url = "http://localhost:8080/predictions/resnet18"

        for image in images:
            image.seek(0)
            files = {"data": image}
            try:
                response = requests.post(url, files=files)
                if response.status_code == 200:
                    st.write(f"Prediction for {image.name}: {response.json()}")
                else:
                    st.write(
                        f"Failed to get prediction for {image.name}: {response.status_code}, {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to TorchServe: {e}")

    # Invoke BERT
    def invoke_bert(self, text):
        st.write("Invoking BERT model for text prediction...")
        url = "http://localhost:8080/predictions/bert"

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(text.encode("utf-8"))
            temp_file_path = temp_file.name

        try:
            with st.spinner("Processing BERT prediction..."):
                with open(temp_file_path, 'rb') as f:
                    files = {'data': f}
                    response = requests.post(url, files=files)
                    st.write(f"Given text is: {text}")
                    if response.status_code == 200:
                        st.write(f"Prediction for input: {response.json()}")
                    else:
                        st.write(
                            f"Failed to get prediction for input: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to TorchServe: {e}")
        finally:
            os.remove(temp_file_path)