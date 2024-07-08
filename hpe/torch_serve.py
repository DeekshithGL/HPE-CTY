import os
import time
import requests
import streamlit as st

MODEL_STORE = "model_store"
# Define hardcoded parameters
MODEL_STORE = "model_store"
MODEL_FILES = {
    "resnet18": "/HPE_CTY/model_store/resnet18.mar",
    "bert": "/HPE_CTY/model_store/bert.mar"
}

class TorchServeManager:
    def __init__(self) -> None:
        self.CURRENT_MODEL = None

    def is_torchserve_running(self):
        try:
            ts_addr = "http://localhost:8080/ping"
            response = requests.get(ts_addr)
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.exceptions.ConnectionError:
            return False

    # Launch torchserve with the selected model
    def launch_torchserve(self, model_name):
        if self.is_torchserve_running():
            if self.CURRENT_MODEL == model_name:
                st.success("TorchServe already running with the selected model")
                return
            else:
                self.stop_torchserve()

        model_file = MODEL_FILES.get(model_name.lower())
        if not model_file:
            st.error(f"Model file for {model_name} not found")
            return

        try:
            with st.spinner(f"Starting TorchServe with {model_name}..."):
                os.system(
                    f"torchserve --start --ncs --model-store {MODEL_STORE} --models {model_name.lower()}={model_file}")
                self.CURRENT_MODEL = model_name
                time.sleep(5)
        except Exception as e:
            st.error(f"Failed to start TorchServe: {e}")

        if self.is_torchserve_running():
            st.success(f"TorchServe started successfully with {model_name}")
        else:
            st.error(f"TorchServe failed to start with {model_name}")

    # Stop torchserve
    def stop_torchserve(self):
        if not self.is_torchserve_running():
            st.success("TorchServe is not running")
            return

        try:
            with st.spinner("Stopping TorchServe..."):
                os.system("torchserve --stop")
                self.CURRENT_MODEL = None
                time.sleep(5)
        except Exception as e:
            st.error(f"Failed to stop TorchServe: {e}")

        if not self.is_torchserve_running():
            st.success("TorchServe stopped successfully")
        else:
            st.error("TorchServe failed to stop")