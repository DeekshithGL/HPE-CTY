import glob
import streamlit as st

from hpe.modelinvoker import ModelInvoker
from hpe.torch_serve import TorchServeManager
from hpe.display_metrics import displayMetrics

# Main function
def main():
    st.title("ML Serving Framework")

    # Model selection dropdown in the sidebar
    serving_engine_choice = st.sidebar.selectbox(
        "Select ML Serving Engine", ["None", "TorchServe", "VLLM"]
    )

    # Model selection dropdown in the sidebar
    model_choice = st.sidebar.selectbox(
        "Select Model", ["None", "ResNet18", "BERT"])

    model_invoker = None
    if serving_engine_choice != "None":
        model_invoker = ModelInvoker(serving_engine_choice)

    torchServe = TorchServeManager()
    # Track if the model has changed
    if st.session_state.get("last_model_choice") != model_choice:
        st.session_state["last_model_choice"] = model_choice
        if model_choice != "None":
            torchServe.stop_torchserve()
            torchServe.launch_torchserve(model_choice)

    # Display input fields based on selected model
    if model_choice == "ResNet18":
        st.sidebar.header("ResNet Model")
        uploaded_images = st.sidebar.file_uploader(
            "Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if st.sidebar.button("Launch TorchServe"):
            if model_choice != "None":
                torchServe.launch_torchserve(model_choice)
        if st.sidebar.button("Stop TorchServe"):
            torchServe.stop_torchserve()

        if uploaded_images:
            st.header("ResNet Model")
            model_invoker.access_images(uploaded_images)
            model_invoker.invoke_resnet(uploaded_images)

    elif model_choice == "BERT":
        st.sidebar.header("BERT Model")
        uploaded_file = st.sidebar.file_uploader(
            "Upload a .txt file", type=["txt"])
        if st.sidebar.button("Launch TorchServe"):
            if model_choice != "None":
                torchServe.launch_torchserve(model_choice)
        if st.sidebar.button("Stop TorchServe"):
            torchServe.stop_torchserve()

        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")
            st.header("BERT Model")
            # Display the text from the uploaded file
            st.write(f"Text in the uploaded file: {text}")
            model_invoker.invoke_bert(text)

    display_metrics = displayMetrics()
    if st.sidebar.button("Show Model Metrics"):
        display_metrics.display_metrics()


if __name__ == "__main__":
    main()