import gradio as gr
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Load Models (Dummy) ---
# In a real scenario, these files would be loaded here.
# For this UI design, we'll use dummy logic in the predict function.
# model = load_model('fake_instagram_model.keras')
# scaler = joblib.load('scaler.pkl')

# --- Prediction Function ---
def predict(profile_pic, nums_length_username, fullname_words, nums_length_fullname, name_username,
            description_length, external_url, private, posts, followers, follows):

    # Convert Yes/No to 1/0
    profile_pic = 1 if profile_pic == "Yes" else 0
    name_username = 1 if name_username == "Yes" else 0
    external_url = 1 if external_url == "Yes" else 0
    private = 1 if private == "Yes" else 0

    # Create the feature vector
    features = np.array([[
        profile_pic, nums_length_username, fullname_words, nums_length_fullname, name_username,
        description_length, external_url, private, posts, followers, follows
    ]])

    # --- Dummy Prediction Logic ---
    # In a real app, you would uncomment the model/scaler loading and use them here.
    # scaled_features = scaler.transform(features)
    # prediction_probs = model.predict(scaled_features)[0]

    # For this demo, we return a fixed dummy prediction
    prediction_probs = [0.2, 0.8] # [Genuine, Fake]

    prediction = "Fake" if np.argmax(prediction_probs) == 1 else "Genuine"
    confidences = {"Genuine": float(prediction_probs[0]), "Fake": float(prediction_probs[1])}

    return prediction, confidences

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="Instagram Fake Account Detector") as demo:
    # Header
    gr.Markdown("# Instagram Fake Account Detector")
    gr.Markdown("Detect whether an Instagram account is genuine or fake using machine learning.")

    with gr.Row():
        # Input Panel
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### Profile Features")
                profile_pic = gr.Radio(["Yes", "No"], label="Has Profile Picture?", value="Yes")
                nums_length_username = gr.Slider(0, 1, label="Username: Numbers-to-Length Ratio", value=0.3)
                fullname_words = gr.Slider(0, 12, step=1, label="Full Name: Number of Words", value=2)
                nums_length_fullname = gr.Slider(0, 1, label="Full Name: Numbers-to-Length Ratio", value=0.1)
                name_username = gr.Radio(["Yes", "No"], label="Is Full Name Same as Username?", value="No")
                description_length = gr.Slider(0, 200, step=1, label="Bio/Description Length", value=50)
                external_url = gr.Radio(["Yes", "No"], label="Has External URL?", value="No")
                private = gr.Radio(["Yes", "No"], label="Is Account Private?", value="No")
                posts = gr.Number(label="# of Posts", value=100)
                followers = gr.Number(label="# of Followers", value=500)
                follows = gr.Number(label="# of Following", value=300)

        # Output Panel
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Prediction Results")
                prediction_label = gr.Label(label="Prediction")
                confidences_plot = gr.BarPlot(
                    x=["Genuine", "Fake"],
                    y=[0.5, 0.5],
                    title="Prediction Probabilities",
                    y_lim=[0, 1],
                    width=250
                )

    # Submit Button
    check_button = gr.Button("Check Account", variant="primary")

    # Footer
    gr.Markdown("---")
    gr.Markdown("Created by Jules. Powered by Gradio.")

    # Button Click Action
    check_button.click(
        fn=predict,
        inputs=[
            profile_pic, nums_length_username, fullname_words, nums_length_fullname,
            name_username, description_length, external_url, private,
            posts, followers, follows
        ],
        outputs=[prediction_label, confidences_plot],
    )

demo.launch()
