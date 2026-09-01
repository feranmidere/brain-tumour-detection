import torch
from torchvision.models import resnet50
import ipywidgets as widgets
from PIL import Image
import io
import scripts.train as train
from torchvision.transforms import v2
import solara

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load('model/model.pt', map_location=device))
model = model.to(device)
model = model.eval()
_, transform = train.set_transforms()
display_transform = v2.Resize((224, 224))
btn_upload = widgets.FileUpload()
prediction_label = widgets.Label()
image_output = widgets.Image(layout=widgets.Layout(
        width="224px",
        height="224px",
        border="1px solid #ccc",
    ))

blank_buffer = io.BytesIO()
Image.new('RGBA', (1, 1), (0, 0, 0, 0)).save(blank_buffer, format='PNG')
blank_bytes = blank_buffer.getvalue()
image_output.value = blank_bytes

btn_run = widgets.Button(description="Classify Scan")
def on_upload_finish(change):
    if btn_upload.value:
        img = Image.open(io.BytesIO(btn_upload.value[0]['content']))
        byte_arr = io.BytesIO()
        (display_transform(img)).save(byte_arr, format='PNG')
        image_output.value = byte_arr.getvalue()

def on_button_clicked(b):
    img = Image.open(io.BytesIO(btn_upload.value[0]['content']))
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = torch.sigmoid(model(img_tensor)).item()
    if output > 0.5:
        prediction_label.value = f"Prediction: Tumour Detected with {output*100:.2f}% confidence."
    else:
        prediction_label.value = f"Prediction: No Tumour Detected with { (1 - output) * 100:.2f}% confidence."
btn_upload.observe(on_upload_finish, names='value')
btn_run.on_click(on_button_clicked)
heading = widgets.HTML(value="<h2>Brain Tumour Classification</h2>")
page = widgets.VBox([heading, widgets.Label('Select a brain scan to classify'), btn_upload, image_output, btn_run, prediction_label])