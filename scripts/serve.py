from fastapi import FastAPI, UploadFile, File,applications
from PIL import Image
import io
from mycv.predictor import Predictor
from fastapi.openapi.docs import get_swagger_ui_html


def swagger_monkey_patch(*args, **kwargs):
    return get_swagger_ui_html(
        *args, **kwargs,
        swagger_js_url="https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.1.0/swagger-ui-bundle.min.js",
        swagger_css_url="https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.1.0/swagger-ui.min.css"
    )


applications.get_swagger_ui_html = swagger_monkey_patch

# 下面再写你的 app = FastAPI(...)
app = FastAPI()
predictor = Predictor("ckpt/best.pth", data="mnist")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    label, prob = predictor.predict(img)
    return {"label": label, "confidence": prob}
