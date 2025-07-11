import asyncio
import base64
import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from chatterbox.tts import ChatterboxTTS

app = FastAPI()

# Precargar modelo
TTS_MODEL = ChatterboxTTS.from_pretrained(device="cuda")

@app.get("/")
async def get():
    with open("client.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()

            # ✅ Usa .generate()
            audio_tensor, sample_rate = TTS_MODEL.generate(text)

            # Convertir a NumPy
            audio_np = audio_tensor.detach().cpu().contiguous().numpy().squeeze()

            # Simular streaming: trocear audio
            chunk_size = int(sample_rate * 0.2)  # ~200ms por chunk
            for i in range(0, len(audio_np), chunk_size):
                chunk = audio_np[i:i+chunk_size]
                int_audio = (chunk * 32767).astype(np.int16).tobytes()
                b64_audio = base64.b64encode(int_audio).decode("utf-8")
                await websocket.send_json({"audio_content": b64_audio})
                await asyncio.sleep(0.05)  # simula real-time

            await websocket.send_json({"audio_end": True})
    except Exception as e:
        print("Error:", e)
