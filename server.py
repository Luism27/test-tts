import asyncio
import base64
import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from chatterbox.tts import load_tts

app = FastAPI()

# Precargar modelo
TTS_MODEL = load_tts("resemble/your_model_id", device="cuda")

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
            audio_gen = TTS_MODEL.stream(text, sample_rate=24000)

            async for audio_chunk in audio_gen:
                if isinstance(audio_chunk, torch.Tensor):
                    audio_np = audio_chunk.detach().cpu().numpy().squeeze()
                else:
                    audio_np = audio_chunk

                # Convertir a bytes PCM int16
                int_audio = (audio_np * 32767).astype(np.int16).tobytes()
                b64_audio = base64.b64encode(int_audio).decode("utf-8")
                await websocket.send_json({"audio_content": b64_audio})

            await websocket.send_json({"audio_end": True})
    except Exception as e:
        print("Error:", e)
