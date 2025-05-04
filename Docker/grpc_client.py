import gradio as gr
import grpc
import tts_pb2
import tts_pb2_grpc
import tempfile
import asyncio

async def tts(text, speaker):
    if not text.strip():
        return None

    try:
        async with grpc.aio.insecure_channel("localhost:50051") as channel:
            stub = tts_pb2_grpc.TextToSpeechStub(channel)
            req = tts_pb2.SynthesizeRequest(text=text, speaker=speaker)
            res = await stub.Generate(req)

            if not res.success:
                print("gRPC server responded with error:", res.message)
                return None

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(res.audio)
                return f.name
    except grpc.aio.AioRpcError as e:
        print(f"gRPC Error: {e.details()}")
        return None
    except Exception as e:
        print(f"General Error: {str(e)}")
        return None

with gr.Blocks(title="Story-to-Audio Generator") as demo:
    gr.Markdown("<h1 style='text-align: center; color: #FF4081;'>🎤 Story-to-Audio Generator</h1>", elem_id="title")
    with gr.Column(elem_id="text-block-bg"):
        gr.Markdown(
        "<p style='text-align: center; font-size: 15px; color: #6c757d;'>"
        "It's a story-to-audio project that uses a gRPC API to convert your story into humanized audio format. "
        "Each part of the system is handled with care, and we hope you enjoy your story."
        "</p>",
        elem_id="description"
        )


    with gr.Row():
        text_input = gr.Textbox(label="Enter Text", lines=3, placeholder="Type something...", elem_id="text_input")
        speaker_dropdown = gr.Dropdown(label="Speaker", choices=["en_0", "en_1", "en_2", "en_3", "en_4", "en_5"], value="en_0", elem_id="speaker_dropdown")

    with gr.Row():
        output_audio = gr.Audio(label="Generated Audio", elem_id="audio_output")

    generate_button = gr.Button("Generate Speech", elem_id="generate_button")

    generate_button.click(fn=tts, inputs=[text_input, speaker_dropdown], outputs=output_audio)

    gr.Markdown(
    "<p style='text-align: center; margin-top: 10px;'>"
    "<a href='https://github.com/sincera315/Pretrained_Story_to_Audio_Model_Integration_CPU-GPU' target='_blank' "
    "style='color: #FF4081; text-decoration: none; font-weight: bold;'>"
    "🔗 View on GitHub</a></p>"
)


demo.css = """
    body {
        background: url('static/back2.jpg') no-repeat center center fixed;
        background-size: cover;
        font-family: 'Arial', sans-serif;
    }

    #title {
        font-size: 48px;
        font-weight: 700;
        color: #FF4081;
        margin-bottom: 20px;
        font-family: 'Lucida Calligraphy', 'Brush Script MT', cursive, sans-serif
    }

    #text-block-bg {
    background-image: url("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAYGBgYH..."); /* full string here */
    background-size: cover;
    background-position: center;
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 20px;
}

    #description {
    font-size: 15px;
    color: #6c757d;
    margin-bottom: 30px;
    font-family: 'Lucida Calligraphy', cursive, sans-serif;
}

    #text_input, #speaker_dropdown {
        width: 85%;
        padding: 15px;
        margin: 10px auto;
        font-size: 16px;
        border-radius: 12px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out;
        background-color: #fff;
    }

    #text_input:focus, #speaker_dropdown:focus {
        border-color: #FF4081;
        box-shadow: 0 0 8px rgba(255, 64, 129, 0.6);
        outline: none;
    }

    #generate_button {
        background-color: #FF4081;
        color: white;
        padding: 18px 36px;
        border-radius: 50px;
        font-size: 20px;
        margin: 20px auto;
        cursor: pointer;
        border: none;
        transition: transform 0.2s ease, background-color 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    #generate_button:hover {
        background-color: #FF3366;
        transform: scale(1.1);
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }

    #generate_button:active {
        transform: scale(0.98);
    }

    #audio_output {
        width: 85%;
        margin-top: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .gradio-container {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
        padding: 30px;
        max-width: 600px;
        margin: 0 auto;
        transition: transform 0.3s ease;
    }

    .gradio-container:hover {
        transform: translateY(-5px);
    }

    @media (max-width: 768px) {
        #text_input, #speaker_dropdown, #generate_button {
            width: 100%;
        }

        #title {
            font-size: 36px;
        }
    }
"""


demo.launch()
