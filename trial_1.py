import os
import torch
import torchaudio
from IPython.display import Audio

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Speech-to-Text example
def test_speech_to_text():
    # Use the proper way to load the STT model directly from torch hub
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_stt',
                                  language='en', # also available 'de', 'es', 'fr'
                                  device=device)
    
    (read_batch, split_into_batches, read_audio, prepare_model_input) = utils
    
    # Download test file
    test_file = 'test.wav'
    if not os.path.isfile(test_file):
        torch.hub.download_url_to_file('https://github.com/snakers4/silero-models/raw/master/examples/en_example.wav', 
                                       test_file)
    
    # Process the audio
    audio = read_audio(test_file)
    input = prepare_model_input(audio, device=device)
    
    # Perform transcription
    output = model(input)
    
    # Get the text
    transcription = output[0].detach().cpu().numpy().tolist()
    print("STT Result:", transcription)
    return transcription

# Text-to-Speech example
def test_text_to_speech(text=None):
    # Use the proper way to load the TTS model
    model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                         model='silero_tts',
                                         language='en',
                                         speaker='v3_en')
    
    if text is None:
        text = example_text
    
    # Get the sample rate
    sample_rate = 48000
    
    # Generate audio
    audio = model.apply_tts(text=text,
                            speaker='en_0',
                            sample_rate=sample_rate)
    
    # Save audio to file
    output_file = "tts_output.wav"
    torchaudio.save(output_file, audio.unsqueeze(0), sample_rate)
    
    print(f"TTS output saved to {output_file}")
    return output_file

if __name__ == "__main__":
    print("Running Speech-to-Text test...")
    try:
        transcription = test_speech_to_text()
        
        print("\nRunning Text-to-Speech test...")
        output_file = test_text_to_speech("This is a test of the Silero text to speech system.")
        
        print("\nAll tests completed. You can listen to the TTS output at:", output_file)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("\nAlternative approach: Try using the Silero API directly...")
        
        # Try the direct approach from Silero's documentation
        print("\nAttempting direct download method...")
        torch.hub.download_url_to_file('https://models.silero.ai/models/tts/en/v3_en.pt',
                                      'silero_tts_en.pt')
        print("Model downloaded successfully as silero_tts_en.pt")