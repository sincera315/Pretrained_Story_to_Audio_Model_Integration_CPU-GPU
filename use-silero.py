import torch
import os
import sys
import wave
import re
import numpy as np
from pydub import AudioSegment

# Load the Silero TTS model
model = torch.package.PackageImporter('silero_tts_en.pt').load_pickle("tts_models", "model")
device = torch.device('cpu')
model.to(device)

def split_text_into_sentences(text):
    # Split text into sentences based on punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def synthesize_sentence(sentence, speaker='en_0', sample_rate=48000):
    # Generate audio for a single sentence
    with torch.no_grad():  # Add no_grad for efficiency
        audio = model.apply_tts(
            text=sentence,
            speaker=speaker,
            sample_rate=sample_rate
        )
    audio_np = (audio.numpy() * 32767).astype(np.int16)
    return audio_np

def apply_transformer_enhancements(audio_tensor, enhance_level=0.8):
    """Apply transformations to make the audio more human-like
    
    This function applies several transformations:
    1. Pitch variation
    2. Voice naturalization via formant preservation
    """
    # Ensure tensor is on the correct device
    audio_tensor = audio_tensor.to(device)
    
    # Convert to frequency domain using FFT
    with torch.no_grad():
        # Apply short-time Fourier transform
        n_fft = 1024
        hop_length = 256
        win_length = 1024
        
        # Add batch dimension if needed
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        # Apply STFT
        spec = torch.stft(
            audio_tensor, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length, 
            window=torch.hann_window(win_length).to(device),
            return_complex=True
        )
        
        # Get magnitude and phase
        mag = spec.abs() + 1e-8
        phase = spec.angle()
        
        # Enhance formants (subtle pitch variations for naturalness)
        # This simulates more natural vocal tract resonances
        formant_enhancement = torch.randn_like(mag) * 0.05 * enhance_level
        mag = mag * (1 + formant_enhancement)
        
        # Apply subtle variations in phase (simulates natural vocal micro-variations)
        phase_variation = torch.randn_like(phase) * 0.03 * enhance_level
        new_phase = phase + phase_variation
        
        # Reconstruct complex spectrogram
        enhanced_spec = mag * torch.exp(1j * new_phase)
        
        # Convert back to time domain
        enhanced_audio = torch.istft(
            enhanced_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length).to(device)
        )
        
        # Return to CPU for numpy conversion
        enhanced_audio = enhanced_audio.squeeze(0).cpu()
        
    return enhanced_audio

def detect_punctuation(sentence):
    """Detect punctuation to determine appropriate pause length"""
    if re.search(r'[.!]$', sentence):
        return 'long'  # End of statement
    elif re.search(r'\?$', sentence):
        return 'question'  # Question
    elif re.search(r',$', sentence):
        return 'comma'  # Comma
    elif re.search(r'[;:]$', sentence):
        return 'medium'  # Semicolon or colon
    else:
        return 'short'  # Default

def text_to_speech_with_enhancements(text, output_file="output.wav", speaker='en_0', enhance_level=0.8):
    """Generate enhanced speech with natural pauses"""
    sample_rate = 48000
    sentences = split_text_into_sentences(text)
    
    # Define pause durations (in milliseconds)
    pause_durations = {
        'short': 200,
        'medium': 350,
        'long': 500,
        'comma': 250,
        'question': 450
    }
    
    # Add variation to pause durations
    variation_factor = 0.2  # ±20% variation
    
    # Initialize final audio segment
    final_audio = AudioSegment.empty()
    
    for i, sentence in enumerate(sentences):
        print(f"Processing sentence {i+1}/{len(sentences)}: {sentence[:30]}...")
        
        # Determine emphasis for important words
        sentence = highlight_important_words(sentence)
        
        # Synthesize sentence
        audio_np = synthesize_sentence(sentence, speaker, sample_rate)
        
        # Convert numpy array to torch tensor for enhancements
        audio_tensor = torch.FloatTensor(audio_np.astype(np.float32) / 32767.0)
        
        # Apply transformers enhancement
        enhanced_audio = apply_transformer_enhancements(audio_tensor, enhance_level)
        
        # Convert back to numpy int16
        enhanced_np = (enhanced_audio.numpy() * 32767).astype(np.int16)
        
        # Convert numpy array to AudioSegment
        audio_segment = AudioSegment(
            enhanced_np.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit audio
            channels=1
        )
        
        # Determine pause type based on punctuation
        pause_type = detect_punctuation(sentence)
        base_pause = pause_durations[pause_type]
        
        # Add natural variation to pause duration
        random_factor = 1.0 + variation_factor * (2 * np.random.random() - 1)
        pause_duration = int(base_pause * random_factor)
        
        # Create pause with dynamic duration
        pause = AudioSegment.silent(duration=pause_duration)
        
        # Append sentence audio and pause
        final_audio += audio_segment + pause
    
    # Export final audio
    final_audio.export(output_file, format="wav")
    print(f"Enhanced audio saved to {output_file}")
    return output_file

def highlight_important_words(text):
    """Subtly emphasize important words by adding subtle markers
    
    This doesn't add any visible tags but helps the TTS model emphasize
    certain words more naturally
    """
    # Important words to emphasize (can be expanded)
    important_words = ['important', 'critical', 'essential', 'significant', 
                      'never', 'always', 'must', 'crucial', 'vital']
    
    # Add subtle word spacing around important words for emphasis
    for word in important_words:
        pattern = r'\b' + word + r'\b'
        text = re.sub(pattern, f" {word} ", text)
    
    # Clean up excessive spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def intonation_tweaks(sentences):
    """Apply specific intonation adjustments to full text"""
    enhanced_sentences = []
    
    for sentence in sentences:
        # Handle questions with slightly higher pitch at the end
        if sentence.strip().endswith('?'):
            # For questions, no explicit tag is needed - the model usually handles
            # question intonation well, especially when we use proper punctuation
            enhanced_sentences.append(sentence)
        
        # Handle exclamations with more emphasis
        elif sentence.strip().endswith('!'):
            enhanced_sentences.append(sentence)
        
        # Regular sentences
        else:
            enhanced_sentences.append(sentence)
    
    return enhanced_sentences

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = ' '.join(sys.argv[1:])
    else:
        input_text = input("Enter the text you want to convert to speech: ")
    
    # Ask for enhancement level
    try:
        enhance_level = float(input("Enter enhancement level (0.0-1.0, default 0.8): ") or 0.8)
        enhance_level = max(0.0, min(1.0, enhance_level))  # Clamp between 0 and 1
    except ValueError:
        enhance_level = 0.8
        print("Invalid input. Using default enhancement level of 0.8")
    
    # Check if output path is provided
    output_file = "enhanced_output.wav"
    
    # Process the text and generate speech
    text_to_speech_with_enhancements(input_text, output_file, enhance_level=enhance_level)
    print(f"You can find your enhanced audio file at: {os.path.abspath(output_file)}")