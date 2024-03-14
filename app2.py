import IPython
from audiocraft.models import MusicGen
from IPython.display import display
from scipy.io import wavfile
import openai
import numpy as np
import librosa

def query_gpt(user_prompt, theme):
    openai.api_key = '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'
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a music expert, skilled in explaining intricacies in music vibe with contextual flair."},
                {"role": "user", "content": f"I am trying to get a highlevel description for {user_prompt} vibe of music for {theme} purpose. Can you please give me that one line music vibe explaining the rhythm."}
            ]
        )
        print(f"GPT Response: {response}")
        if response.choices:
            choice = response.choices[0]
            if choice.finish_reason == "stop":
                message = choice.message
                content = message.content
                print("Content:", content)
                return content
        else:
            print("No choices found in the response")
            return ""

    except Exception as e:
        print(f"An error occurred during the OpenAI API request: {e}")
        return ""

def generate_music_tensors(prompt, model, duration, sr):
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=int(duration)
    )
    print("Your custom tune is under generation....")
    output = model.generate(
        descriptions=[prompt],
        progress=True,
        return_tokens=True
    )

    audio = output[0]
    return audio[:, :int(float(duration) * sr)]

def compute_fad_score(generated_audio, target_audio, sr):
    gen_spec = np.abs(librosa.stft(generated_audio.squeeze().cpu().numpy()))
    target_spec = np.abs(librosa.stft(target_audio.squeeze()))
    fad_score = np.mean(np.abs(gen_spec - target_spec))
    return fad_score

def process_input(user_prompt, theme, model, duration, sr):
    print("User input:", user_prompt)
    print("Theme:", theme)
    print("Duration:", duration)
    s = ""
    if user_prompt:
        s += user_prompt + ", "
    if theme != "":
        res = query_gpt(user_prompt, theme)
        if res != "":
            s += res

    print("Combined prompt:", s)
    if s and duration:
        music_tensors = generate_music_tensors(s, model, duration, sr)
        return music_tensors, s
    else:
        print("Invalid prompt")
        return None, None

def main():
    model = MusicGen.get_pretrained('facebook/musicgen-small')

    user_prompt = input("Enter user input: ")
    theme = input("Enter your theme/tune mood: ")
    duration = input("Enter tune duration (in seconds): ")
    sampling_rate = 32000

    audio_data, theme_description = process_input(user_prompt, theme, model, duration, sampling_rate)

    if audio_data is not None and theme_description is not None:
        print("Audio file generated")

        IPython.display.display(IPython.display.Audio(audio_data.cpu().numpy().squeeze(), rate=sampling_rate))
        wavfile.write("/content/sample_data/text-tune-ai/output/download3.wav", rate=sampling_rate, data=audio_data[0, 0].cpu().numpy())
        # Assuming you have a target audio file for comparison
        target_audio, _ = librosa.load('/content/sample_data/text-tune-ai/output/download2.wav', sr=sampling_rate)

        # Calculate FAD Score
        fad_score = compute_fad_score(audio_data, target_audio, sampling_rate)
        print("FAD Score:", fad_score)

    else:
        print("There is something wrong with the audio generation")

if __name__ == "__main__":
    main()
