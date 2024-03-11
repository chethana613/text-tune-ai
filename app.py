import scipy,os,json
import torch, openai
from flask import Flask, render_template, request
from audiocraft.models import MusicGen

app = Flask(__name__)

def query_gpt(user_prompt, theme):
    openai.api_key = 'Enter API Key'
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a music expert, skilled in explaining intricacies in music vibe with contextual flair."},
                {"role": "user", "content": f"I am trying to get a highlevel description for {user_prompt} vibe of music for {theme} purpose. Can you please give me that one line music vibe explaining the rhythm."}
            ]
        )
        print(f"GPT Response: {response}")
        #print(response['choices'][0]['message']['content'])
        #return response['choices'][0]['message']['content']
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
        # Handle any exception that occurs during the OpenAI API request
        print(f"An error occurred during the OpenAI API request: {e}")
        return ""

def generate_music_tensors(prompt, model, duration,sr):
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
    scipy.io.wavfile.write("FILE_PATH/audio_output/text_tune.wav", rate=sr, data=audio.cpu().numpy().squeeze())
    return audio[:, :int(float(duration) * sr)]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_input', methods=['POST'])
def process_input():
    user_prompt = request.form['user_prompt']
    theme = request.form['theme']
    duration = request.form['duration']
    sr = 32000
    print("user+prompt: ", user_prompt)
    print("theme: ", theme)
    print("duration: ", duration)

    s = ""
    if user_prompt:
        s += user_prompt + ", "
    if (theme != ""):
        res = query_gpt(user_prompt,theme)
        if (res != ""):
            s += res

    print("Combined prompt:", s)
    if s and duration:
        music_tensors = generate_music_tensors(s, model,duration,sr)      
    else:
        print("Invalid prompt")            
    audio_file_path = "FILE_PATH/audio_output/text_tune.wav"
    if os.path.exists(audio_file_path):
        return render_template('index.html', audio_file=audio_file_path)
    else:
        return "File not found", 404

if __name__ == '__main__':
    model = MusicGen.get_pretrained('facebook/musicgen-stereo-small')
    app.run(debug=True)
