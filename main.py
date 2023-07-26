import openai
import asyncio
import whisper
import boto3
import pydub
from pydub import playback
import speech_recognition as sr

# Initialize the OpenAI API
# this is your private key from openAI
openai.api_key = "[API_KEY]"

# Create a recognizer object and wake word variables
recognizer = sr.Recognizer()
WAKEUP_PHRASE = "chat"

def get_wake_word(phrase):
    print("identifying wake phrase...")
    if WAKEUP_PHRASE in phrase.lower():
        return WAKEUP_PHRASE
    else:
        return None
    
# convert text to audio using aws polly
def synthesize_speech(text, output_filename):
    polly = boto3.client('polly', region_name='us-west-2')
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Salli',
        Engine='neural'
    )

    with open(output_filename, 'wb') as file:
        file.write(response['AudioStream'].read())

def play_audio(file):
    sound = pydub.AudioSegment.from_file(file, format="mp3")
    playback.play(sound)


async def main():
    while True:

        # You might have many input devices, check /proc/asound/card* and use the corresponding index
        with sr.Microphone(device_index=0) as source:
            recognizer.adjust_for_ambient_noise(source)
            print(f"Awaiting for wakeup phrase...")
            while True:

                audio = recognizer.listen(source)
                try:
                    
                    with open("audio.wav", "wb") as file:
                        file.write(audio.get_wav_data())
                        
                    # Use the preloaded tiny_model
                    model = whisper.load_model("tiny")

                    # your current CPU doesn't support fp16
                    result = model.transcribe("audio.wav", fp16=False)
                    phrase = result["text"]
                    print(f"You said: {phrase}")

                    if WAKEUP_PHRASE in phrase.lower():
                        break
                    else:
                        print("Invalid wakeup phrase. Try again.")
                except Exception as e:
                    print("Error transcribing audio: {0}".format(e))
                    continue

            print("Speak a prompt...")
            synthesize_speech('What can I help you with?', 'response.mp3')
            play_audio('response.mp3')
            audio = recognizer.listen(source)

            try:
                with open("audio_prompt.wav", "wb") as file:
                    file.write(audio.get_wav_data())
                model = whisper.load_model("base")
                result = model.transcribe("audio_prompt.wav", fp16=False)
                user_input = result["text"]
                print(f"You said: {user_input}")
            except Exception as e:
                print("Error transcribing audio: {0}".format(e))
                continue

            
            # Send prompt to GPT-3.5-turbo API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content":
                    "You are a helpful assistant."},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.5,
                max_tokens=200,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=1,
                stop=["\nUser:"],
            )

            bot_response = response["choices"][0]["message"]["content"]
                
        print("Bot's response:", bot_response)
        synthesize_speech(bot_response, 'response.mp3')
        play_audio('response.mp3')
        # await bot.close()

if __name__ == "__main__":
    asyncio.run(main())