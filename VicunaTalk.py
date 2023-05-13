from pathlib import Path

import librosa
import soundfile as sf
from rich.console import Console
from rich.prompt import Prompt

from model import ASR, TTS, Vicuna
from utils import ConfigLoader, DemoLoader


def main():
    console = Console()
    console.rule("[bold]Program Initialization")

    # load configurations
    console.log("Load configuration...")
    cfgLoader = ConfigLoader(console=console)
    asr_cfg, tts_cfg, llm_cfg = cfgLoader()

    # load models
    asr = ASR(asr_cfg, cnt=True, console=console)
    tts = TTS(tts_cfg, cnt=True, console=console)
    vicuna = Vicuna(cnt=True, console=console)

    asr()
    tts()
    vicuna(llm_cfg)

    # talking engine starts
    console.rule("[bold violet]Talking Engine")
    cnt = 1
    audio_list = []

    while True:
        if not audio_list:
            choice = Prompt.ask(
                "Select one way to chat (F for specifying an audio file, D for loading a demo, Q for quiting)",
                choices=["F", "D", "Q"],
            )
            if choice == "Q":
                break
            elif choice == "D":
                if not Path("demo").exists():
                    console.print(
                        "Directory demo is not detected; Please modify your directory and retry"
                    )
                    continue
                id = Prompt.ask("Please input the name of demo you want to load")
                dl = DemoLoader(demoRoot="demo", id=id)

                _, _, demo_audios = dl()
                audio_list = list(
                    map(lambda x: str(Path(f"demo/{id}/{x}")), demo_audios)
                )
            else:
                path = Prompt.ask("Please input the audio path")
                audio_list.append(path)
        path = audio_list.pop(0)

        if Path(path).exists() is False:
            console.print("[red bold]The path specified does not exist!")
            continue
        if Path(path).suffix not in [".wav"]:
            console.print("[yellow bold]The file specified is not an audio file!")
            continue

        # convert audio file into text
        query_array = librosa.load(path, sr=16000)[0]
        query = asr.inference(query_array)

        # push the query into language model
        answer = vicuna.inference(query)

        # convert the answer into audio again
        speech = tts.inference(answer)

        sf.write(f"answer_{cnt}.wav", speech, samplerate=16000)
        console.log(f"answer_{cnt}.wav has been saved!")
        cnt += 1

    console.rule("[bold violet]End Talking")
    console.log(
        ":smiley: Thank you for using [bold blue]vicuna talk[/bold blue]! See you!"
    )
    return


if __name__ == "__main__":
    main()
