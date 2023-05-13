from model import TTS
from typing import List
from pathlib import Path
import soundfile as sf
from rich.console import Console
import yaml


class Demo:
    name: str
    query: List[str]
    audio: List[str]
    console: Console

    def __init__(self, name, query, console=Console(), root="./demo", **kwargs) -> None:
        self.name = name
        self.query = query
        self.console = console
        self.root = Path(root)

        self.audio = [
            f"query_{str(i + 1).zfill(2)}.wav" for i in range(len(self.query))
        ]
        if "audio" in kwargs:
            self.audio = kwargs["audio"]
            assert len(self.audio) == len(
                self.query
            ), "The number of audio names does not equal to the query."

        cfg = {
            "model-id": "microsoft/speecht5_tts",
            "vocoder": "microsoft/speecht5_hifigan",
            "voice_path": "xvectors.json",
        }
        self.tts = TTS(cfg, False, console)

    def create_dirtree(self):
        demo_dir = self.root / self.name
        demo_dir.mkdir(parents=True, exist_ok=True)
        cfg = demo_dir / "config.yaml"
        content = {"name": self.name, "query": self.query, "audio": self.audio}
        with cfg.open("w") as f:
            yaml.safe_dump(content, f)

    def create_speech(self):
        self.tts()
        for q, a in zip(self.query, self.audio):
            path = Path(self.root / self.name / a)
            speech = self.tts.inference(q)
            sf.write(str(path), speech, samplerate=16000)
            print(f"Audio file {a} is written.")

    def __call__(self):
        self.create_dirtree()
        self.create_speech()


if __name__ == "__main__":
    query = [
        "Tell me three sorting algorithms, just the names.",
        "Which one has the least time complexity? Just tell me the name.",
    ]
    d = Demo("intro_sorting", query)
    d()
