from pathlib import Path
import yaml
from typing import Tuple, List
from rich import print
from rich.console import Console


def process_stream(output_stream):
    pre = 0
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            # yield " ".join(output_text[pre:now])
            pre = now
    # yield " ".join(output_text[pre:])
    return " ".join(output_text)


class DemoLoader:
    demoRoot: Path
    id: str
    name: str
    query: List[str]
    audio: List[str]
    console: Console

    def __init__(self, **kwargs) -> None:
        self.demoRoot = Path("./demo")
        self.id = "01"
        self.console = Console()

        if "demoRoot" in kwargs:
            self.demoRoot = Path(kwargs["demoRoot"])
        if "id" in kwargs:
            self.id = kwargs["id"]
        if "console" in kwargs:
            self.console = kwargs["console"]

    def checker(self) -> bool:
        assert self.demoRoot.exists(), "The root directory of demo is not valid!"
        assert (self.demoRoot / self.id).exists(), "The number of demo is not valid!"

        cfgName = "config.yaml"
        cfg = self.demoRoot / self.id / cfgName
        assert cfg.exists(), "No configuration file detected in the demo folder."

        cfg_file = cfg.open("r")
        cfg_dict = yaml.safe_load(cfg_file)

        assert set(["name", "query", "audio"]) <= set(
            cfg_dict.keys()
        ), "Invalid configuration keys!"

        self.name = cfg_dict["name"]
        self.query = cfg_dict["query"]
        self.audio = cfg_dict["audio"]

        assert list(
            map(lambda x: (cfg / x).exists(), self.audio)
        ), "The audio file specified in config.yaml is not valid!"

        self.console.log(
            ":heavy_check_mark:",
            f"Check passed. The demoLoader [bold]{self.name}[/bold] is ready for usage.",
        )
        return True

    def __call__(self) -> Tuple[str, List[str], List[str]]:
        assert self.checker()
        return self.name, self.query, self.audio


class ConfigLoader:
    config: Path
    console: Console
    asr: dict
    tts: dict
    llm: dict

    def __init__(self, **kwargs) -> None:
        self.config = Path("config.yaml")
        self.console = Console()

        if "config" in kwargs:
            self.config = kwargs["config"]
        if "console" in kwargs:
            self.console = kwargs["console"]

    def checker(self) -> bool:
        assert self.config.exists(), "The configuration file does not exist."
        cfg_file = self.config.open("r")
        cfg_dict = yaml.safe_load(cfg_file)

        for key in ["speech-to-text", "text-to-speech", "language-model"]:
            assert key in cfg_dict, f"No config written for {key} part."

        self.asr: dict = cfg_dict["speech-to-text"]
        self.tts: dict = cfg_dict["text-to-speech"]
        self.llm: dict = cfg_dict["language-model"]

        assert (
            "model-id" in self.asr
        ), "No model id specified in the speech-to-text part."
        assert set(["model-id", "vocoder", "voice_path"]) <= set(
            self.tts.keys()
        ), "Configuration in text-to-speech part is not complete."
        assert set(
            [
                "model-id",
                "device",
                "num_gpus",
                "max_gpu_memory",
                "load_8bit",
                "cpu_offloading",
                "temperature",
                "max_new_tokens",
            ]
        ) <= set(
            self.llm.keys()
        ), "The configuration for language model is not complete."

        assert Path(
            self.tts["voice_path"]
        ).exists(), "The voice path specified is not valid!"
        assert Path(
            self.llm["model-id"]
        ).exists(), "The language model path is not valid!"

        self.console.log(
            ":heavy_check_mark:",
            f"Check passed. The ConfigLoader is ready for usage.",
        )
        return True

    def __call__(self) -> Tuple[dict, dict, dict]:
        assert self.checker()
        return self.asr, self.tts, self.llm


if __name__ == "__main__":
    # dl = DemoLoader()
    # name, query, audio = dl()

    # print(name, query, audio)
    cfg = ConfigLoader()
    asr, tts, llm = cfg()

    print(asr, tts, llm)
