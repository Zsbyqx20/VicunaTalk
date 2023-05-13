import json
import time
from typing import Tuple

import numpy as np
import torch
from rich.console import Console
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from fastchat.model.model_adapter import get_conversation_template, load_model
from fastchat.serve.inference import generate_stream
from utils import process_stream


class ASR:
    def __init__(self, cfg: dict, cnt: bool, console=Console()) -> None:
        self.id = cfg["model-id"]
        self.cnt = cnt
        self.console = console

    def __call__(self) -> Tuple[Wav2Vec2Processor, Wav2Vec2ForCTC]:
        st = time.time()
        self.processor = Wav2Vec2Processor.from_pretrained(self.id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.id)
        if self.cnt:
            span = round(time.time() - st, 3)
            self.console.log(
                f"The [bold]ASR[/bold] initiate finished; Time Cost: [red]{span}[/red]s."
            )
        return self.processor, self.model

    def inference(self, array: np.ndarray):
        self.console.log("Start to convert speech into text.")
        inputs = self.processor(
            array, sampling_rate=16000, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = self.model(
                inputs.input_values, attention_mask=inputs.attention_mask
            ).logits
        ids = torch.argmax(logits, dim=-1).squeeze()
        sentence = self.processor.decode(ids)
        self.console.log(f"[red]{sentence}[red]")
        return sentence


class TTS:
    def __init__(self, cfg: dict, cnt: bool, console=Console()) -> None:
        self.id = cfg["model-id"]
        self.cnt = cnt
        self.vocoder_id = cfg["vocoder"]
        self.voice_path = cfg["voice_path"]
        self.console = console

    def __call__(
        self,
    ) -> Tuple[
        SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, torch.Tensor
    ]:
        st = time.time()
        self.processor = SpeechT5Processor.from_pretrained(self.id)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.id)
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.vocoder_id)

        with open(self.voice_path) as f:
            xvec = json.load(f)
        self.voice = torch.tensor(xvec[0]["xvector"]).unsqueeze(0)

        if self.cnt:
            span = round(time.time() - st, 3)
            self.console.log(
                f"The [bold]TTS[/bold] initiate finished; Time Cost: [red]{span}[/red]s."
            )
        return self.processor, self.model, self.vocoder, self.voice

    def inference(self, text):
        self.console.log("Start to convert text into speech.")
        inputs = self.processor(text=text, return_tensors="pt")
        speech = self.model.generate_speech(
            inputs["input_ids"], self.voice, vocoder=self.vocoder
        )
        return speech


class Vicuna:
    def __init__(self, cnt: bool, console=Console()) -> None:
        self.cnt = cnt
        self.console = console

    def __call__(self, cfg: dict):
        st = time.time()
        self.model, self.tokenizer = load_model(
            cfg["model-id"],
            cfg["device"],
            cfg["num_gpus"],
            cfg["max_gpu_memory"],
            cfg["load_8bit"],
            cfg["cpu_offloading"],
        )
        self.conv = get_conversation_template(cfg["model-id"])
        if self.cnt:
            span = round(time.time() - st, 3)
            self.console.log(
                f"The [bold]LLM[/bold] initiate finished; Time Cost: [red]{span}[/red]s."
            )
        self.cfg = cfg
        return self.model, self.tokenizer, self.conv

    def inference(self, query):
        self.conv.append_message(self.conv.roles[0], query)
        self.conv.append_message(self.conv.roles[1], None)

        prompt = self.conv.get_prompt()
        gen_params = {
            "model": self.cfg["model-id"],
            "prompt": prompt,
            "temperature": self.cfg["temperature"],
            "max_new_tokens": self.cfg["max_new_tokens"],
            "stop": self.conv.stop_str,
            "stop_token_ids": self.conv.stop_token_ids,
            "echo": False,
        }

        output_stream = generate_stream(
            self.model, self.tokenizer, gen_params, self.cfg["device"]
        )
        outputs = process_stream(output_stream)
        self.conv.messages[-1][-1] = outputs.strip()
        self.console.log(f"[green]{outputs}[/green]")
        return outputs
