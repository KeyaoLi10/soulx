import os, sys
import re
import argparse
import uvicorn
from threading import Thread
from omegaconf import OmegaConf
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from clients.tts_client import IndexTTS_VLLM, Cosyvoice_Streaming_VLLM
from modules.utils.text_utils import split_cn_en


app = FastAPI()
# Global variables
tokenizer = None
model = None
index_tts = None
cosyvoice = None
gen_kwargs = {}


class StopOnDisconnect(StoppingCriteria):
    def __init__(self):
        self.should_stop = False

    def __call__(self, input_ids, scores, **kwargs):
        return self.should_stop


def load_model(model_path: str, config_path: str):
    global tokenizer, model, gen_kwargs, index_tts, cosyvoice
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True
    ).eval()

    gen_kwargs = {
        "max_new_tokens": cfg.max_tokens,
        "temperature": cfg.temp,
        "top_p": cfg.top_p,
        "do_sample": True,
    }

    print("Loading TTS client...")
    index_tts = IndexTTS_VLLM(speaker="ada")
    cosyvoice = Cosyvoice_Streaming_VLLM()

    print("LLM and TTS client loaded.")


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    messages = data.get("messages", [])

    if not messages:
        return StreamingResponse(iter([]), media_type="text/plain")

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    # Initialize stopping criteria
    stopper = StopOnDisconnect()
    stopping_criteria = StoppingCriteriaList([stopper])

    generation_kwargs = dict(
        inputs, streamer=streamer, stopping_criteria=stopping_criteria, **gen_kwargs
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    async def stream_generator():
        try:
            for new_text in streamer:
                if await request.is_disconnected():
                    break
                yield new_text
        except Exception:
            pass
        finally:
            # Signal the model to stop generating when client disconnects or stream ends
            stopper.should_stop = True

    return StreamingResponse(stream_generator(), media_type="text/plain")


@app.post("/chat_indextts")
async def chat_indextts(request: Request):
    data = await request.json()
    messages = data.get("messages", [])

    if not messages:
        return StreamingResponse(iter([]), media_type="text/plain")

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    def sample_next_token(logits, temperature=1.0, top_p=1.0, do_sample=True):
        if not do_sample:
            return torch.argmax(logits, dim=-1)
        if temperature and temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        if top_p and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > top_p
            mask_shifted = mask[..., :-1].clone()
            mask[..., 1:] = mask_shifted
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            return sorted_indices.gather(-1, sampled).squeeze(-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return sampled.squeeze(-1)

    def iterative_generate(inputs, gen_kwargs):
        max_new_tokens = gen_kwargs.get("max_new_tokens", 256)
        temperature = gen_kwargs.get("temperature", 1.0)
        top_p = gen_kwargs.get("top_p", 1.0)
        do_sample = gen_kwargs.get("do_sample", True)

        eos_token_id = model.generation_config.eos_token_id
        eos_token_ids = (
            {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
        )
        current_input_ids = inputs["input_ids"].clone()
        past_key_values = None
        buffer = ""
        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = model(
                        input_ids=current_input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
                    next_token = sample_next_token(
                        logits,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                    token_id = int(next_token.item())
                    if eos_token_ids and token_id in eos_token_ids:
                        break

                    current_input_ids = next_token.unsqueeze(-1)
                    current_text = tokenizer.decode(token_id, skip_special_tokens=True)
                    buffer += current_text
                    yield b"S" + current_text.encode("utf-8")

                    matches = list(re.finditer(r"[，。！？：；,.!?:;\n]", buffer))
                    if not matches:
                        continue
                    last = matches[-1]
                    end_idx = last.end()
                    if len(split_cn_en(buffer[:end_idx])) <= 3:
                        continue
                    seg = buffer[:end_idx]
                    buffer = buffer[end_idx:]
                    if seg.strip():
                        for wav_chunk in index_tts.synthesize(seg.strip()):
                            yield b"B" + wav_chunk

                if buffer.strip():
                    for wav_chunk in index_tts.synthesize(buffer.strip()):
                        yield b"B" + wav_chunk
        except Exception as e:
            print(f"Iterative generation failed: {e}")

    async def stream_generator():
        try:
            for item in iterative_generate(inputs, gen_kwargs):
                if await request.is_disconnected():
                    break
                yield item
        except Exception:
            pass

    return StreamingResponse(stream_generator())


@app.post("/chat_cosyvoice")
async def chat_cosyvoice(request: Request):
    data = await request.json()
    messages = data.get("messages", [])

    if not messages:
        return StreamingResponse(iter([]), media_type="text/plain")

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    def sample_next_token(logits, temperature=1.0, top_p=1.0, do_sample=True):
        if not do_sample:
            return torch.argmax(logits, dim=-1)
        if temperature and temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        if top_p and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > top_p
            mask_shifted = mask[..., :-1].clone()
            mask[..., 1:] = mask_shifted
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            return sorted_indices.gather(-1, sampled).squeeze(-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return sampled.squeeze(-1)

    def iterative_generate(inputs, gen_kwargs):
        max_new_tokens = gen_kwargs.get("max_new_tokens", 256)
        temperature = gen_kwargs.get("temperature", 1.0)
        top_p = gen_kwargs.get("top_p", 1.0)
        do_sample = gen_kwargs.get("do_sample", True)

        eos_token_id = model.generation_config.eos_token_id
        eos_token_ids = (
            {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
        )
        current_input_ids = inputs["input_ids"].clone()
        past_key_values = None
        buffer = ""
        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = model(
                        input_ids=current_input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
                    next_token = sample_next_token(
                        logits,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                    token_id = int(next_token.item())
                    if eos_token_ids and token_id in eos_token_ids:
                        break

                    current_input_ids = next_token.unsqueeze(-1)
                    current_text = tokenizer.decode(token_id, skip_special_tokens=True)
                    buffer += current_text
                    yield b"S" + current_text.encode("utf-8")

                    matches = list(re.finditer(r"[，。！？：；,.!?:;\n]", buffer))
                    if not matches:
                        continue
                    last = matches[-1]
                    end_idx = last.end()
                    if len(split_cn_en(buffer[:end_idx])) <= 3:
                        continue
                    seg = buffer[:end_idx]
                    buffer = buffer[end_idx:]
                    if seg.strip():
                        for wav_chunk in cosyvoice.synthesize(
                            seg.strip(), streaming=True
                        ):
                            yield b"B" + wav_chunk

                if buffer.strip():
                    for wav_chunk in cosyvoice.synthesize(
                        buffer.strip(), streaming=True
                    ):
                        yield b"B" + wav_chunk
        except Exception as e:
            print(f"Iterative generation failed: {e}")

    async def stream_generator():
        try:
            for item in iterative_generate(inputs, gen_kwargs):
                if await request.is_disconnected():
                    break
                yield item
        except Exception:
            pass

    return StreamingResponse(stream_generator())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6007)
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    )
    parser.add_argument("--model_dir", type=str, default=None)
    args = parser.parse_args()

    load_model(args.model_dir, args.config)
    uvicorn.run(app, host=args.host, port=args.port)
