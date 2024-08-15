import os
import time

import modal

MODELS_DIR = "/llamaguard7b"
VOLUME_NAME = "llamaguard7b"

MODEL_NAME = "meta-llama/LlamaGuard-7b"
DEFAULT_REVISION = "dfcfa3409b9994a4722d44e05f82e81ea73c5106"


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm",
        "torch",
        "transformers",
        "ray",
        "huggingface_hub",
        "hf-transfer",
        "accelerate"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(f"vllm-{MODEL_NAME}", image=image)

with image.imports():
    import vllm

try:
    volume = modal.Volume.lookup(VOLUME_NAME, create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_model.py")


GPU_CONFIG = modal.gpu.A10G(count=1)


@app.cls(gpu=GPU_CONFIG, secrets=[modal.Secret.from_name("huggingface-secret")], volumes={MODELS_DIR: volume},)
class Model:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_directory = MODELS_DIR + "/" + MODEL_NAME
        self.model = AutoModelForCausalLM.from_pretrained(
            model_directory,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)

    @modal.method()
    def generate(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        from torch.nn.functional import softmax

        tokenizer = self.tokenizer
        model = self.model

        device = "cuda"
        

        def moderate(chat):
            print(tokenizer.apply_chat_template(chat, tokenize=False))
            input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
            output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
            prompt_len = input_ids.shape[-1]
            return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

        res = moderate([
            {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
            {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
        ])

        print(res)

    @modal.exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()


@app.local_entrypoint()
def main():
    model = Model()
    model.generate.remote()