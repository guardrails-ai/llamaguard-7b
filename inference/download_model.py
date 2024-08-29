import modal


MODELS_DIR = "/llamaguard7b"
VOLUME_NAME = "llamaguard7b"

MODEL_NAME = "meta-llama/LlamaGuard-7b"
DEFAULT_REVISION = "dfcfa3409b9994a4722d44e05f82e81ea73c5106"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "huggingface_hub",  # download models from the Hugging Face Hub
            "hf-transfer",  # download models faster with Rust
            "transformers",
            "jinja2"
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


MINUTES = 60
HOURS = 60 * MINUTES


app = modal.App(
    image=image, secrets=[modal.Secret.from_name("huggingface-secret")]
)

@app.function(
    volumes={MODELS_DIR: volume}, 
    timeout=4 * HOURS,
)
def download_model(model_name, model_revision, force_download=False):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    import os

    volume.reload()

    model_path = MODELS_DIR + "/" + model_name

    if not os.path.exists(model_path):
        print(f"Model {model_name} does not exist in {model_path}, downloading...")
        snapshot_download(
            model_name,
            local_dir=model_path,
            ignore_patterns=[
                "*.pt",
                "*.bin",
                "*.pth",
                "original/*",
            ],  # Ensure safetensors
            revision=model_revision,
            force_download=force_download,
        )
    else:
        print(f"Model {model_name} already exists in {model_path}, skipping download...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        chat = [
            {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
        ]
        guideline = "GR_GUIDELINES"
        chat_template = tokenizer.apply_chat_template(chat, guideline=guideline, tokenize=False)

        print("Chat template:")
        print(chat_template)
        with open(model_path + "/chat_template.txt", "w") as f:
            f.write(chat_template)

    volume.commit()


@app.local_entrypoint()
def main(
    model_name: str = MODEL_NAME,
    model_revision: str = DEFAULT_REVISION,
    force_download: bool = False,
):
    download_model.remote(model_name, model_revision, force_download)