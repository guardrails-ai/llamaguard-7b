import os
import time

import modal

MODEL_ALIAS = "llamaguard7b"
MODELS_DIR = f"/{MODEL_ALIAS}"
VOLUME_NAME = f"{MODEL_ALIAS}"

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

app = modal.App(f"{MODEL_ALIAS}-non-optimized", image=image)


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
    def generate(self, chat):
        import torch
        from torch.nn.functional import softmax

        tokenizer = self.tokenizer
        model = self.model

        device = "cuda"

        print(f"Model: Loaded on device: {device}")
        print(f"Model: Chat {chat}")

        chat_template_display = tokenizer.apply_chat_template(chat, tokenize=False)

        print(f"Model: Chat Template: {chat_template_display}")
        
        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

        print(f"Model: Response: {response}")

        return response

    @modal.exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()

@app.function(
    keep_warm=1,
    allow_concurrent_inputs=10,
    timeout=60 * 10,
    secrets=[modal.Secret.from_dotenv()],
    volumes={MODELS_DIR: volume}
)
@modal.asgi_app(label="fa-hg-lg7b")
def tgi_app():
    import os

    import fastapi
    from fastapi.middleware.cors import CORSMiddleware

    from typing import List
    from pydantic import BaseModel

    TOKEN = os.getenv("TOKEN")
    if TOKEN is None:
        raise ValueError("Please set the TOKEN environment variable")
    
    volume.reload()  # ensure we have the latest version of the weights

    web_app = fastapi.FastAPI()

    http_bearer = fastapi.security.HTTPBearer(
        scheme_name="Bearer Token",
        description="See code for authentication details.",
    )
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def is_authenticated(api_key: str = fastapi.Security(http_bearer)):
        if api_key.credentials != TOKEN:
            raise fastapi.HTTPException(
                status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return {"username": "authenticated_user"}

    router = fastapi.APIRouter(dependencies=[fastapi.Depends(is_authenticated)])

    class ChatMessages(BaseModel):
        role: str
        content: str

    class ChatClassificationRequestBody(BaseModel):
        chat: List[ChatMessages]

  
    @web_app.post("/v1/chat/classification")
    async def chat_classification_response(body: ChatClassificationRequestBody):
        chat = body.model_dump().get("chat",[])

        print("Serving request for chat classification...")
        print(f"Chat: {chat}")
        response = Model().generate.remote(chat)

        is_unsafe = None
        subclass = None

        cleaned_response = response.lower().strip()

        if "unsafe" in cleaned_response:
            is_unsafe = True
            split_cln_response = response.strip().split(os.linesep)
            subclass = split_cln_response[1] if len(split_cln_response) > 1 else None
        else:
            is_unsafe = False

        return {
            "class": "unsafe" if is_unsafe else "safe",
            "subclass": subclass,
            "response": response
        }


    return web_app


# @app.local_entrypoint()
# def main():
#     model = Model()
#     model.generate.remote()