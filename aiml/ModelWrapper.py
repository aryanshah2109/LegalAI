import torch

class RetrieverModelWrapper:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
            self,
            prompt: str,
            max_new_tokens: int = 128,
            do_sample: bool = False,
            temperature: float = 0.0,
            top_p: float = 1.0
        ) -> str:

        inputs = self.tokenizer(
            prompt,
            return_tensors = "pt"
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample = do_sample,
            temperature = temperature,
            top_p = top_p
        )

        text = self.tokenizer.decode(
            output[0],
            skip_special_tokens = True
        )

        return text[len(prompt):].strip()

    def __call__(self, prompt: str) -> str:
        
        from app.config import RETRIEVER_CONFIG

        self.generate(prompt, **RETRIEVER_CONFIG)


