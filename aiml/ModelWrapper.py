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
        '''
        __call__ is a dunder method that will allow to call an object with paranthesis
        It looks like we are calling a function with some parameters
        For example:
        Suppose a class MyClass has a __call__ method which takes 2 parameters and adds and returns the sum

        obj = MyClass()
        obj(4,5)

        This allows a clean look.

        This will be used in Retriever part where the LLM will automatically call the __call__ method when using the LLM

        :param self: Description
        :param prompt: Description
        :type prompt: str
        :return: Description
        :rtype: str
        '''
        
        from app.model_config import RETRIEVER_CONFIG

        return self.generate(prompt, **RETRIEVER_CONFIG)


class RAGModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


    def generate(
            self,
            prompt: str,
            max_new_tokens: int = 512,
            do_sample: bool = False,
            temperature: float = 0.0,
            top_p: float = 1.0,
            repetition_penalty: float = 1.15
        ) -> str:

        input = self.tokenizer(
            prompt,
            return_tensors = "pt"
        ).to(self.model.device)

        output = self.model.generate(
            **input,
            max_new_tokens = max_new_tokens,
            do_sample = do_sample,
            temperature = temperature,
            top_p = top_p,
            repetition_penalty = repetition_penalty
        )

        text = self.tokenizer.decode(
            output[0],
            skip_special_tokens = True
        )

        return text[len(prompt):].strip()
    
    def __call__(self, prompt: str) -> str:

        from app.model_config import RAG_CONFIG

        return self.generate(prompt, **RAG_CONFIG)

