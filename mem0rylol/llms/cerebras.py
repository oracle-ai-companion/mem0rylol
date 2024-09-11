from src.base.llms import BaseLLM

class CerebrasLLM(BaseLLM):
    def __init__(self, model_name: str, callback_manager: Optional[CallbackManager] = None):
        """
        Initialize the CerebrasLLM.

        @param model_name The name of the Cerebras model to use.
        @param callback_manager Optional CallbackManager to use for callbacks during LLM calls.
        """
        self.model_name = model_name
        self.callback_manager = callback_manager
        self.llm = ChatCerebras(model=model_name, callback_manager=callback_manager)

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Generate text using the Cerebras LLM.

        @param prompt The prompt to generate text from.
        @param stop Optional list of stop sequences to use when generating text.
        @return The generated text.
        """
        return self.llm([{"role": "user", "content": prompt}], stop=stop)

    def generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        """
        Generate text from multiple prompts using the Cerebras LLM.

        @param prompts List of prompts to generate text from.
        @param stop Optional list of stop sequences to use when generating text.
        @return The LLMResult containing the generated texts.
        """
        messages = [{"role": "user", "content": prompt} for prompt in prompts]
        return self.llm.generate(messages, stop=stop, **kwargs)
