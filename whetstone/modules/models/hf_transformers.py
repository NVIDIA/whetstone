import logging
import torch
import torch.nn.functional as F
from torch import Tensor
from whetstone.core.model import Model
from whetstone.core.module import ModuleRegistry
from transformers import AutoTokenizer, AutoModelForCausalLM

log = logging.getLogger(__name__)

@ModuleRegistry.register(make_dataclass=False)
class HFTransformersModel(Model):
    """
    A wrapper around a Hugging Face Transformers model.

    Implements the Model interface using a pre-trained model and tokenizer
    from the `transformers` library.
    """

    def __init__(self, model_name: str, device: str | None = None, generation_kwargs: dict | None = None):
        """Initializes the HFTransformersModel.

        Args:
            model_name: The name of the pre-trained Hugging Face model (e.g., 'gpt2').
            device: The device to load the model onto ('cpu', 'cuda', etc.). Auto-detects if None.
            generation_kwargs: Default keyword arguments for the `generate` method.
        """
        self.model_name = model_name
        
        # Determine device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.generation_kwargs = generation_kwargs or {}
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True).to(self.device)

    def generate(self, x: str, max_tokens: int = 100, **kwargs) -> str:
        """Generates text completion for a given prompt using the loaded HF model.
        
        Args:
            x: The input prompt string.
            max_tokens: The maximum number of new tokens to generate.
            **kwargs: Additional keyword arguments passed directly to the underlying `model.generate` method.

        Returns:
            The generated text string, excluding the prompt.
        """
        inputs = self.tokenizer(x, return_tensors="pt").to(self.device)

        # default temperature is 0.0
        if "temperature" not in kwargs:
            kwargs["do_sample"] = False
        
        # Combine default generation kwargs with method-specific ones
        gen_kwargs = {**self.generation_kwargs, **kwargs, "max_new_tokens": max_tokens}
        
        # Generate token IDs
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode the generated tokens, skipping special tokens and the prompt
        # We need to slice the output_ids to remove the input prompt tokens
        input_length = inputs.input_ids.shape[1]
        generated_tokens = output_ids[:, input_length:]
        
        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    def tokenize(self, x: str, **kwargs) -> Tensor:
        """Tokenizes a string using the model's tokenizer.

        Args:
            x: The input string to tokenize.
            **kwargs: Additional arguments passed to the tokenizer (e.g., `add_special_tokens`).
                      `add_special_tokens` defaults to False.

        Returns:
            A 1D Tensor containing the token IDs.
        """
        if "add_special_tokens" not in kwargs:
            kwargs["add_special_tokens"] = False
        return self.tokenizer(x, return_tensors="pt", **kwargs).input_ids.to(self.device)[0]

    def detokenize(self, x: Tensor, **kwargs) -> str:
        """Detokenizes a tensor of token IDs using the model's tokenizer.

        Handles both single sequences (1D Tensor) and batches (2D Tensor).

        Args:
            x: A Tensor of token IDs (1D or 2D).
            **kwargs: Additional arguments passed to the tokenizer's decode/batch_decode methods.

        Returns:
            The detokenized string or list of strings.
        """
        if x.dim() > 1:
            return self.tokenizer.batch_decode(x, **kwargs)
        else:
            return self.tokenizer.decode(x, **kwargs)

    def vocab_size(self) -> int:
        """Returns the size of the tokenizer's vocabulary."""
        return self.tokenizer.vocab_size
    
    def render_conversation_template(self, conversation: list[dict[str, str]]) -> str:
        """Renders a conversation into a single string using the tokenizer's chat template."""
        try:
            # apply_chat_template requires add_generation_prompt=True for generation
            return self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            log.warning(f"Model {self.model_name} tokenizer does not appear to support chat templates. Falling back to basic formatting. Error: {e}")
            return '\n'.join([f"{msg['role']}: \n{msg['content']}\n" for msg in conversation])