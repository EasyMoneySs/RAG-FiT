import logging
from pathlib import Path
from typing import Dict

from transformers import AutoConfig, AutoTokenizer

from ragfit.utils import check_package_installed

logger = logging.getLogger(__name__)


class VLLMInference:
    """
    Initializes a vLLM-based inference engine.

    Args:
        model_name_or_path (str): The name or path of the model.
        instruction (Path): path to the instruction file.
        instruct_in_prompt (bool): whether to include the instruction in the prompt for models without system role.
        template (Path): path to a prompt template file if tokenizer does not include chat template. Optional.
        num_gpus (int, optional): The number of GPUs to use. Defaults to 1.
        llm_params (Dict, optional): Additional parameters for the LLM model. Supports all parameters define by vLLM LLM engine. Defaults to an empty dictionary.
        generation (Dict, optional): Additional parameters for text generation. Supports all the keywords of `SamplingParams` of vLLM. Defaults to an empty dictionary.
    """

    def __init__(
        self,
        model_name_or_path: str,
        instruction: Path,
        instruct_in_prompt: bool = False,
        template: Path = None,
        num_gpus: int = 1,
        llm_params: Dict = {},
        generation: Dict = {},
        lora_path: str = None,
        trust_remote_code: bool = False,
    ):
        check_package_installed(
            "vllm",
            "please refer to vLLM website for installation instructions, or run: pip install vllm",
        )
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.model_name = model_name_or_path
        self.instruct_in_prompt = instruct_in_prompt
        self.template = open(template).read() if template else None
        self.instruction = open(instruction).read()
        logger.info(f"Using the following instruction: {self.instruction}")

        self.sampling_params = SamplingParams(**generation)
        
        self.lora_path = lora_path
        self.lora_request = None
        if self.lora_path:
            logger.info(f"Enabling LoRA with adapter: {self.lora_path}")
            llm_params["enable_lora"] = True
            # Use a fixed ID (1) and name ("adapter") for single-adapter inference
            self.lora_request = LoRARequest("adapter", 1, self.lora_path)

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=num_gpus,
            trust_remote_code=trust_remote_code,
            **llm_params,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=trust_remote_code
        )
        if "baichuan" in self.model_name.lower() and not self.tokenizer.chat_template:
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}{{ '<reserved_106>' + message['content'] }}"
                "{% elif message['role'] == 'assistant' %}{{ '<reserved_107>' + message['content'] }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{ '<reserved_107>' }}{% endif %}"
            )

        self.config = AutoConfig.from_pretrained(
            self.model_name, trust_remote_code=trust_remote_code
        )

    def generate(self, prompts: str | list[str]) -> str | list[str]:
        """
        Generates text based on the given prompt or list of prompts.
        """
        is_batch = isinstance(prompts, list)
        prompt_list = prompts if is_batch else [prompts]
        
        all_inputs = []
        for p in prompt_list:
            prompt_token_ids = None
            prompt_str = None

            if self.template:
                prompt_str = self.template.format(instruction=self.instruction, query=p)
                all_inputs.append(prompt_str)
            else:
                if self.instruct_in_prompt or "baichuan" in self.model_name.lower():
                    full_prompt = self.instruction + "\n" + p
                    messages = [{"role": "user", "content": full_prompt}]
                else:
                    messages = [
                        {"role": "system", "content": self.instruction},
                        {"role": "user", "content": p},
                    ]

                prompt_token_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    truncation=True,
                    max_length=(
                        self.config.max_position_embeddings - self.sampling_params.max_tokens
                    ),
                )
                all_inputs.append({"prompt_token_ids": prompt_token_ids})

        outputs = self.llm.generate(
            prompts=all_inputs,
            sampling_params=self.sampling_params,
            lora_request=self.lora_request
        )
        
        results = [output.outputs[0].text for output in outputs]
        return results if is_batch else results[0]
