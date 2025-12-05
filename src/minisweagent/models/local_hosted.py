import os
from typing import Any
from minisweagent.models.litellm_model import LitellmModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import logging
import litellm
from minisweagent.config import builtin_config_dir
import csv

logger = logging.getLogger("local_model")

class LocallyHostedModelConfig:
    easy_name: str
    model_name: str
    model_kwargs: dict[str, Any]
    ip: str

    def __init__(self, easy_name: str, model_name: str, port: int, ip: str, model_kwargs: dict[str, Any] = {}):
        self.easy_name = easy_name
        self.model_name = model_name
        defaults = {
            "custom_llm_provider": "openai",
            "api_base": f"http://{ip}:{port}/v1",
        }
        self.model_kwargs = defaults | model_kwargs

    @classmethod
    def load_model_mappings(cls):
        mappings = {}

        if local_config_model_mappings := cls.get_local_config_model_mappings():
            mappings.update(local_config_model_mappings)

        return mappings

    @classmethod
    def get_local_config_model_mappings(cls):
        # read from csv in default config that has 4 fields: `easy_name,ip,port,vllm_model_name`
        file_path = builtin_config_dir / "locally_hosted_models.csv"
        if not os.path.exists(file_path):
            return {}
        mappings = {}
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                model_kwargs = {}
                if len(row) > 4:
                    model_kwargs = {k: v for k, v in [arg.split('=') for arg in row[4:] if "=" in arg]}
                    # del key, value if key is "comment"
                    if "comment" in model_kwargs:
                        del model_kwargs["comment"]
                print(model_kwargs)
                easy_name, ip, port, vllm_model_name = row[:4]
                mappings[easy_name] = LocallyHostedModelConfig(
                    easy_name=easy_name,
                    model_name=vllm_model_name,
                    port=int(port),
                    ip=ip,
                    model_kwargs=model_kwargs
                )
        return mappings


KNOWN_LOCAL_MODELS = LocallyHostedModelConfig.load_model_mappings()
    
class LocallyHostedModel(LitellmModel):
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        local_config = KNOWN_LOCAL_MODELS[self.config.model_name]

        try:
            return litellm.completion(
                model=local_config.model_name, messages=messages, **(local_config.model_kwargs | kwargs)
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e
