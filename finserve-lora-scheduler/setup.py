from setuptools import setup, find_packages

setup(
    name="finserve-lora-scheduler",
    version="0.1.0",
    description=(
        "LoRA-Aware Chunked Prefill Scheduler Plugin for vLLM — "
        "reduces adapter switching overhead in Multi-LoRA serving"
    ),
    author="FinServe-MLO",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "vllm>=0.8.0",
    ],
    entry_points={
        "vllm.general_plugins": [
            "finserve_lora_scheduler = finserve_lora_scheduler:register",
        ],
    },
)
