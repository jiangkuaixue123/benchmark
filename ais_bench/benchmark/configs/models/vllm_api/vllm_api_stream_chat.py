from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-stream-chat",
        path="/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite",
        model="/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite",
        # stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="127.0.0.1",
        host_port=8022,
        url="",
        # max_out_len=512,
        batch_size=256,
        trust_remote_code=True,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=True,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
