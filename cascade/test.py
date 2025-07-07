from vllm import LLM, SamplingParams

#model_path = "/mnt/c/joev/models/_hf/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/e9f8effbab1cbdc515c11ee6e098e3d5a9f51e14"
model_path = "/home/alvion/models/llama-3.2-1b-instruct"
llm = LLM(model=model_path, max_model_len=4096)


prompt = """<|start_header_id|>user<|end_header_id|>

What is the tallest mountain on Earth?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


sampling_params = SamplingParams(temperature=0)
outputs = llm.generate(prompt, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
