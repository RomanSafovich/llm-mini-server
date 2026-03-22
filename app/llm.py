from fastapi import HTTPException
import torch

def generate_text(prompt_str, model, tokenizer):
    try:
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
            new_tokens = output_tokens[0][input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))