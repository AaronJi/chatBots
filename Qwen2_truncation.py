from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.generation import GenerationConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model.eval()

system_prompt_general = "You are a helpful assistant."
system_prompt_human = "你的任务是像一个真人一样聊天对话，回答需在四句话以内，不需分段或列举，逻辑顺序不必过于明显。\n\
\n\
尊重并倾听对方的话题，避免过多谈论自己。\n\
请严格遵守以上规定进行聊天。"

#system_prompt = system_prompt_general
system_prompt = system_prompt_human

do_sample = True

messages = [{"role": "system", "content": system_prompt}]
while True:
    input_text = input('input(or type quit/clear to quit/clear):')
    if input_text.lower() == 'quit':
        break
    elif input_text.lower() == 'clear':
        print(f"previous message: {messages}")
        messages = [{"role": "system", "content": system_prompt}]
        continue

    messages.append({'role': 'user', 'content': input_text})
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    idx = prompt.find('<|im_start|>user')
    prompt_system = prompt[0:idx]
    prompt_else = prompt[idx:]

    # print('='*100)
    # print(prompt)
    # print('-'*100)
    # print(prompt_1)
    # print('-'*100)
    # print(prompt_2)
    # print('='*100)

    input_ids = tokenizer([prompt_system], [prompt_else], max_length=256, truncation='only_second', return_tensors='pt')['input_ids'].to(model.device)
    tokenizer.truncation_side='left'

    input_ids_list = input_ids.tolist()
    end_ids = tokenizer(['<|im_end|>'], max_length=256)['input_ids']
    user_ids = tokenizer(['<|im_start|>user'], max_length=256)['input_ids']
    def find_sublist(lst, target):
      m, n = len(lst), len(target)
      for i in range(m-n+1):
          if lst[i:i+n] == target: return i
      return -1
    idx_user = find_sublist(input_ids_list[0], user_ids[0])
    if idx_user != -1:
      idx_end = find_sublist(input_ids_list[0], end_ids[0])+len(end_ids[0])+1
      # print(idx_user, idx_end)
      input_ids = torch.cat((input_ids[:, :idx_end], input_ids[:, idx_user:]),dim=1)
      # print(input_ids)

    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    print(f'Q: {input_text}')
    
    # print('='*100)
    # print(input_text)
    # print('='*100)

    masks = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)

    outputs = model.generate(input_ids=input_ids,
                             attention_mask=masks,
                             max_new_tokens=256,
                             pad_token_id=tokenizer.eos_token_id,
                             do_sample=do_sample,
                             eos_token_id=[151645])

    output = outputs[0][input_ids.shape[1]:]
    # print('='*100)
    # print(output)
    # print('='*100)
    response = tokenizer.decode(output, skip_special_tokens=True)
    print('='*100)
    print(f'A: {response}')
    print('='*100)
    messages.append({'role': 'assistant', 'content': response})
print(f"history message: {messages}")