import os
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

# Define the dataset directory
dataset_dir = "dataset/batch1"  # Change this to the parent directory of different batches

# Extract batch name dynamically
batch_name = os.path.basename(dataset_dir)

# Get all image files in the directory (supports jpg, jpeg, png)
image_extensions = (".jpg", ".jpeg", ".png")
image_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith(image_extensions)]

if not image_files:
    print("No images found in the directory.")
    exit()

# Load DeepSeek-VL2 model and processor
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

extracted_texts = []

# Process each image
for image_path in image_files:
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\n<|ref|>Please extract all the information from the given image<|/ref|>.",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # Load and prepare images for input
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    # Run image encoder to get embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Generate text response
    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    extracted_text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    extracted_texts.append(f"\n{extracted_text}\n")

# Save extracted text to a file named after the batch
output_file = f"{batch_name}.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.writelines(extracted_texts)

print(f"Extracted information saved to {output_file}")
