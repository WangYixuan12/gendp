import base64
import time

from openai import OpenAI

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "/home/yixuan/bdai/general_dp/general_dp/tests/gpt_test_img.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

policy_names = ['open box', 'put objects into box', 'close box']
task_name = 'collect container into box'

sys_texts = "You are a robot equipped with following policy: \n"
for i in range(len(policy_names)):
    sys_texts += f"{i}. {policy_names[i]}\n"
sys_texts += f"The target task is {task_name}. Select the policy according to incoming observation. Explain why you select the policy. In the end of your response, output the final answer in the format of 'Final Answer: [policy index]'."

start_time = time.time()
response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
        "role": "system",
        "content": [
            {"type": "text", "text": sys_texts},
        ],
    },
    {
        "role": "user",
        "content": [
            # {"type": "text", "text": "What's in this image?"},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            },
        ],
    }
  ],
  max_tokens=300,
)
print(f'gpt-4-vision-preview: {time.time()-start_time}')

print(response.choices[0])
