from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

modelPath = "D:/ProgramData/torchHome/mymodel/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(modelPath)
model = BlipForConditionalGeneration.from_pretrained(modelPath)

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

testImg = './星穹铁道.jpg'
raw_image = Image.open(testImg).convert('RGB')

# conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt")

out = model.generate(**inputs)
print(testImg + "是一张：" + processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt")

out = model.generate(**inputs)
print(testImg + "是一张：" + processor.decode(out[0], skip_special_tokens=True))

# conditional image captioning
text = "The price of apples in Jun. is $"
inputs = processor(raw_image, text, return_tensors="pt")

out = model.generate(**inputs)
print(testImg + " 问题:" + text + "：\n" + processor.decode(out[0], skip_special_tokens=True))
