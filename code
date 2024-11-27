import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
import math

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

wind_data = {
"""Provide the Wind Data of the particular Region
Example:
"U Wind Component": -4.96,
    "V Wind Component": 0.358,
    "Mean Sea Level Pressure": 101412.937,
    "Wave Direction": 76.18,
    "Sea Surface Temperature": 9.999e+20,
    "Wave Height": 0.417,
    "Total Precipitation": 0.00
"""
}


# Calculate wind speed
wind_speed = math.sqrt(wind_data["U Wind Component"]**2 + wind_data["V Wind Component"]**2)


region ={
"""provide regional data of the particular region
Example:
  "Label": "Mask 3",
  "Area (pixels)": 21981,
  "Area (km²)": 2.1981,
  "Perimeter": 1151.4975,
  "Perimeter to Area": 0.05239,
  "Complexity": 60.3224,
  "Shape Factor 1": 0.2083,
  "Shape Factor 2": 4.8003,
  "Mean Intensity": 38.7693,
  "Std Intensity": 48.4183,
  "Power to Mean Ratio": 46067.04,
  "Mean Contrast": 48.8040,
  "Max Contrast": 208.2307,
  "Local Contrast": 48.4183,
  "Mean Gradient": 0.1652,
  "Std Gradient": 0.1490,
  "Max Gradient": 0.8518,
  "Mean Haralick Texture": 3246.10
"""
}


prompt = f"""

You are an expert environmental analyst specializing in oil spill detection and classification. You are given regional and environmental data. Your task is to analyse the given parameters througlt andh classify the region as either an 'Oil Spill' or a 'Look-Alike' region. Explain your reasoning step-by-step in short. Search the internet and analyse the given data completely and give the confirm accurate answer whether the given region is classified as either an 'Oil Spill' or a 'Look-Alike' region.

### Regional Data ###
- Label: {region['Label']}
- Area: {region['Area (km²)']} km²
- Perimeter: {region['Perimeter']} m
- Perimeter to Area: {region['Perimeter to Area']}
- Complexity: {region['Complexity']}
- Shape Factor 1: {region['Shape Factor 1']}
- Shape Factor 2: {region['Shape Factor 2']}
- Mean Intensity: {region['Mean Intensity']}
- Std Intensity: {region['Std Intensity']}
- Power to Mean Ratio: {region['Power to Mean Ratio']}
- Mean Contrast: {region['Mean Contrast']}
- Max Contrast: {region['Max Contrast']}
- Local Contrast: {region['Local Contrast']}
- Mean Gradient: {region['Mean Gradient']}
- Std Gradient: {region['Std Gradient']}
- Max Gradient: {region['Max Gradient']}
- Mean Haralick Texture: {region['Mean Haralick Texture']}


### Environmental Data ###
- Average Wind Speed: {wind_speed:.2f} m/s
- Wave Direction: {wind_data['Wave Direction']} degrees
- Wave Height: {wind_data['Wave Height']} m
- Sea Surface Temperature: {wind_data['Sea Surface Temperature']} K
- Precipitation: {wind_data['Total Precipitation']} mm
- Mean Sea Level Pressure: {wind_data['Mean Sea Level Pressure']} Pa


### Reasoning ###
Please provide your accurate confirmed classification  and explain why the given region is an 'Oil Spill' or a 'Look-Alike' in short. Consider the physical shape characteristics of given regional data as well as the environmental data and provide a scientific explanation for your decision accuurately.
"""

inputs = processor(
    text=prompt,
    add_special_tokens=True,
    return_tensors="pt"
).to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.7
)

response = processor.decode(output[0], skip_special_tokens=True).strip()
print(f"Region {region['Label']} analysis:\n{response}")
