import torch
import os
import argparse

parser = argparse.ArgumentParser(description="Merge Autoencoder and Full Model Checkpoints")
parser.add_argument("--full_model_path", type=str, required=True, help="Path to the full model checkpoint")
parser.add_argument("--autoencoder_path", type=str, required=True, help="Path to the autoencoder checkpoint")
parser.add_argument("--save_path", type=str, required=True, help="Path to save the merged checkpoint")
args = parser.parse_args()

full_model_path = args.full_model_path
autoencoder_path = args.autoencoder_path
save_path = args.save_path

full_model = torch.load(full_model_path, map_location="cpu")
auto_model = torch.load(autoencoder_path, map_location="cpu")

full_sd = full_model["state_dict"]
auto_sd = auto_model["state_dict"]

encoder_weights = {
    f"first_stage_model.encoder.{k.replace('encoder.', '')}": v
    for k, v in auto_sd.items() if k.startswith("encoder.")
}

quant_conv_weights = {
    f"first_stage_model.quant_conv.{k.replace('quant_conv.', '')}": v
    for k, v in auto_sd.items() if k.startswith("quant_conv.")
}

post_quant_conv_weights = {
    f"first_stage_model.post_quant_conv.{k.replace('post_quant_conv.', '')}": v
    for k, v in auto_sd.items() if k.startswith("post_quant_conv.")
}

decoder_weights = {
    f"first_stage_model.decoder.{k.replace('decoder.', '')}": v
    for k, v in auto_sd.items() if k.startswith("decoder.")
}

print(f"Replacing: encoder {len(encoder_weights)}, quant_conv {len(quant_conv_weights)}, post_quant_conv {len(post_quant_conv_weights)}, decoder {len(decoder_weights)}")

full_sd.update(encoder_weights)
full_sd.update(quant_conv_weights)
full_sd.update(post_quant_conv_weights)
full_sd.update(decoder_weights)

# 保存
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(full_model, save_path)
print(f"Saved to {save_path}")