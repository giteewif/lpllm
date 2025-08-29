import os
import argparse
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import re
import json

def natural_sort_key(s):
    """Helper function for natural sorting of strings with numbers"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def group_layer_weights(input_dir, output_dir):
    """Group weights from the same layer into a single safetensors file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取权重映射文件
    with open(input_dir + "/model.safetensors.index.json", "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]

    # 按层组织权重
    layer_weights = {}
    for weight_name, file_name in weight_map.items():
        # 提取层号
        if "model.layers." in weight_name:
            layer_idx = int(weight_name.split("model.layers.")[1].split(".")[0])
            if layer_idx not in layer_weights:
                layer_weights[layer_idx] = {}
            layer_weights[layer_idx][weight_name] = file_name
        else:
            # 其他权重(如embedding)保持原样
            if "other" not in layer_weights:
                layer_weights["other"] = {}
            layer_weights["other"][weight_name] = file_name

    # 为每一层创建新的safetensors文件
    for layer_idx, weights in layer_weights.items():
        layer_tensors = {}
        
        # 读取该层的所有权重
        for weight_name, file_name in weights.items():
            weight_file = safe_open(f"{input_dir}/{file_name}", framework="pt", device="cpu")
            layer_tensors[weight_name] = weight_file.get_tensor(weight_name)
        
        # 保存为新的safetensors文件
        if layer_idx == "other":
            output_file = f"{output_dir}/other.safetensors"
        else:
            output_file = f"{output_dir}/layer_{layer_idx}.safetensors"
        
        save_file(layer_tensors, output_file)
        print(f"Saved {len(layer_tensors)} weights to {output_file}")

    # 创建新的index文件
    new_weight_map = {}
    for layer_idx, weights in layer_weights.items():
        file_name = f"layer_{layer_idx}.safetensors" if layer_idx != "other" else "other.safetensors"
        for weight_name in weights.keys():
            new_weight_map[weight_name] = file_name
            
    new_index = {
        "metadata": index_data["metadata"],
        "weight_map": new_weight_map
    }
    
    with open(f"{output_dir}/model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

def copy_others_file(input_dir, output_dir):
    """Copy other configuration files like config.json to output directory"""
    other_files = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
    for file in other_files:
        input_path = os.path.join(input_dir, file)
        if os.path.exists(input_path):
            output_path = os.path.join(output_dir, file)
            with open(input_path, 'rb') as src, open(output_path, 'wb') as dst:
                dst.write(src.read())
            print(f"Copied {file} to output directory")

def main():
    parser = argparse.ArgumentParser(description='Combine layer weights into single safetensors files')
    parser.add_argument('--module_dir', type=str, required=True, 
                        help='Directory containing input weight files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save combined weight files')
    
    args = parser.parse_args()
    group_layer_weights(args.module_dir, args.output_dir)
    copy_others_file(args.module_dir, args.output_dir)
    print(f"finish copy files")
if __name__ == "__main__":
    main()