import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import time
from utils.model import ECG_XNOR_Full_Bin, ECG_XNOR_Full_akx_plus_b
from utils.OP import WeightOperation
from utils.dataset import Loader
from utils.engine import train, test_step
from torchinfo import summary
import torch.nn.functional as F
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from pathlib import Path
from tqdm import tqdm        # optional progress-bar

quant_desc = QuantDescriptor(num_bits=16, fake_quant=False, axis=1, unsigned=False)
quantizer = TensorQuantizer(quant_desc)

def txt_to_coe(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    with open(output_file, 'w') as outfile:
        outfile.write("memory_initialization_radix=2;\n")
        outfile.write("memory_initialization_vector=\n")

        # Flatten lines into a single string, removing newlines
        data = ''.join(line.strip() for line in lines)

        # Split the data into chunks of 32 bits (or another size if needed)
        chunk_size = 64
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        # Write each chunk followed by a comma, except for the last chunk
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:
                outfile.write(f"{chunk},\n")
            else:
                outfile.write(f"{chunk};\n")


def concatenate_thre_files(input_files, output_file):
    with open(output_file, 'w') as outfile:
        for input_file in input_files:
            with open(input_file, 'r') as infile:
                outfile.write(infile.read())
                # outfile.write('\n')  # Optional: Add a newline between files


def merge_threshold_files(base_folder_path):
    # Define the path to the threshold files
    threhold_path = os.path.abspath(os.path.join(os.getcwd(), base_folder_path, "threhold/txt"))

    # List of files to concatenate
    input_files = [
        os.path.join(threhold_path, f"block{i}_thre.txt") for i in range(1, 6)
    ]

    # Output file path
    output_file = os.path.join(threhold_path, "block_thre_merge.txt")

    # Concatenate the files
    concatenate_thre_files(input_files, output_file)

    return output_file


from pathlib import Path

def concatenate_files(input_file1: str, input_file2: str, output_file: str):
    with open(output_file, 'w') as out, \
         open(input_file1, 'r') as f1, \
         open(input_file2, 'r') as f2:

        last_line = None
        for last_line in f1:            # copy file 1
            out.write(last_line)

        # If file 1 wasn't empty and its last line lacked '\n', add one
        if last_line is not None and not last_line.endswith('\n'):
            out.write('\n')

        out.writelines(f2)              # copy file 2


def save_ifmap_tensor_to_coe(tensor, folder_path, file_name_pattern):
    coe_folder = os.path.join(folder_path, "coe")
    if not os.path.exists(coe_folder):
        os.makedirs(coe_folder)

    num_channels = tensor.size(1)
    values_per_channel = tensor.size(2)
    bits_per_word = 32

    for channel in range(num_channels):
        channel_values = tensor[0, channel, :].cpu().numpy()
        binary_values = (channel_values > 0).astype(int)  # Convert +1 to 1 and -1 to 0

        coe_content = "memory_initialization_radix=2;\nmemory_initialization_vector=\n"
        word = ''
        for i, value in enumerate(binary_values):
            word = str(value) + word  # Append new bit to the left side of the word
            if (i + 1) % bits_per_word == 0:
                coe_content += word + ",\n"
                word = ''

        if word:
            word = word.zfill(bits_per_word)  # Pad the remaining bits to form a 32-bit word
            coe_content += word + ",\n"

        coe_content = coe_content.rstrip(",\n") + ";"

        file_name = file_name_pattern.replace("~channel", f"{channel}")
        file_path = os.path.join(coe_folder, file_name)

        with open(file_path, 'w') as f:
            f.write(coe_content)


def save_ifmap_tensor_to_txt(tensor, folder_path, file_name_pattern):
    txt_folder = os.path.join(folder_path, "txt")
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    num_channels = tensor.size(1)
    values_per_channel = tensor.size(2)
    bits_per_word = 32

    for channel in range(num_channels):
        channel_values = tensor[0, channel, :].cpu().numpy()
        binary_values = (channel_values > 0).astype(int)  # Convert +1 to 1 and -1 to 0

        txt_content = ""
        word = ''
        for i, value in enumerate(binary_values):
            word = str(value) + word  # Append new bit to the left side of the word
            if (i + 1) % bits_per_word == 0:
                txt_content += word + "\n"
                word = ''

        if word:
            word = word.zfill(bits_per_word)  # Pad the remaining bits to form a 32-bit word
            txt_content += word + "\n"

        file_name = file_name_pattern.replace("~channel", f"{channel}")
        file_path = os.path.join(txt_folder, file_name)

        with open(file_path, 'w') as f:
            f.write(txt_content)


def generate_int8_ecg_coe_file(tensor, file_name="ecgdata_s_int8.coe"):
    """
    Generates a COE file for initializing a 64-bit BRAM in Xilinx from a given tensor with binary format.

    Args:
    tensor (torch.Tensor): The input tensor containing 8-bit signed numbers as floats.
    file_name (str): The name of the output COE file.
    """
    tensor = tensor.squeeze().cpu().numpy().astype(int)

    # Ensure values are within 8-bit signed integer range (-128 to 127)
    tensor = tensor.clip(-128, 127)

    # Convert 8-bit signed integers to 8-bit binary strings
    bin_data = ['{:08b}'.format((byte + 256) % 256) for byte in tensor]

    # Calculate padding needed to make the data a multiple of 8 bytes
    padding_needed = (8 - len(bin_data) % 8) % 8
    bin_data.extend(['00000000'] * padding_needed)

    # Group data into 64-bit chunks (8 bytes per chunk)
    chunk_size = 8
    chunks = [bin_data[i:i + chunk_size] for i in range(0, len(bin_data), chunk_size)]

    # Reverse each chunk
    reversed_chunks = [chunk[::-1] for chunk in chunks]

    # Convert reversed chunks to binary format
    bin_chunks = []
    for chunk in reversed_chunks:
        bin_str = ''.join(chunk)
        bin_chunks.append(bin_str)

    # Write to COE file
    with open(file_name, 'w') as f:
        f.write("memory_initialization_radix=2;\n")
        f.write("memory_initialization_vector=\n")
        f.write(",\n".join(bin_chunks))
        f.write(";\n")


def generate_int8_ecg_txt_file(tensor, file_name="ecgdata_s_int8.txt"):
    """
    Generates a TXT file containing 64-bit chunks of binary data from a given tensor with 8-bit signed integers.

    Args:
    tensor (torch.Tensor): The input tensor containing 8-bit signed numbers as floats.
    file_name (str): The name of the output TXT file.
    """
    tensor = tensor.squeeze().cpu().numpy().astype(int)

    # Ensure values are within 8-bit signed integer range (-128 to 127)
    tensor = tensor.clip(-128, 127)

    # Convert 8-bit signed integers to 8-bit binary strings
    bin_data = ['{:08b}'.format((byte + 256) % 256) for byte in tensor]

    # Calculate padding needed to make the data a multiple of 8 bytes
    padding_needed = (8 - len(bin_data) % 8) % 8
    bin_data.extend(['00000000'] * padding_needed)

    # Group data into 64-bit chunks (8 bytes per chunk)
    chunk_size = 8
    chunks = [bin_data[i:i + chunk_size] for i in range(0, len(bin_data), chunk_size)]

    # Reverse each chunk
    reversed_chunks = [chunk[::-1] for chunk in chunks]

    # Convert reversed chunks to binary format
    bin_chunks = []
    for chunk in reversed_chunks:
        bin_str = ''.join(chunk)
        bin_chunks.append(bin_str)

    # Write to TXT file
    with open(file_name, 'w') as f:
        f.write("\n".join(bin_chunks))


def extract_and_save_ifmap(model, input_data, block_name, base_folder_path):
    block_folder_path = os.path.join(base_folder_path, block_name)
    model.eval()
    with torch.no_grad():
        x = input_data.to(model.device)
        x = x.clone().detach().requires_grad_(True)
        x = model.block1.pad(x)
        x = model.block1.conv(x)
        save_ofmap_to_txt(x, os.path.join(block_folder_path, "block1_conv_out.txt"))
        x = model.block1.pool(x)
        x = model.block1.prelu(x)
        x = model.block1.bn(x)
        x = model.block2.pad(x)


def extract_and_save_ofmap(model, input_data, block_name, stage, base_folder_path):
    block_folder_path = os.path.join(base_folder_path)
    model.eval()
    with torch.no_grad():
        x = input_data.to(model.device)
        x = x.clone().detach().requires_grad_(True)

        blocks = [model.block1, model.block2, model.block3, model.block4, model.block5, model.block6]
        block_names = ["block1", "block2", "block3", "block4", "block5", "block6"]

        for i, block in enumerate(blocks):
            current_block_folder = os.path.join(block_folder_path, block_names[i])
            if not os.path.exists(current_block_folder):
                os.makedirs(current_block_folder)

            x = block.pad(x)
            if block_name == block_names[i]:
                if block_name == "block3":
                    # Calculate psum for M0C0, M0C1, M1C0, M1C1
                    conv_weights = block.conv.weight.data
                    ifmaps = x.clone()  # Use the input data for conv

                    psum_results = {}
                    for m in range(2):  # M0, M1
                        for c in range(2):  # C0, C1
                            if m == 0 and c == 0:
                                weight_slice = conv_weights[0:16, 0:8, :]
                                ifmap_slice = ifmaps[0, 0:8, :]
                            elif m == 0 and c == 1:
                                weight_slice = conv_weights[0:16, 8:16, :]
                                ifmap_slice = ifmaps[0, 8:16, :]
                            elif m == 1 and c == 0:
                                weight_slice = conv_weights[16:32, 0:8, :]
                                ifmap_slice = ifmaps[0, 0:8, :]
                            elif m == 1 and c == 1:
                                weight_slice = conv_weights[16:32, 8:16, :]
                                ifmap_slice = ifmaps[0, 8:16, :]

                            # psum = torch.nn.functional.conv1d(ifmap_slice.unsqueeze(0), weight_slice.unsqueeze(0), stride=1)
                            psum = torch.nn.functional.conv1d(ifmap_slice.sign().unsqueeze(0), weight_slice.sign(), bias=None, stride=1, padding=0)
                            psum_results[f'M{m}C{c}'] = psum

                    # Save each psum to separate text files
                    psum_folder = os.path.join(current_block_folder, "psum")
                    for key, value in psum_results.items():
                        save_ofmap_to_txt(value.squeeze(), os.path.join(psum_folder, f"{block_name}_{key}_psum.txt"))
                if block_name == "block4":
                    # Calculate psum for M0C0, M0C1, M0C2, M0C3, M1C0, M1C1, M1C2, M1C3
                    conv_weights = block.conv.weight.data
                    ifmaps = x.clone()  # Use the input data for conv

                    psum_results = {}
                    for m in range(2):  # M0, M1
                        for c in range(4):  # C0, C1
                            if m == 0 and c == 0:
                                weight_slice = conv_weights[0:16, 0:8, :]
                                ifmap_slice = ifmaps[0, 0:8, :]
                            elif m == 0 and c == 1:
                                weight_slice = conv_weights[0:16, 8:16, :]
                                ifmap_slice = ifmaps[0, 8:16, :]
                            elif m == 0 and c == 2:
                                weight_slice = conv_weights[0:16, 16:24, :]
                                ifmap_slice = ifmaps[0, 16:24, :]
                            elif m == 0 and c == 3:
                                weight_slice = conv_weights[0:16, 24:32, :]
                                ifmap_slice = ifmaps[0, 24:32, :]
                            elif m == 1 and c == 0:
                                weight_slice = conv_weights[16:32, 0:8, :]
                                ifmap_slice = ifmaps[0, 0:8, :]
                            elif m == 1 and c == 1:
                                weight_slice = conv_weights[16:32, 8:16, :]
                                ifmap_slice = ifmaps[0, 8:16, :]
                            elif m == 1 and c == 2:
                                weight_slice = conv_weights[16:32, 16:24, :]
                                ifmap_slice = ifmaps[0, 16:24, :]
                            elif m == 1 and c == 3:
                                weight_slice = conv_weights[16:32, 24:32, :]
                                ifmap_slice = ifmaps[0, 24:32, :]
                            # psum = torch.nn.functional.conv1d(ifmap_slice.unsqueeze(0), weight_slice.unsqueeze(0), stride=1)
                            psum = torch.nn.functional.conv1d(ifmap_slice.sign().unsqueeze(0), weight_slice.sign(), bias=None, stride=1, padding=0)
                            psum_results[f'M{m}C{c}'] = psum
                if block_name == "block5":
                    # Calculate psum for M0C0, M0C1, M0C2, M0C3, M1C0, M1C1, M1C2, M1C3, M2C0, M2C1, M2C2, M2C3, M3C0, M3C1, M3C2, M3C3
                    conv_weights = block.conv.weight.data
                    ifmaps = x.clone()  # Use the input data for conv

                    psum_results = {}
                    for m in range(4):  # M0, M1, M2, M3
                        for c in range(4):  # C0, C1, C2, C3
                            if m == 0 and c == 0:
                                weight_slice = conv_weights[0:16, 0:8, :]
                                ifmap_slice = ifmaps[0, 0:8, :]
                            elif m == 0 and c == 1:
                                weight_slice = conv_weights[0:16, 8:16, :]
                                ifmap_slice = ifmaps[0, 8:16, :]
                            elif m == 0 and c == 2:
                                weight_slice = conv_weights[0:16, 16:24, :]
                                ifmap_slice = ifmaps[0, 16:24, :]
                            elif m == 0 and c == 3:
                                weight_slice = conv_weights[0:16, 24:32, :]
                                ifmap_slice = ifmaps[0, 24:32, :]
                            elif m == 1 and c == 0:
                                weight_slice = conv_weights[16:32, 0:8, :]
                                ifmap_slice = ifmaps[0, 0:8, :]
                            elif m == 1 and c == 1:
                                weight_slice = conv_weights[16:32, 8:16, :]
                                ifmap_slice = ifmaps[0, 8:16, :]
                            elif m == 1 and c == 2:
                                weight_slice = conv_weights[16:32, 16:24, :]
                                ifmap_slice = ifmaps[0, 16:24, :]
                            elif m == 1 and c == 3:
                                weight_slice = conv_weights[16:32, 24:32, :]
                                ifmap_slice = ifmaps[0, 24:32, :]
                            elif m == 2 and c == 0:
                                weight_slice = conv_weights[32:48, 0:8, :]
                                ifmap_slice = ifmaps[0, 0:8, :]
                            elif m == 2 and c == 1:
                                weight_slice = conv_weights[32:48, 8:16, :]
                                ifmap_slice = ifmaps[0, 8:16, :]
                            elif m == 2 and c == 2:
                                weight_slice = conv_weights[32:48, 16:24, :]
                                ifmap_slice = ifmaps[0, 16:24, :]
                            elif m == 2 and c == 3:
                                weight_slice = conv_weights[32:48, 24:32, :]
                                ifmap_slice = ifmaps[0, 24:32, :]
                            elif m == 3 and c == 0:
                                weight_slice = conv_weights[48:64, 0:8, :]
                                ifmap_slice = ifmaps[0, 0:8, :]
                            elif m == 3 and c == 1:
                                weight_slice = conv_weights[48:64, 8:16, :]
                                ifmap_slice = ifmaps[0, 8:16, :]
                            elif m == 3 and c == 2:
                                weight_slice = conv_weights[48:64, 16:24, :]
                                ifmap_slice = ifmaps[0, 16:24, :]
                            elif m == 3 and c == 3:
                                weight_slice = conv_weights[48:64, 24:32, :]
                                ifmap_slice = ifmaps[0, 24:32, :]

                            # Perform 1D convolution for each slice
                            psum = torch.nn.functional.conv1d(ifmap_slice.sign().unsqueeze(0), weight_slice.sign(),
                                                              bias=None, stride=1, padding=0)
                            psum_results[f'M{m}C{c}'] = psum
                if block_name == "block6":
                    conv_weights = block.conv.weight.data
                    ifmaps = x.clone()  # Use the input data for conv

                    psum_results = {}
                    for c in range(8):  # C0, C1, C2, C3……C7
                        if c == 0:
                            weight_slice = conv_weights[0:5, 0:8, :]
                            ifmap_slice = ifmaps[0, 0:8, :]
                        elif c == 1:
                            weight_slice = conv_weights[0:5, 8:16, :]
                            ifmap_slice = ifmaps[0, 8:16, :]
                        elif c == 2:
                            weight_slice = conv_weights[0:5, 16:24, :]
                            ifmap_slice = ifmaps[0, 16:24, :]
                        elif c == 3:
                            weight_slice = conv_weights[0:5, 24:32, :]
                            ifmap_slice = ifmaps[0, 24:32, :]
                        elif c == 4:
                            weight_slice = conv_weights[0:5, 32:40, :]
                            ifmap_slice = ifmaps[0, 32:40, :]
                        elif c == 5:
                            weight_slice = conv_weights[0:5, 40:48, :]
                            ifmap_slice = ifmaps[0, 40:48, :]
                        elif c == 6:
                            weight_slice = conv_weights[0:5, 48:56, :]
                            ifmap_slice = ifmaps[0, 48:56, :]
                        elif c == 7:
                            weight_slice = conv_weights[0:5, 56:64, :]
                            ifmap_slice = ifmaps[0, 56:64, :]
                            # Perform 1D convolution for each slice
                        psum = torch.nn.functional.conv1d(ifmap_slice.sign().unsqueeze(0), weight_slice.sign(),
                                                          bias=None, stride=1, padding=0)
                        psum_results[f'C{c}'] = psum
                    # Save each psum to separate text files
                    psum_folder = os.path.join(current_block_folder, "psum")
                    for key, value in psum_results.items():
                        save_ofmap_to_txt(value.squeeze(), os.path.join(psum_folder, f"{block_name}_{key}_psum.txt"))
            x_conv = block.conv(x)
            if stage == "conv" or stage == "all":
                if block_name == "all" or block_name == block_names[i]:
                    conv_folder = os.path.join(current_block_folder, "conv")
                    if not os.path.exists(conv_folder):
                        os.makedirs(conv_folder)
                    save_ofmap_to_txt(x_conv, os.path.join(conv_folder, f"{block_names[i]}_conv_out.txt"))
            x = x_conv

            x_pool = block.pool(x)
            if stage == "pool" or stage == "all":
                if block_name == "all" or block_name == block_names[i]:
                    pool_folder = os.path.join(current_block_folder, "pool")
                    if not os.path.exists(pool_folder):
                        os.makedirs(pool_folder)
                    save_ofmap_to_txt(x_pool, os.path.join(pool_folder, f"{block_names[i]}_pool_out.txt"))
            x = x_pool

            x_prelu = block.prelu(x)
            if stage == "prelu" or stage == "all":
                if block_name == "all" or block_name == block_names[i]:
                    prelu_folder = os.path.join(current_block_folder, "prelu")
                    if not os.path.exists(prelu_folder):
                        os.makedirs(prelu_folder)
                    save_ofmap_to_txt(x_prelu, os.path.join(prelu_folder, f"{block_names[i]}_prelu_out.txt"))
            x = x_prelu

            x_bn = block.bn(x)
            if stage == "bn" or stage == "all":
                if block_name == "all" or block_name == block_names[i]:
                    bn_folder = os.path.join(current_block_folder, "bn")
                    if not os.path.exists(bn_folder):
                        os.makedirs(bn_folder)
                    save_ofmap_to_txt(x_bn, os.path.join(bn_folder, f"{block_names[i]}_bn_out.txt"))
            x = x_bn

            if stage == "bin" or stage == "all":
                if block_name == "all" or block_name == block_names[i]:
                    bin_folder = os.path.join(current_block_folder, "bin")
                    if not os.path.exists(bin_folder):
                        os.makedirs(bin_folder)
                    bin_output = x_bn.sign()
                    bin_output[bin_output == -1] = 0
                    save_ofmap_to_txt(bin_output, os.path.join(bin_folder, f"{block_names[i]}_bin_out.txt"))

            if block_name != "all" and block_name == block_names[i]:
                break


def save_convolution_weights_to_coe(tensor, folder_path, file_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    tensor = tensor.flatten().int().cpu().numpy()

    with open(os.path.join(folder_path, f"{file_name}.coe"), 'w') as f:
        f.write("memory_initialization_radix=2;\nmemory_initialization_vector=\n")
        for i in range(0, len(tensor), 64):
            chunk = tensor[i:i + 64]
            bin_string = ''.join(f'{val:01b}' for val in chunk)
            if len(bin_string) < 64:
                bin_string = bin_string.ljust(64, '0')
            bin_string = bin_string[::-1]
            f.write(bin_string + ",\n")
        f.write(";")


def save_convolution_weights_to_txt(tensor, folder_path, file_name):
    tensor = tensor.flatten().int().cpu().numpy()
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(os.path.join(folder_path, f"{file_name}.txt"), 'w') as f:
        for i in range(0, len(tensor), 64):
            chunk = tensor[i:i + 64]
            bin_string = ''.join(f'{val:01b}' for val in chunk)
            if len(bin_string) < 64:
                bin_string = bin_string.ljust(64, '0')
            bin_string = bin_string[::-1]
            f.write(bin_string + "\n")


def save_weights_and_input(model, input_data, base_folder_path):
    os.makedirs(base_folder_path,exist_ok=True)
    weights_folder = os.path.join(base_folder_path, "ecg_weight")
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)

    # Save model input as COE and TXT
    generate_int8_ecg_coe_file(input_data, os.path.join(weights_folder, "ecg_data.coe"))
    generate_int8_ecg_txt_file(input_data, os.path.join(weights_folder, "ecg_data.txt"))

    # Save convolutional weights as COE and TXT
    all_weights = []
    for i in range(1, 7):
        weight = model.state_dict()[f"block{i}.conv.weight"].clone()
        # weight[weight > 0] = 1
        # weight[weight <= 0] = 0
        weight[weight >= 0] = 1
        weight[weight < 0] = 0
        # Expanding the weight to shape (x, x, 8)
        weight_shape = weight.shape
        if weight_shape[-1] < 8:
            padding = torch.zeros_like(weight[..., :8 - weight_shape[-1]])
            expanded_weight = torch.cat([weight, padding], dim=-1)
        else:
            expanded_weight = weight

        all_weights.append(expanded_weight.flatten())

        save_convolution_weights_to_coe(expanded_weight, os.path.join(weights_folder, "block/coe"), f"block{i}_weights")
        save_convolution_weights_to_txt(expanded_weight, os.path.join(weights_folder, "block/txt"), f"block{i}_weights")

    merged_weights = torch.cat(all_weights)
    save_convolution_weights_to_coe(merged_weights, weights_folder, "weights_all")
    save_convolution_weights_to_txt(merged_weights, weights_folder, "weights_all")
    ecg_weight_merge_path = os.path.join(weights_folder, "ecg_weight_merge.txt")
    concatenate_files(os.path.join(weights_folder, "ecg_data.txt"), os.path.join(weights_folder, "weights_all.txt"),
                      ecg_weight_merge_path)
    txt_to_coe(ecg_weight_merge_path, os.path.join(weights_folder, "ecg_weight_merge.coe"))


def merge_ecg_weight_threhold(base_folder_path):
    #生成2个文件，一是ecg+weight+threhold。一是weight+threhold
    weights_folder = os.path.join(base_folder_path, "ecg_weight")
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)
    ecg_weight_threhold_merge_path = os.path.join(weights_folder, "ecg_weight_threhold_merge.txt")

    concatenate_files(os.path.join(weights_folder, "ecg_weight_merge.txt"),
                      os.path.join(base_folder_path, "threhold/txt/block_thre_merge.txt"),
                      ecg_weight_threhold_merge_path)

    weight_threhold_merge_path = os.path.join(weights_folder, "weight_threhold_merge.txt")
    concatenate_files(os.path.join(weights_folder, "weights_all.txt"),
                      os.path.join(base_folder_path, "threhold/txt/block_thre_merge.txt"),
                      weight_threhold_merge_path)

    txt_to_coe(ecg_weight_threhold_merge_path, os.path.join(weights_folder, "ecg_weight_threhold_merge.coe"))

    txt_to_coe(weight_threhold_merge_path, os.path.join(weights_folder, "weight_threhold_merge.coe"))


def save_ofmap_to_txt(tensor, file_path):
    """
    Save a PyTorch tensor to a text file in the same format as the Verilog output.

    Args:
    - tensor: PyTorch tensor of shape [1, 16, 908]
    - file_path: The path where the text file should be saved
    """
    tensor = tensor.squeeze(0).int()  # Remove the first dimension and convert to integer

    with open(file_path, 'w') as file:
        for i in range(tensor.size(1)):
            values = [int(val.item()) for val in tensor[:, i]]
            file.write(", ".join(map(str, values)) + "\n")

    print(f"PyTorch tensor saved to {file_path}")


def int_to_signed_bit(n, num_of_bits):
    if n < -2 ** (num_of_bits - 1) or n > (2 ** (num_of_bits - 1) - 1):
        raise ValueError("Integer out of range")
    if n < 0:
        # Two's complement of negative number
        binary = bin((1 << num_of_bits) + n)[2:]
    else:
        binary = bin(n)[2:].zfill(num_of_bits)
    return binary


def tensor_to_signed_bits_file(tensor1, tensor2, plus_sign, minus_sign, filename, num_ot_bits):
    plus_sign[plus_sign > 0] = 1
    plus_sign[plus_sign < 0] = 0
    minus_sign[minus_sign < 0] = 0
    with open(filename, 'w') as file:
        for n1, n2, sign1, sign2 in zip(tensor1.to(torch.int16).tolist(), tensor2.to(torch.int16).tolist(),
                                        plus_sign.to(torch.int16).tolist(), minus_sign.to(torch.int16).tolist()):
            binary_1 = int_to_signed_bit(n1, num_of_bits=num_ot_bits)
            binary_2 = int_to_signed_bit(n2, num_of_bits=num_ot_bits)
            file.write(str(sign1) + binary_1 + str(sign2) + binary_2 + '\n')


def calcu_threhold(model_best, model_target, base_folder_path, save: bool = False):
    block1_k = model_best.block1.bn.weight / torch.sqrt(model_best.block1.bn.running_var + model_best.block1.bn.eps)
    block1_b = model_best.block1.bn.bias - (
            model_best.block1.bn.running_mean * model_best.block1.bn.weight) / torch.sqrt(
        model_best.block1.bn.running_var + model_best.block1.bn.eps)
    block1_ak = model_best.block1.prelu.weight * block1_k
    block1_threshold_plus_sign = torch.sign(block1_k)
    block1_threshold_minus_sign = torch.sign(block1_ak)
    block1_thre_plus = - block1_b / block1_k
    block1_thre_minus = - block1_b / block1_ak
    block1_thre_plus = block1_thre_plus.ceil() - 0.5
    block1_thre_minus = block1_thre_minus.ceil() - 0.5

    block2_k = model_best.block2.bn.weight / torch.sqrt(model_best.block2.bn.running_var + model_best.block2.bn.eps)
    block2_b = model_best.block2.bn.bias - (
            model_best.block2.bn.running_mean * model_best.block2.bn.weight) / torch.sqrt(
        model_best.block2.bn.running_var + model_best.block2.bn.eps)
    block2_ak = model_best.block2.prelu.weight * block2_k
    block2_threshold_plus_sign = torch.sign(block2_k)
    block2_threshold_minus_sign = torch.sign(block2_ak)
    block2_thre_plus = - block2_b / block2_k
    block2_thre_minus = - block2_b / block2_ak
    block2_thre_plus = block2_thre_plus.ceil() - 0.5
    block2_thre_minus = block2_thre_minus.ceil() - 0.5

    block3_k = model_best.block3.bn.weight / torch.sqrt(model_best.block3.bn.running_var + model_best.block3.bn.eps)
    block3_b = model_best.block3.bn.bias - (
            model_best.block3.bn.running_mean * model_best.block3.bn.weight) / torch.sqrt(
        model_best.block3.bn.running_var + model_best.block3.bn.eps)
    block3_ak = model_best.block3.prelu.weight * block3_k
    block3_threshold_plus_sign = torch.sign(block3_k)
    block3_threshold_minus_sign = torch.sign(block3_ak)
    block3_thre_plus = - block3_b / block3_k
    block3_thre_minus = - block3_b / block3_ak
    block3_thre_plus = block3_thre_plus.ceil() - 0.5
    block3_thre_minus = block3_thre_minus.ceil() - 0.5

    block4_k = model_best.block4.bn.weight / torch.sqrt(model_best.block4.bn.running_var + model_best.block4.bn.eps)
    block4_b = model_best.block4.bn.bias - (
            model_best.block4.bn.running_mean * model_best.block4.bn.weight) / torch.sqrt(
        model_best.block4.bn.running_var + model_best.block4.bn.eps)
    block4_ak = model_best.block4.prelu.weight * block4_k
    block4_threshold_plus_sign = torch.sign(block4_k)
    block4_threshold_minus_sign = torch.sign(block4_ak)
    block4_thre_plus = - block4_b / block4_k
    block4_thre_minus = - block4_b / block4_ak
    block4_thre_plus = block4_thre_plus.ceil() - 0.5
    block4_thre_minus = block4_thre_minus.ceil() - 0.5

    block5_k = model_best.block5.bn.weight / torch.sqrt(model_best.block5.bn.running_var + model_best.block5.bn.eps)
    block5_b = model_best.block5.bn.bias - (
            model_best.block5.bn.running_mean * model_best.block5.bn.weight) / torch.sqrt(
        model_best.block5.bn.running_var + model_best.block5.bn.eps)
    block5_ak = model_best.block5.prelu.weight * block5_k
    block5_threshold_plus_sign = torch.sign(block5_k)
    block5_threshold_minus_sign = torch.sign(block5_ak)
    block5_thre_plus = - block5_b / block5_k
    block5_thre_minus = - block5_b / block5_ak
    block5_thre_plus = block5_thre_plus.ceil() - 0.5
    block5_thre_minus = block5_thre_minus.ceil() - 0.5

    block6_k = model_best.block6.bn.weight / torch.sqrt(model_best.block6.bn.running_var + model_best.block6.bn.eps)
    block6_b = model_best.block6.bn.bias - (
            model_best.block6.bn.running_mean * model_best.block6.bn.weight) / torch.sqrt(
        model_best.block6.bn.running_var + model_best.block6.bn.eps)

    model_target.block1.threhold.threshold_plus.data = block1_thre_plus
    model_target.block1.threhold.threshold_minus.data = block1_thre_minus
    model_target.block1.threhold.threshold_plus_sign.data = block1_threshold_plus_sign
    model_target.block1.threhold.threshold_minus_sign.data = block1_threshold_minus_sign
    model_target.block1.bn = model_best.block1.bn
    model_target.block1.prelu.weight = model_best.block1.prelu.weight
    model_target.block1.conv.weight.data = model_best.block1.conv.weight.sign()

    model_target.block2.threhold.threshold_plus.data = block2_thre_plus
    model_target.block2.threhold.threshold_minus.data = block2_thre_minus
    model_target.block2.threhold.threshold_plus_sign.data = block2_threshold_plus_sign
    model_target.block2.threhold.threshold_minus_sign.data = block2_threshold_minus_sign
    model_target.block2.bn = model_best.block2.bn
    model_target.block2.prelu.weight = model_best.block2.prelu.weight
    model_target.block2.conv.weight.data = model_best.block2.conv.weight.sign()

    model_target.block3.threhold.threshold_plus.data = block3_thre_plus
    model_target.block3.threhold.threshold_minus.data = block3_thre_minus
    model_target.block3.threhold.threshold_plus_sign.data = block3_threshold_plus_sign
    model_target.block3.threhold.threshold_minus_sign.data = block3_threshold_minus_sign
    model_target.block3.conv.weight.data = model_best.block3.conv.weight.sign()
    model_target.block3.bn = model_best.block3.bn
    model_target.block3.prelu.weight = model_best.block3.prelu.weight

    model_target.block4.threhold.threshold_plus.data = block4_thre_plus
    model_target.block4.threhold.threshold_minus.data = block4_thre_minus
    model_target.block4.threhold.threshold_plus_sign.data = block4_threshold_plus_sign
    model_target.block4.threhold.threshold_minus_sign.data = block4_threshold_minus_sign
    model_target.block4.conv.weight.data = model_best.block4.conv.weight.sign()
    model_target.block4.bn = model_best.block4.bn
    model_target.block4.prelu.weight = model_best.block4.prelu.weight

    model_target.block5.threhold.threshold_plus.data = block5_thre_plus
    model_target.block5.threhold.threshold_minus.data = block5_thre_minus
    model_target.block5.threhold.threshold_plus_sign.data = block5_threshold_plus_sign
    model_target.block5.threhold.threshold_minus_sign.data = block5_threshold_minus_sign
    model_target.block5.conv.weight.data = model_best.block5.conv.weight.sign()
    model_target.block5.bn = model_best.block5.bn
    model_target.block5.prelu.weight = model_best.block5.prelu.weight

    model_target.block6.kx_plus_b.k.data = block6_k
    model_target.block6.kx_plus_b.b.data = block6_b
    model_target.block6.kx_plus_b.a.data = model_best.block6.prelu.weight
    model_target.block6.kx_plus_b.ak.data = block6_k * model_best.block6.prelu.weight
    model_target.block6.kx_plus_b.k_ak_b.data = torch.cat(
        [block6_k, block6_k * model_best.block6.prelu.weight, block6_b], dim=0)
    #量化以便于verilog实现
    model_target.block6.kx_plus_b.k_ak_b.data = quantizer(model_target.block6.kx_plus_b.k_ak_b.data)


    model_target.block6.conv.weight.data = model_best.block6.conv.weight.sign()
    model_target.block6.bn = model_best.block6.bn
    model_target.block6.prelu.weight = model_best.block6.prelu.weight
    if save:
        threhold_path = os.path.abspath(os.path.join(os.getcwd(), base_folder_path, "threhold"))
        threhold_txt_path = os.path.abspath(os.path.join(os.getcwd(), base_folder_path, "threhold", "txt"))

        os.makedirs(threhold_path, exist_ok=True)
        os.makedirs(threhold_txt_path, exist_ok=True)

        for i in range(1, 6):
            block_thre_plus = locals()[f"block{i}_thre_plus"]
            block_thre_minus = locals()[f"block{i}_thre_minus"]
            block_thre_plus_sign = locals()[f"block{i}_threshold_plus_sign"]
            block_thre_minus_sign = locals()[f"block{i}_threshold_minus_sign"]
            tensor_to_signed_bits_file(block_thre_plus * 2, block_thre_minus * 2, block_thre_plus_sign,
                                       block_thre_minus_sign, f"{threhold_path}/txt/block{i}_thre.txt", num_ot_bits=31)
        merge_threshold_files(base_folder_path)
        merge_ecg_weight_threhold(base_folder_path)
    return model_target


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    classes_num = 5
    test_size = 0.2

    if classes_num == 5:
        batch_size = 512
        lr = 0.02
        seed = 169
    else:
        batch_size = 512
        lr = 0.02
        seed = 142

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    loader = Loader(batch_size=batch_size, classes_num=classes_num, device=device, test_size=test_size)
    labels, train_loader, test_loader = loader.loader()

    kernel_size, padding, poolsize = 7, 5, 7
    padding_value = -1
    A = [[1, 8, kernel_size, 2, padding, padding_value, poolsize, 2],
         [8, 16, kernel_size, 1, padding, padding_value, poolsize, 2],
         [16, 32, kernel_size, 1, padding, padding_value, poolsize, 2],
         [32, 32, kernel_size, 1, padding, padding_value, poolsize, 2],
         [32, 64, kernel_size, 1, padding, padding_value, poolsize, 2],
         [64, classes_num, kernel_size, 1, padding, padding_value, poolsize, 2],
         ]

    model = ECG_XNOR_Full_Bin(block1=A[0], block2=A[1], block3=A[2], block4=A[3],
                              block5=A[4] if len(A) > 4 else None,
                              block6=A[5] if len(A) > 5 else None,
                              block7=A[6] if len(A) > 6 else None,
                              device=device).to(device)
    model_threhold = ECG_XNOR_Full_akx_plus_b(block1=A[0], block2=A[1], block3=A[2], block4=A[3],
                                              block5=A[4] if len(A) > 4 else None,
                                              block6=A[5] if len(A) > 5 else None,
                                              block7=A[6] if len(A) > 6 else None,
                                              device=device).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Check if a pre-trained model exists
    model_path = './models/ECG_Net_for_5_example.pth'
    if os.path.exists(model_path):
        model = torch.load(model_path, weights_only=False)
        print(f"Loaded pre-trained model from {model_path}")
    else:
        num_epochs = 1000
        train(model=model,
              train_dataloader=train_loader,
              test_dataloader=test_loader,
              optimizer=optimizer,
              loss_fn=loss_fn,
              epochs=num_epochs,
              device=device,
              writer=False,
              classes_num=classes_num)
        torch.save(model.state_dict(), model_path)
        print(f"Trained and saved model as {model_path}")

    return model, loader, model_threhold, device, loss_fn, test_loader


def save_ecg_in_values(input_data, stride, kernel_size, padding, padding_value, output_file):
    # Apply padding to input_data with the specified padding_value
    padded_input = F.pad(input_data, (padding, padding), value=padding_value)

    # Get the shape of the input tensor
    _, _, signal_length = padded_input.shape

    # List to store extracted values
    extracted_values = []

    # Extract values based on the convolution parameters
    for i in range(0, signal_length - kernel_size + 1, stride):
        window = padded_input[0, 0, i:i + kernel_size].tolist()  # Get 7 values
        window = [int(value) for value in window]
        extracted_values.append(window)

    # Save to txt file
    with open(output_file, "w") as f:
        for values in extracted_values:
            f.write(" ".join(map(str, values)) + "\n")

    print(f"Values successfully saved to {output_file}")

def add_psum_files(block_name, base_folder, file1_suffix, file2_suffix, output_suffix):
    # Construct file paths based on block name and folder structure
    folder_path = os.path.join(base_folder, block_name, 'psum')
    file1 = os.path.join(folder_path, f'{block_name}_M1C012_{file1_suffix}.txt')
    file2 = os.path.join(folder_path, f'{block_name}_M1C3_{file2_suffix}.txt')
    output_file = os.path.join(folder_path, f'{block_name}_M1C0123_{output_suffix}.txt')

    def read_file(filename):
        with open(filename, 'r') as f:
            data = [list(map(int, line.split(','))) for line in f]
        return data

    def write_file(filename, data):
        with open(filename, 'w') as f:
            for line in data:
                f.write(', '.join(map(str, line)) + '\n')

    data1 = read_file(file1)
    data2 = read_file(file2)

    result = []
    for line1, line2 in zip(data1, data2):
        result.append([x + y for x, y in zip(line1, line2)])

    write_file(output_file, result)

def add_multiple_psum_files(block_name, base_folder, suffix_list, output_suffix):
    # Construct file paths based on block name and folder structure
    folder_path = os.path.join(base_folder, block_name, 'psum')
    input_files = [os.path.join(folder_path, f'{block_name}_M2C{i}_{suffix_list[i]}.txt') for i in range(len(suffix_list))]
    output_file = os.path.join(folder_path,
                               f'{block_name}_M2C{"".join(map(str, range(len(suffix_list))))}_{output_suffix}.txt')

    def read_file(filename):
        with open(filename, 'r') as f:
            data = [list(map(int, line.split(','))) for line in f]
        return data

    def write_file(filename, data):
        with open(filename, 'w') as f:
            for line in data:
                f.write(', '.join(map(str, line)) + '\n')

    # Read data from all input files
    data_list = [read_file(f) for f in input_files]

    # Initialize result with zeros (assuming all files have the same structure)
    result = [[0] * len(data_list[0][0]) for _ in range(len(data_list[0]))]

    # Sum corresponding elements from each file
    for data in data_list:
        for i, line in enumerate(data):
            result[i] = [x + y for x, y in zip(result[i], line)]

    write_file(output_file, result)


def calculate_shifts(multiplier):
    def find_shifts(multiplier):
        # List to store shifts
        shifts = []

        # Start from the largest power of 2 smaller than or equal to the multiplier
        power = 1
        while power <= multiplier:
            power <<= 1

        # Move back to the actual largest power of 2
        power >>= 1

        while multiplier > 0:
            if multiplier >= power:
                shifts.append(power)
                multiplier -= power
            power >>= 1

        return shifts

    def shifts_to_representation(shifts):
        representation = []
        for shift in shifts:
            if shift == 1:
                representation.append("1")
            else:
                # Calculate how many shifts (log2 of the power)
                representation.append(f"1 << {shift.bit_length() - 1}")
        return representation

    # Find shifts required
    shifts = find_shifts(multiplier)

    # Get the representation in terms of shifts
    shift_rep = shifts_to_representation(shifts)

    # Return the result as a dictionary
    return {
        "representation": " + ".join(shift_rep),
        "total_shifts": len(shift_rep)
    }

def find_data_index(loader, target_data):
    # Loop through the dataset to compare each entry
    for idx, (data, _) in enumerate(loader.train_dataset):
        # Compare the first few elements of the data with the target
        if torch.equal(data[0][:len(target_data)], target_data):
            return idx
    return None  # Return None if not found

def find_data(loader, target_data):
    # Loop through the dataset to compare each entry
    for idx, (data, _) in enumerate(loader.train_dataset):
        # Compare the first few elements of the data with the target
        if torch.equal(data[0][:len(target_data)], target_data):
            return data.unsqueeze(0)  # Return the matching data with unsqueeze
    for idx, (data, _) in enumerate(loader.test_dataset):
        # Compare the first few elements of the data with the target
        if torch.equal(data[0][:len(target_data)], target_data):
            return data.unsqueeze(0)  # Return the matching data with unsqueeze
    return None


def save_all_test_ecgs(test_loader, base_folder_path):
    """
    Convert every ECG in `test_loader` to .coe and .txt files.

    Parameters
    ----------
    test_loader : torch.utils.data.DataLoader
        The DataLoader that yields (data, label) pairs.  `data` is expected
        to be shaped (batch, 1, 3600) or (batch, 3600).
    base_folder_path : str or Path
        Where the sub-folder `ecg_weight` will be created (if it doesn’t exist).
    """
    base_folder_path = Path(base_folder_path)
    ecg_folder   = base_folder_path / "test_dataset_ecg"
    ecg_folder.mkdir(parents=True, exist_ok=True)

    sample_idx = 0
    with torch.no_grad():
        for batch_data, _ in tqdm(test_loader, desc="Converting ECGs"):
            # ---  ensure shape (B, 1, 3600)  ---
            if batch_data.ndim == 2:          # (B, 3600) → (B, 1, 3600)
                batch_data = batch_data.unsqueeze(1)

            for ecg in batch_data:            # iterate one sample at a time
                ecg = ecg.cpu()               # keep it on CPU for file conversion
                # filenames like ecg_00001.coe / ecg_00001.txt
                name_stem = f"ecg_{sample_idx:05d}"
                coe_path  = ecg_folder / f"{name_stem}.coe"
                txt_path  = ecg_folder / f"{name_stem}.txt"

                generate_int8_ecg_coe_file(ecg, coe_path)
                generate_int8_ecg_txt_file(ecg, txt_path)
                sample_idx += 1
import os
from pathlib import Path
from tqdm import tqdm           # optional progress bar

def save_all_test_ecgs_and_preds(model,
                                 test_loader,
                                 device,
                                 base_folder_path,
                                 pred_filename="py_predictions.txt",
                                 true_label_filename: str = "true_labels.txt"
):
    """
    For every ECG in `test_loader`:
      • writes <ecg_weight>/ecg_#####.coe and .txt
      • records the model's predicted label
    Once finished, the predictions are saved to <ecg_weight>/<pred_filename>
    so that line *N* corresponds to ecg_##### where ##### == N-1.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classification model.
    test_loader : torch.utils.data.DataLoader
        Loader that yields (ecg, label) *with shuffle=False*.
    device : torch.device
        Where inference is performed.
    base_folder_path : str | Path
        Parent directory; a sub-folder `ecg_weight` is (re)used/created.
    pred_filename : str
        Name of the text file that will hold the predictions (one per line).
    """
    model.eval()

    base_folder_path = Path(base_folder_path)
    weights_folder   = base_folder_path / "test_dataset_ecg"
    weights_folder.mkdir(parents=True, exist_ok=True)

    test_label_folder = base_folder_path / "test_label"
    test_label_folder.mkdir(parents=True, exist_ok=True)
    predictions, labels = [], []
    sample_idx  = 0

    with torch.inference_mode():
        for batch_ecg, batch_labels in tqdm(test_loader, desc="Saving ECGs + preds"):
            batch_ecg = batch_ecg.to(device)
            batch_labels = batch_labels.to(device)
            # --- model inference ---
            logits = model(batch_ecg)
            _, batch_pred = torch.max(logits.data, dim=1)   # shape (B,)
            batch_pred = batch_pred.cpu().tolist()
            batch_labels = batch_labels.cpu().tolist()

            # --- make sure ECG shape is (B, 1, 3600) before writing ---
            if batch_ecg.ndim == 2:                         # (B, 3600)
                batch_ecg = batch_ecg.unsqueeze(1)           # → (B, 1, 3600)

            # --- per-sample file output ---
            for ecg, pred, true_lbl in zip(batch_ecg.cpu(), batch_pred, batch_labels):
                stem = f"ecg_{sample_idx:05d}"
                generate_int8_ecg_coe_file(ecg, weights_folder / f"{stem}.coe")
                generate_int8_ecg_txt_file(ecg, weights_folder / f"{stem}.txt")

                predictions.append(str(pred))
                labels.append(str(true_lbl))
                sample_idx += 1

    # --- write all predictions at once, keeping order identical to file order ---
    # pred_path = test_label_folder / pred_filename
    (test_label_folder / pred_filename).write_text("\n".join(predictions))
    (test_label_folder / true_label_filename).write_text("\n".join(labels))
    # with open(pred_path, "w") as f:
    #     f.write("\n".join(predictions))

    # print(f"✔ Saved {sample_idx} ECGs and predictions to '{test_label_folder}'.")
    print(f"✔ Saved {sample_idx} ECGs, predictions → '{pred_filename}', "
          f"labels → '{true_label_filename}' in '{test_label_folder}'.")
if __name__ == "__main__":
    model, loader, model_threhold, device, loss_fn, test_loader = main()
    target_data = torch.tensor([-45., -46., -45., -43., -42., -42., -40., -39.]).to(device=device, dtype=torch.float32)
    input_data = find_data(loader, target_data)

    base_folder_path = os.path.abspath(os.path.join(os.getcwd(), "ecg_weight"))
    save_weights_and_input(model, input_data, base_folder_path)
    # ENG: Compute and save the threshold information for each block, and merge the thresholds, ECG data, and weights into a single .coe file to initialize ECG_Weight_Threhold_Mem.
    model_fpga = calcu_threhold(model, model_threhold, base_folder_path, save=True).to(device)
    
    _, threhold_model_acc = test_step(model=model_fpga, dataloader=test_loader, loss_fn=loss_fn, device=device)
    print(threhold_model_acc)
    _, ori_model_acc = test_step(model=model, dataloader=test_loader, loss_fn=loss_fn, device=device)
    print(ori_model_acc)

