import argparse
import os
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
import librosa
from torch.package import PackageImporter

def pad(x, max_len=96000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def load_sample(sample_path, max_len=96000):
    y_list = []
    y, sr = librosa.load(sample_path, sr=None)
    if sr != 24000:
        y = librosa.resample(y, orig_sr=sr, target_sr=24000)
    if len(y) <= max_len:
        return [Tensor(pad(y, max_len))]
    for i in range(int(np.ceil(len(y) / max_len))):
        start = i * max_len
        end = min((i + 1) * max_len, len(y))
        y_seg = y[start:end]
        y_list.append(Tensor(pad(y_seg, max_len)))
    return y_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    input_path = args.input_path
    model_path = args.model_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    # ✅ Load the pretrained model using PackageImporter
    print('Loading pretrained model from:', model_path)
    importer = PackageImporter(model_path)
    model = importer.load_pickle("model", "model.pkl")  # this matches the original saved object
    model.to(device)
    model.eval()

    out_list_multi = []
    out_list_binary = []

    for m_batch in load_sample(input_path):
        m_batch = m_batch.to(device=device, dtype=torch.float).unsqueeze(0)
        logits, multi_logits = model(m_batch)

        probs = F.softmax(logits, dim=-1)
        probs_multi = F.softmax(multi_logits, dim=-1)

        out_list_multi.append(probs_multi.tolist()[0])
        out_list_binary.append(probs.tolist()[0])

    result_multi = np.average(out_list_multi, axis=0).tolist()
    result_binary = np.average(out_list_binary, axis=0).tolist()

    print('Multi classification result : gt:{}, wavegrad:{}, diffwave:{}, parallel wave gan:{}, wavernn:{}, wavenet:{}, melgan:{}'.format(
        result_multi[0], result_multi[1], result_multi[2], result_multi[3],
        result_multi[4], result_multi[5], result_multi[6]
    ))
    print('Binary classification result : fake:{}, real:{}'.format(result_binary[0], result_binary[1]))
