import json

import torch
from flask import Flask, request, jsonify

from model.lofi2lofi_model import Decoder as Lofi2LofiDecoder
from model.lyrics2lofi_model import Lyrics2LofiModel
from lofi2lofi_generate import decode
from lyrics2lofi_predict import predict

device = "cpu"
app = Flask(__name__)

lofi2lofi_checkpoint = "checkpoints/lofi2lofi_decoder.pth"
print("Loading lofi model...", end=" ")
lofi2lofi_model = Lofi2LofiDecoder(device=device)
lofi2lofi_model.load_state_dict(torch.load(lofi2lofi_checkpoint, map_location=device))
print(f"Loaded {lofi2lofi_checkpoint}.")
lofi2lofi_model.to(device)
lofi2lofi_model.eval()

lyrics2lofi_checkpoint = "checkpoints/lyrics2lofi.pth"
print("Loading lyrics2lofi model...", end=" ")
lyrics2lofi_model = Lyrics2LofiModel(device=device)
lyrics2lofi_model.load_state_dict(torch.load(lyrics2lofi_checkpoint, map_location=device))
print(f"Loaded {lyrics2lofi_checkpoint}.")
lyrics2lofi_model.to(device)
lyrics2lofi_model.eval()


@app.route('/')
def home():
    return 'Server running'


@app.route('/decode', methods=['GET'])
def decode_input():
    input = request.args.get('input')
    number_list = json.loads(input)
    json_output = decode(lofi2lofi_model, torch.tensor([number_list]).float())
    response = jsonify(json_output)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.route('/predict', methods=['GET'])
def lyrics_to_track():
    input = request.args.get('input')
    json_output = predict(lyrics2lofi_model, input)
    response = jsonify(json_output)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5173)
