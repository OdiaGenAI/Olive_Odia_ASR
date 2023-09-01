<p align="center">
  <a href="url"><img src="https://github.com/OdiaGenAI/Olive_Odia_ASR/blob/main/odiagen_asr-modified.png" height="auto" width="200" style="border-radius:100%"></a>
</p>


<h1 align ="center">Olive Odia ASR</h1>


# Introduction OpenAI Whisper
OpenAI has made the Whisper project publicly available, which boasts human-level speech recognition capabilities in English and additionally offers automatic speech recognition for 98 other languages. Whisper facilitates automatic speech recognition and translation tasks, enabling the conversion of spoken language into text across multiple languages, followed by translation into English. The primary objective of this initiative is to optimize the Whisper model using Lora. This optimization can be applied to both timestamped and non-timestamped data, as well as data without speech information.

Currently, OpenAI has made several models within this project open source, and specific details can be found on the OpenAI website. Notably, the project also includes support for accelerated reasoning through CTranslate2 and GGML. Accelerated reasoning allows for the direct utilization of the original Whisper model transformation without the necessity for extensive fine-tuning

## Supporting models
- openai/whisper-tiny
- openai/whisper-base
- openai/whisper-small
- openai/whisper-medium
- openai/whisper-large
- openai/whisper-large-v2

## Introduction of the main program of the project

1. `aishell.py`: Generate AIShell training data.
2. `finetune.py`: Refine the model through fine-tuning.
3. `merge_lora.py`: Combine Whisper and Lora models.
4. `evaluation.py`: Assess the performance of either the fine-tuned model or the original Whisper model.
5. `infer_tfs.py`: Utilize the transformers library to directly invoke the fine-tuned model or the original Whisper model for predictions, suitable for short audio clip inference.
6. `infer_ct2.py`: Apply the converted CTranslate2 model for predictions, primarily for reference in program usage.
7. `infer_gui.py`: Offer a graphical user interface for operation, employing the converted CTranslate2 model for predictions.
8. `infer_server.py`: Deploy the converted CTranslate2 model to the server for use in client applications.
9. `convert-ggml.py`: Adapt the model into GGML format for utilization in Android or Windows applications.


## OpenAI Whisper Model


|      Model       | Language | aishell_test | test_net | test_meeting |                            Download link                             |                             CTranslate2                              |                                 GGML                                 |
|:----------------:|:--------:|:------------:|:--------:|:------------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|
|   whisper-tiny   | Odia  |   0.0    | 0.0  |   0.0    | [Download]() | [Download]() | [Download]() |
|   whisper-base   | Odia  |   0.0    | 0.0  |   0.50378    | [Download]() | [Download]() | [Download]() |
|  whisper-small   | Odia  |   0.0    | 0.0  |   0.0    | [Download]() | [Download]() | [Download]() |
|  whisper-medium  | Odia  |   0.0    | 0.0  |   0.0    | [Download]() | [Download]() | [Download]() |
|  whisper-large   | Odia  |   0.0    | 0.0  |   0.0    | [Download]() | [Download]() | [Download]() |
| whisper-large-v2 | Odia  |   0.0    | 0.0  |   0.0    | [Download]() | [Download]() | [Download]() |


## Fine-tune

Once our data is prepared, we can proceed with the fine-tuning of our model. The critical aspects in this process are the two parameters in  [HuggingFace](https://huggingface.co/openai): `--base_model` for specifying the Whisper model to be fine-tuned and `--output_path` for designating the checkpoint path where Lora saves progress during training. When choosing a --base_model, you can either provide a HuggingFace model URL (which doesn't require pre-downloading) or specify a local path if `--local_files_only` is set to True. To optimize training speed, it's advisable to set `--use_8bit` to False. For additional available parameters, refer to the program documentation.

### Single-GPU

The single card training command is as follows. Windows can do this without the `CUDA_VISIBLE_DEVICES` parameter.

```shell
CUDA_VISIBLE_DEVICES=0 python finetune.py --base_model=openai/whisper-tiny --output_dir=output/
```

## Inference

Run the provided code to enable speech recognition, which leverages transformers to interact with either the fine-tuned model or the original Whisper model for predictions. This approach is best suited for processing brief audio clips. In case you need to handle longer speeches, you can consult the `infer_ct2.py` script. The initial argument, `--audio_path`, is used to specify the path to the audio file you want to analyze, while the second argument, `--model_path`, designates the location of the combined model. You can also opt to utilize the original Whisper model directly, for instance, by using `openai/whisper-large-v2`. For additional configuration options, please refer to the documentation of this program.

```shell
python infer_tfs.py --audio_path=dataset/test.wav --model_path=models/whisper-tiny-finetune
```

## Web deploy

Web deployment can be expedited using CTranslate2, as indicated in the documentation above. The `--host` option designates the address at which the service will initiate, denoted as `0.0.0.0`, allowing access from any address. The `--port` parameter specifies the port number for usage. The `--model_path` parameter designates the transformed CTranslate2 model. Additionally, the `--num_workers` parameter specifies the number of threads for concurrent inference, a crucial consideration for web deployments handling multiple simultaneous requests. For additional parameters, refer to the provided program.

```shell
python infer_server.py --host=0.0.0.0 --port=5000 --model_path=models/whisper-tiny-finetune-ct2 --num_workers=2
```

### API docs

Currently, there are two available interfaces: the standard recognition interface, denoted as `/recognition`, and the streaming return result interface, referred to as `/recognition_stream`. It's important to clarify that the term "stream" in this context pertains to the process of uploading the entire audio and subsequently receiving the recognition results in real-time. This approach is particularly beneficial for achieving a seamless long speech recognition experience. It's worth noting that both interfaces share identical documentation and utilize the following interface parameters.

|   Field    | Need |  type  |  Default   |                                  Explain                                  |
|:----------:|:----:|:------:|:----------:|:-------------------------------------------------------------------------:|
|   audio    | Yes  |  File  |            |                                Audio File                                 |
| to_simple  |  No  |  int   |     1      |                 Traditional Odia to Simplified Odia                 |
| remove_pun |  No  |  int   |     0      |                       Whether to remove punctuation                       |
|    task    |  No  | String | transcribe |         Identify task types and support transcribe and translate          |
|  language  |  No  | String |     or     | Set the language, shorthand, to automatically detect the language if None |

Return result:

|  Field  | type |                       Explain                       |
|:-------:|:----:|:---------------------------------------------------:|
| results | list | Recognition results separated into individual parts |
| +result | str  |   Text recognition result for each separated part   |
| +start  | int  |    Start time in seconds for each separated part    |
|  +end   | int  |     End time in seconds for each separated part     |
|  code   | int  |   Error code, 0 indicates successful recognition    |

Example:

```json
{
  "results": [
    {
      "result": "ମୁଁ ଭାଷା ଅନୁବାଦକ |",
      "start": 0,
      "end": 3
    }
  ],
  "code": 0
}
```

To make it easier to understand, here is the Python code to call the Web interface. Here is how to call `/recognition`.

API URL : https://odiagenai-olive-whisper-asr.hf.space

```python
import requests

response = requests.post(url="http://127.0.0.1:5000/recognition",
                         files=[("audio", ("test.wav", open("dataset/test.wav", 'rb'), 'audio/wav'))],
                         json={"to_simple": 0, "remove_pun": 0, "language": "or", "task": "transcribe"}, timeout=20)
print(response.text)
```

Here is how `/recognition stream` is called.

```python
import json
import requests

response = requests.post(url="http://127.0.0.1:5000/recognition_stream",
                         files=[("audio", ("test.wav", open("dataset/test_long.wav", 'rb'), 'audio/wav'))],
                         json={"to_simple": 0, "remove_pun": 0, "language": "or", "task": "transcribe"}, stream=True,
                         timeout=20)
for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
    if chunk:
        result = json.loads(chunk.decode())
        text = result["result"]
        start = result["start"]
        end = result["end"]
        print(f"[{start} - {end}]：{text}")
```

The provided test page is as follows:

The home page `http://127.0.0.1:5000/` looks like this:

Document page `http://127.0.0.1:5000/docs` page is as follows:


```
@misc{OdiaGenAI,
  author = { Sambit Sekhar and Shantipriya Parida },
  title = {OdiaGenAI: Odia ASR },
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sam-ai}},
}
```

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

