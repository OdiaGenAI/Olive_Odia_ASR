import argparse
import functools

import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("audio_path", type=str, default="dataset/test.wav",              help="")
add_arg("model_path", type=str, default="models/whisper-tiny-finetune",  help="")
add_arg("language",   type=str, default="Oriya",                         help="")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="")
add_arg("local_files_only", type=bool, default=True,  help="")
args = parser.parse_args()
print_arguments(args)

# Whisper
processor = WhisperProcessor.from_pretrained(args.model_path,
                                             language=args.language,
                                             task=args.task,
                                             local_files_only=args.local_files_only)
forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)

# 
model = WhisperForConditionalGeneration.from_pretrained(args.model_path,
                                                        device_map="auto",
                                                        local_files_only=args.local_files_only).half()
model.eval()

# 
sample, sr = librosa.load(args.audio_path, sr=16000)
duration = sample.shape[-1]/sr
assert duration < 30, f"This program is only suitable for inferring audio less than 30 seconds, the current audio {duration} seconds, use another inference program!"

# 
input_features = processor(sample, sampling_rate=sr, return_tensors="pt", do_normalize=True).input_features.cuda().half()
# 
predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=256)
# 
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(f"result ：{transcription}")
