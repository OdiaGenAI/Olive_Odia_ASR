import argparse
import functools
import os

from faster_whisper import WhisperModel

from utils.utils import print_arguments, add_arguments

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("audio_path",  type=str,  default="dataset/test.wav",        help="")
add_arg("model_path",  type=str,  default="models/whisper-tiny-finetune-ct2", help="")
add_arg("language",    type=str,  default="zh",   help="")
add_arg("use_gpu",     type=bool, default=True,   help="")
add_arg("use_int8",    type=bool, default=False,  help="int8")
add_arg("beam_size",   type=int,  default=10,     help="")
add_arg("num_workers", type=int,  default=1,      help="")
add_arg("vad_filter",  type=bool, default=False,  help="")
add_arg("local_files_only", type=bool, default=True, help="")
args = parser.parse_args()
print_arguments(args)

# 
assert os.path.exists(args.model_path), f"{args.model_path}"
# 
if args.use_gpu:
    if not args.use_int8:
        model = WhisperModel(args.model_path, device="cuda", compute_type="float16", num_workers=args.num_workers,
                             local_files_only=args.local_files_only)
    else:
        model = WhisperModel(args.model_path, device="cuda", compute_type="int8_float16", num_workers=args.num_workers,
                             local_files_only=args.local_files_only)
else:
    model = WhisperModel(args.model_path, device="cpu", compute_type="int8", num_workers=args.num_workers,
                         local_files_only=args.local_files_only)
# 
_, _ = model.transcribe("dataset/test.wav", beam_size=5)


# 
segments, info = model.transcribe(args.audio_path, beam_size=args.beam_size, language=args.language,
                                  vad_filter=args.vad_filter)
for segment in segments:
    text = segment.text
    print(f"[{round(segment.start, 2)} - {round(segment.end, 2)}]：{text}\n")
