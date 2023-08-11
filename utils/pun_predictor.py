import json
import os
import re

import numpy as np
import paddle.inference as paddle_infer
from paddlenlp.transformers import ErnieTokenizer


__all__ = ['PunctuationExecutor']


class PunctuationExecutor:
    def __init__(self, model_dir, use_gpu=True, gpu_mem=500, num_threads=4):
        #  config
        model_path = os.path.join(model_dir, 'model.pdmodel')
        params_path = os.path.join(model_dir, 'model.pdiparams')
        if not os.path.exists(model_path) or not os.path.exists(params_path):
            raise Exception("{}{}".format(model_path, params_path))
        self.config = paddle_infer.Config(model_path, params_path)
        # 
        pretrained_token = 'ernie-1.0'
        if os.path.exists(os.path.join(model_dir, 'info.json')):
            with open(os.path.join(model_dir, 'info.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
                pretrained_token = data['pretrained_token']

        if use_gpu:
            self.config.enable_use_gpu(gpu_mem, 0)
        else:
            self.config.disable_gpu()
            self.config.set_cpu_math_library_num_threads(num_threads)
        # enable memory optim
        self.config.enable_memory_optim()
        self.config.disable_glog_info()

        #  config  predictor
        self.predictor = paddle_infer.create_predictor(self.config)

        # 
        self.input_ids_handle = self.predictor.get_input_handle('input_ids')
        self.token_type_ids_handle = self.predictor.get_input_handle('token_type_ids')

        # 
        self.output_names = self.predictor.get_output_names()

        self._punc_list = []
        if not os.path.join(model_dir, 'vocab.txt'):
            raise Exception("{}".format(os.path.join(model_dir, 'vocab.txt')))
        with open(os.path.join(model_dir, 'vocab.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                self._punc_list.append(line.strip())

        self.tokenizer = ErnieTokenizer.from_pretrained(pretrained_token)

        # 
        self('')

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub('[^A-Za-z0-9\u4e00-\u9fa5]', '', text)
        text = re.sub(f'[{"".join([p for p in self._punc_list][1:])}]', '', text)
        return text

    # 
    def preprocess(self, text: str):
        clean_text = self._clean_text(text)
        if len(clean_text) == 0: return None
        tokenized_input = self.tokenizer(list(clean_text), return_length=True, is_split_into_words=True)
        input_ids = tokenized_input['input_ids']
        seg_ids = tokenized_input['token_type_ids']
        seq_len = tokenized_input['seq_len']
        return input_ids, seg_ids, seq_len

    def infer(self, input_ids: list, seg_ids: list):
        # 
        self.input_ids_handle.reshape([1, len(input_ids)])
        self.token_type_ids_handle.reshape([1, len(seg_ids)])
        self.input_ids_handle.copy_from_cpu(np.array([input_ids]).astype('int64'))
        self.token_type_ids_handle.copy_from_cpu(np.array([seg_ids]).astype('int64'))

        # predictor
        self.predictor.run()

        # 
        output_handle = self.predictor.get_output_handle(self.output_names[0])
        output_data = output_handle.copy_to_cpu()
        return output_data

    # 
    def postprocess(self, input_ids, seq_len, preds):
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[1:seq_len - 1])
        labels = preds[1:seq_len - 1].tolist()
        assert len(tokens) == len(labels)

        text = ''
        for t, l in zip(tokens, labels):
            text += t
            if l != 0:
                text += self._punc_list[l]
        return text

    def __call__(self, text: str) -> str:
        # 
        input_ids, seg_ids, seq_len = self.preprocess(text)
        preds = self.infer(input_ids=input_ids, seg_ids=seg_ids)
        if len(preds.shape) == 2:
            preds = preds[0]
        text = self.postprocess(input_ids, seq_len, preds)
        return text