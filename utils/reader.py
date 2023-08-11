import json
import os
import random
import sys
from typing import List

import librosa
import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.binary import DatasetReader


class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 processor,
                 mono=True,
                 language=None,
                 timestamps=False,
                 sample_rate=16000,
                 min_duration=0.5,
                 max_duration=30,
                 augment_config_path=None):
        """
        Args:
            data_list_path: 
            processor: Whisper
            mono: True
            language: 
            timestamps: 
            sample_rate: 16000
            min_duration: 0.5s
            max_duration: 30s
            augment_config_path: 
        """
        super(CustomDataset, self).__init__()
        assert min_duration >= 0.5, f"min_duration 0.5：{min_duration}"
        assert max_duration <= 30, f"max_duration 30：{max_duration}"
        self.data_list_path = data_list_path
        self.processor = processor
        self.data_list_path = data_list_path
        self.sample_rate = sample_rate
        self.mono = mono
        self.language = language
        self.timestamps = timestamps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.vocab = self.processor.tokenizer.get_vocab()
        self.timestamp_begin = self.vocab['<|notimestamps|>'] + 1
        self.startoftranscript = self.vocab['<|startoftranscript|>']
        self.endoftext = self.vocab['<|endoftext|>']
        self.nocaptions = self.vocab['<|nocaptions|>']
        self.data_list: List[dict] = []
        # 
        self._load_data_list()
        # 
        self.augment_configs = None
        self.noises_path = None
        self.speed_rates = None
        if augment_config_path:
            with open(augment_config_path, 'r', encoding='utf-8') as f:
                self.augment_configs = json.load(f)

    # 
    def _load_data_list(self):
        if self.data_list_path.endswith(".header"):
            # 
            self.dataset_reader = DatasetReader(data_header_path=self.data_list_path,
                                                min_duration=self.min_duration,
                                                max_duration=self.max_duration)
            self.data_list = self.dataset_reader.get_keys()
        else:
            # 
            with open(self.data_list_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            self.data_list = []
            for line in tqdm(lines, desc=''):
                if isinstance(line, str):
                    line = json.loads(line)
                if not isinstance(line, dict): continue
                # 
                if line["duration"] < self.min_duration:
                    continue
                if self.max_duration != -1 and line["duration"] > self.max_duration:
                    continue
                self.data_list.append(dict(line))

    # 
    def _get_list_data(self, idx):
        if self.data_list_path.endswith(".header"):
            data_list = self.dataset_reader.get_data(self.data_list[idx])
        else:
            data_list = self.data_list[idx]
        # 
        audio_file = data_list["audio"]['path']
        transcript = data_list["sentences"] if self.timestamps else data_list["sentence"]
        language = data_list["language"] if 'language' in data_list.keys() else None
        if 'start_time' not in data_list["audio"].keys():
            sample, sample_rate = soundfile.read(audio_file, dtype='float32')
        else:
            start_time, end_time = data_list["audio"]["start_time"], data_list["audio"]["end_time"]
            # 
            sample, sample_rate = self.slice_from_file(audio_file, start=start_time, end=end_time)
        sample = sample.T
        # 
        if self.mono:
            sample = librosa.to_mono(sample)
        # 
        if self.augment_configs:
            sample, sample_rate = self.augment(sample, sample_rate)
        # 
        if self.sample_rate != sample_rate:
            sample = self.resample(sample, orig_sr=sample_rate, target_sr=self.sample_rate)
        return sample, sample_rate, transcript, language

    def _load_timestamps_transcript(self, transcript: List[dict]):
        assert isinstance(transcript, list), f"transcript list：{type(transcript)}"
        data = dict()
        labels = self.processor.tokenizer.prefix_tokens[:3]
        for t in transcript:
            # 
            start = t['start'] if round(t['start'] * 100) % 2 == 0 else t['start'] + 0.01
            start = self.timestamp_begin + round(start * 100) // 2
            end = t['end'] if round(t['end'] * 100) % 2 == 0 else t['end'] - 0.01
            end = self.timestamp_begin + round(end * 100) // 2
            label = self.processor(text=t['text']).input_ids[4:-1]
            labels.extend([start])
            labels.extend(label)
            labels.extend([end])
        data['labels'] = labels + [self.endoftext]
        return data

    def __getitem__(self, idx):
        try:
            # 
            sample, sample_rate, transcript, language = self._get_list_data(idx=idx)
            # 
            self.processor.tokenizer.set_prefix_tokens(language=language if language is not None else self.language)
            if len(transcript) > 0:
                # 
                if self.timestamps:
                    data = self._load_timestamps_transcript(transcript=transcript)
                    # 
                    data["input_features"] = self.processor(audio=sample, sampling_rate=self.sample_rate).input_features
                else:
                    # 
                    data = self.processor(audio=sample, sampling_rate=self.sample_rate, text=transcript)
            else:
                # 
                data = self.processor(audio=sample, sampling_rate=self.sample_rate)
                data['labels'] = [self.startoftranscript, self.nocaptions, self.endoftext]
            return data
        except Exception as e:
            print(f'idx：{idx} error - {e}', file=sys.stderr)
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.data_list)

    # 
    @staticmethod
    def slice_from_file(file, start, end):
        sndfile = soundfile.SoundFile(file)
        sample_rate = sndfile.samplerate
        duration = round(float(len(sndfile)) / sample_rate, 3)
        start = round(start, 3)
        end = round(end, 3)
        # 
        if start < 0.0: start += duration
        if end < 0.0: end += duration
        # 
        if start < 0.0: start = 0.0
        if end > duration: end = duration
        if end < 0.0:
            raise ValueError("(%f s)" % end)
        if start > end:
            raise ValueError("(%f s)(%f s)" % (start, end))
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        sample = sndfile.read(frames=end_frame - start_frame, dtype='float32')
        return sample, sample_rate

    # 
    def augment(self, sample, sample_rate):
        for config in self.augment_configs:
            if config['type'] == 'speed' and random.random() < config['prob']:
                if self.speed_rates is None:
                    min_speed_rate, max_speed_rate, num_rates = config['params']['min_speed_rate'], \
                        config['params']['max_speed_rate'], config['params']['num_rates']
                    self.speed_rates = np.linspace(min_speed_rate, max_speed_rate, num_rates, endpoint=True)
                rate = random.choice(self.speed_rates)
                sample = self.change_speed(sample, speed_rate=rate)
            if config['type'] == 'shift' and random.random() < config['prob']:
                min_shift_ms, max_shift_ms = config['params']['min_shift_ms'], config['params']['max_shift_ms']
                shift_ms = random.randint(min_shift_ms, max_shift_ms)
                sample = self.shift(sample, sample_rate, shift_ms=shift_ms)
            if config['type'] == 'volume' and random.random() < config['prob']:
                min_gain_dBFS, max_gain_dBFS = config['params']['min_gain_dBFS'], config['params']['max_gain_dBFS']
                gain = random.randint(min_gain_dBFS, max_gain_dBFS)
                sample = self.volume(sample, gain=gain)
            if config['type'] == 'resample' and random.random() < config['prob']:
                new_sample_rates = config['params']['new_sample_rates']
                new_sample_rate = np.random.choice(new_sample_rates)
                sample = self.resample(sample, orig_sr=sample_rate, target_sr=new_sample_rate)
                sample_rate = new_sample_rate
            if config['type'] == 'noise' and random.random() < config['prob']:
                min_snr_dB, max_snr_dB = config['params']['min_snr_dB'], config['params']['max_snr_dB']
                if self.noises_path is None:
                    self.noises_path = []
                    noise_dir = config['params']['noise_dir']
                    if os.path.exists(noise_dir):
                        for file in os.listdir(noise_dir):
                            self.noises_path.append(os.path.join(noise_dir, file))
                noise_path = random.choice(self.noises_path)
                snr_dB = random.randint(min_snr_dB, max_snr_dB)
                sample = self.add_noise(sample, sample_rate, noise_path=noise_path, snr_dB=snr_dB)
        return sample, sample_rate

    # 
    @staticmethod
    def change_speed(sample, speed_rate):
        if speed_rate == 1.0:
            return sample
        if speed_rate <= 0:
            raise ValueError("error")
        old_length = sample.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        sample = np.interp(new_indices, old_indices, sample).astype(np.float32)
        return sample

    # 
    @staticmethod
    def shift(sample, sample_rate, shift_ms):
        duration = sample.shape[0] / sample_rate
        if abs(shift_ms) / 1000.0 > duration:
            raise ValueError("shift_ms")
        shift_samples = int(shift_ms * sample_rate / 1000)
        if shift_samples > 0:
            sample[:-shift_samples] = sample[shift_samples:]
            sample[-shift_samples:] = 0
        elif shift_samples < 0:
            sample[-shift_samples:] = sample[:shift_samples]
            sample[:-shift_samples] = 0
        return sample

    # 
    @staticmethod
    def volume(sample, gain):
        sample *= 10.**(gain / 20.)
        return 

    # 
    @staticmethod
    def resample(sample, orig_sr, target_sr):
        sample = librosa.resample(sample, orig_sr=orig_sr, target_sr=target_sr)
        return sample

    # 
    def add_noise(self, sample, sample_rate, noise_path, snr_dB, max_gain_db=300.0):
        noise_sample, sr = librosa.load(noise_path, sr=sample_rate)
        # 
        target_db = -20
        gain = min(max_gain_db, target_db - self.rms_db(sample))
        sample *= 10. ** (gain / 20.)
        # 
        sample_rms_db, noise_rms_db = self.rms_db(sample), self.rms_db(noise_sample)
        noise_gain_db = min(sample_rms_db - noise_rms_db - snr_dB, max_gain_db)
        noise_sample *= 10. ** (noise_gain_db / 20.)
        # 
        if noise_sample.shape[0] < sample.shape[0]:
            diff_duration = sample.shape[0] - noise_sample.shape[0]
            noise_sample = np.pad(noise_sample, (0, diff_duration), 'wrap')
        elif noise_sample.shape[0] > sample.shape[0]:
            start_frame = random.randint(0, noise_sample.shape[0] - sample.shape[0])
            noise_sample = noise_sample[start_frame:sample.shape[0] + start_frame]
        sample += noise_sample
        return sample

    @staticmethod
    def rms_db(sample):
        mean_square = np.mean(sample ** 2)
        return 10 * np.log10(mean_square)
        