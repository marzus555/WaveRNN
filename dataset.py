import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, ids, path, mel_len, hop_length, mode, pad, ap, eval=False):
        self.path = path
        self.metadata = ids
        self.eval = eval
        self.mel_len = mel_len
        self.pad = pad 
        self.hop_length = hop_length
        self.mode = mode
        self.ap = ap

        #wav_files = [f"{self.path}wavs/{file}.wav" for file in self.metadata]
        #with Pool(4) as pool:
        #    self.wav_cache = pool.map(self.ap.load_wav, wav_files)

    def __getitem__(self, index):
        file = self.metadata[index]
        m = np.load(f"{self.path}mel/{file}.npy")
        #x = self.wav_cache[index]
        if 5 > m.shape[-1]:
            print(' [!] Instance is too short! : {}'.format(file))
            self.metadata[index] = self.metadata[index + 1]
            file = self.metadata[index]
            m = np.load(f"{self.path}mel/{file}.npy")
        if self.mode in ['gauss', 'mold']:
            x = self.ap.load_wav(f"{self.path}wavs/{file}.wav")
        elif type(self.mode) is int:
            x = np.load(f'{self.path}quant/{file}.npy')
        else:
            raise RuntimeError("Unknown dataset mode - ", self.mode)
        return m, x, file

    def __len__(self):
        return len(self.metadata)

    def collate(self, batch):
        seq_len = self.mel_len * self.hop_length
        pad = self.pad  # kernel size 5
        mel_win = seq_len // self.hop_length + 2 * pad
        max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
        if self.eval:
            mel_offsets = [100] * len(batch)
        else:
            mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
        sig_offsets = [(offset + pad) * self.hop_length for offset in mel_offsets]
            
        for i, x in enumerate(batch):
            sigOffsetBefore = sig_offsets[i]
            maxSigOffsetBefore = sig_offsets[i] + seq_len + 1
            
            maxSize = x[1].shape[0]
            maxSigOffset = sig_offsets[i] + seq_len + 1
            if maxSigOffset > maxSize:
                maxSigOffset = maxSize
            sig_offsets[i] = maxSize - (seq_len + 2)
            
            sigOffsetAfter = sig_offsets[i]
            maxSigOffsetAfter = sig_offsets[i] + seq_len + 1
            if (sigOffsetBefore - sigOffsetAfter) != 0:
                print('there was a difference in sig offset')
                print('sigOffsetBefore')
                print(sigOffsetBefore)
                print('maxSigOffsetBefore')
                print(maxSigOffsetBefore)
                print('sigOffsetAfter')
                print(sigOffsetAfter)
                print('maxSigOffsetAfter')
                print(maxSigOffsetAfter)
            
        mels = [
            x[0][:, mel_offsets[i] : mel_offsets[i] + mel_win]
            for i, x in enumerate(batch)
        ]
        coarse = [
            x[1][sig_offsets[i] : sig_offsets[i] + seq_len + 1]
            for i, x in enumerate(batch)
        ]
        
        mels = np.stack(mels).astype(np.float32)
        if self.mode in ['gauss', 'mold']:
            try:
                coarse = np.stack(coarse).astype(np.float32)
            except ValueError:
                print('value error')
                for i, x in enumerate(batch):
                    print(sig_offsets[i])
                    print(sig_offsets[i] + seq_len + 1)
                    print(x[1].shape)
                    print('coarse.items shape')
                    print((x[1][sig_offsets[i] : sig_offsets[i] + seq_len + 1]).shape)
                coarse = np.stack(coarse).astype(np.float32)
                
            coarse = torch.FloatTensor(coarse)
            x_input = coarse[:, :seq_len]
        elif type(self.mode) is int:
            coarse = np.stack(coarse).astype(np.int64)
            coarse = torch.LongTensor(coarse)
            x_input = 2 * coarse[:, :seq_len].float() / (2 ** self.mode - 1.0) - 1.0
        y_coarse = coarse[:, 1:]
        mels = torch.FloatTensor(mels)
        return x_input, mels, y_coarse
