import os

import numpy as np
import torch
import torchaudio
import tqdm

CLASSES = [
    "дальше",
    "вперед",
    "назад",
    "вверх",
    "вниз",
    "выше",
    "ниже",
    "домой",
    "громче",
    "тише",
    "лайк",
    "дизлайк",
    "следующий",
    "предыдущий",
    "сначала",
    "перемотай",
    "выключи",
    "стоп",
    "хватит",
    "замолчи",
    "заткнись",
    "останови",
    "пауза",
    "включи",
    "смотреть",
    "продолжи",
    "играй",
    "запусти",
    "ноль",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
]
MAX_LEN = 321


def get_prepare_fn():
    melspec_tr = torchaudio.transforms.MelSpectrogram(
        n_mels=128,  # args.n_mels,
        n_fft=128 * 20,  # args.n_fft,
        hop_length=251,  # args.hop_length,
    )
    amp_to_db_tr = torchaudio.transforms.AmplitudeToDB()

    def prepare_fn(audio):
        mel = melspec_tr(audio)
        out = amp_to_db_tr(mel)

        return out

    return prepare_fn


class AudioDataset(torch.utils.data.Dataset):
    """
    Датасет умеющий выдавать чистые + грязные данные для тренировки,
    как модели голосовой активации, так и денойзера
    """

    def __init__(
        self,
        audio_path,
        prepare_fun,
    ):
        """
        audio_path -- Путь к чистым данны
        noise_path -- Путь к шуму для накладывания на чистые данные, если нужен
        min_amp -- минимальное отношение мощности шума к мощности речи
        max_amp -- максимальное отношение мощности шума к мощности речи

        min_amp и max_amp можно варьировать, чтобы подготовить модель к различным видам зашумления.
        """
        super().__init__()
        self._prepare_fun = prepare_fun
        self._noise = None
        self._pathes, self._labels = self._path_traversall(audio_path)

    def _path_traversall(self, path, label=None):
        """
        Функция для обхода папки с данными
        """
        pathes = []
        labels = []
        for filename in os.listdir(path):
            if os.path.splitext(filename)[1] == ".wav":
                pathes.append(os.path.join(path, filename))
                labels.append(CLASSES.index(label) if label else None)
            elif os.path.isdir(os.path.join(path, filename)):
                new_pathes, new_labels = self._path_traversall(
                    os.path.join(path, filename), label=filename
                )
                pathes.extend(new_pathes)
                labels.extend(new_labels)
            else:
                raise RuntimeError(f"Unknorwn file extension for {filename}")
        return pathes, labels

    def __len__(self):
        return len(self._pathes)

    def __getitem__(self, idx):
        audio, _ = torchaudio.load(self._pathes[idx])
        return_dict = {
            "noise_feats": self._prepare_fun(audio)[0],
            "uttid": os.path.split(self._pathes[idx])[-1],
        }

        return return_dict

def join_in_batch(batch, key):
    """
    Фунция для того, чтобы создать батч из аудиозаписей разной длины.
    """
    max_len = max(b[key].shape[1] for b in batch)
    X = torch.stack(
        [
            torch.nn.functional.pad(b[key], (0, max_len - b[key].shape[1], 0, 0))
            for b in batch
        ],
        dim=0,
    )

    return X


def test_collate_fn(batch):
    X = join_in_batch(batch, "noise_feats")
    X = X.unsqueeze(1)

    Y = [item["uttid"] for item in batch]
    return X, Y



def final_eval(spotter, test_loader, denoiser=None, save_path="results.tsv"):
    spotter.eval()
    if denoiser:
        denoiser.eval()
    results = []
    for x, y in tqdm.tqdm(test_loader):
        if isinstance(x, list):
            preds = [spotter.forward(a.cuda()).softmax(dim=-1) for a in x]
            sms = sum(preds) / len(preds)
        else:
            preds = spotter.forward(x.cuda())
            sms = preds.softmax(dim=-1)

        classes = torch.argmax(sms, dim=-1).cpu().data.numpy()
        labels = [CLASSES[idx] for idx in classes]
        sms = [sms[i, idx] for i, idx in enumerate(classes)]
        results.extend([uttid, label, sm] for uttid, label, sm in zip(y, labels, sms))

    with open(save_path, "w") as fout:
        for uttid, label, _ in results:
            fout.write(f"{uttid}\t{label}\n")

    with open(f"{save_path}_sm", "w") as fout:
        for uttid, _, sm in results:
            fout.write(f"{uttid}\t{sm}\n")


class Ensemble:
    def __init__(self, model_paths):

        self.models = [
            torch.jit.load(model_path).cuda().eval() for model_path in model_paths
        ]

    def forward(self, x):
        x = [model(x).softmax(dim=-1) for model in self.models]
        x = sum(x) / len(x)

        return x

    def eval(self):
        return


def main():
    # model_path = ""
    # model = torch.jit.load(model_path).cuda().eval()

    model = Ensemble(
        [
            "./chkps/efficientnet-b4/model_best_0.953.pt",
            "./chkps/efficientnet-b3/model_best_0.952.pt",
            "./chkps/efficientnet-b2/model_best_0.951.pt",
        ]
    )

    prepare_fn = get_prepare_fn()
    test_dataset = AudioDataset("./data/speech_commands_test", prepare_fun=prepare_fn)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        collate_fn=test_collate_fn,
    )
    with torch.no_grad():
        final_eval(model, test_loader)


if __name__ == "__main__":
    main()
