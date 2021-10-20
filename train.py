import argparse
import os
import random
from collections import OrderedDict
from pathlib import Path

import albumentations as A
import apex
import cv2
import efficientnet_pytorch
import numpy as np
import timm
import torch
import torch.distributed
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchaudio
import tqdm
from torch.utils.tensorboard import SummaryWriter

# Список классов для предсказаний
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        help="path to train dir",
        default="./data",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="logs")
    parser.add_argument("--model-name", type=str, default="efficientnet-b7")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--n-classes", type=int, default=len(CLASSES))
    parser.add_argument("--optim", type=str, default="fusedadam", help="optimizer name")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", type=str, default="cosa", help="scheduler name")
    parser.add_argument("--T-max", type=int, default=25)
    parser.add_argument(
        "--num-workers",
        type=int,
        help="number of data loader workers",
        default=16,
    )
    parser.add_argument(
        "--num-epochs", type=int, help="number of epochs to train", default=1025
    )
    parser.add_argument("--batch-size", type=int, help="batch size", default=32)
    parser.add_argument(
        "--random-state",
        type=int,
        help="random seed",
        default=42,
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        help="number of folds",
        default=3,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        help="fold",
        default=0,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="gpu",
        default=0,
    )
    parser.add_argument(
        "--distributed", action="store_true", help="distributed training"
    )
    parser.add_argument("--syncbn", action="store_true", help="sync batchnorm")
    parser.add_argument(
        "--deterministic", action="store_true", help="deterministic training"
    )
    parser.add_argument(
        "--load", type=str, default="", help="path to pretrained model weights"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="path to pretrained model to resume training",
    )
    parser.add_argument(
        "--channels-last", action="store_true", help="Use channels_last memory layout"
    )
    parser.add_argument("--fp16", action="store_true", help="fp16 training")
    parser.add_argument("--albu", action="store_true", help="use albu augs")
    parser.add_argument("--ft", action="store_true", help="use whole dataset")
    parser.add_argument("--mixup", action="store_true", help="use mixup")
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--pseudo", action="store_true", help="use mixup")

    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=128 * 20)  # 128*20)
    parser.add_argument("--hop-length", type=int, default=251)

    parser.add_argument("--tmask", type=int, default=60)
    parser.add_argument("--fmask", type=int, default=40)

    args = parser.parse_args()

    return args


def add_noise(clean, noise, min_amp, max_amp):
    """
    Функция, которая используется для зашумления данных
    clean и noise -- это считанные аудиозаписи с чистой речью и шумом соответственно
    min_amp -- минимальное отношение мощности шума к мощности речи
    max_amp -- максимальное отношение мощности шума к мощности речи
    """
    # степень шума случайна от min_amp до max_amp
    noise_amp = np.random.uniform(min_amp, max_amp)
    # если запись с шумом короче, чем чистая, то она дублируется нужное число раз
    noise = noise.repeat(1, clean.shape[1] // noise.shape[1] + 2)
    # так как теперь шумная запись длиннее, то выбираем случайный момент начала шумной записи
    start = np.random.randint(0, noise.shape[1] - clean.shape[1] + 1)
    noise_part = noise[:, start : start + clean.shape[1]]
    # накладываем шум
    noise_mult = clean.abs().max() / noise_part.abs().max() * noise_amp
    return (clean + noise_part * noise_mult) / (1 + noise_amp)


def get_img_augment(args):
    tmask = torchaudio.transforms.TimeMasking(time_mask_param=args.tmask)  # 60)  # 40
    fmask = torchaudio.transforms.FrequencyMasking(
        freq_mask_param=args.fmask
    )  # 40)  # 40

    def img_augment(img):
        img = tmask(img)
        img = fmask(img)

        return img

    return img_augment


def get_prepare_fn(args):
    melspec_tr = torchaudio.transforms.MelSpectrogram(
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    amp_to_db_tr = torchaudio.transforms.AmplitudeToDB()

    def prepare_fn(audio):
        mel = melspec_tr(audio)
        out = amp_to_db_tr(mel)
        # out /= 80

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
        noise_path=None,
        min_amp: float = 0.0,
        max_amp: float = 5.0,
        augment_fn=None,
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
        self._min_amp = min_amp
        self._max_amp = max_amp
        self._noise = None
        self._augment_fn = augment_fn
        if audio_path is None:
            assert noise_path is None
            return
        self._pathes, self._labels = self._path_traversall(audio_path)
        if noise_path:
            noise, _ = self._path_traversall(noise_path)
            self._noise = [torchaudio.load(path)[0] for path in noise]

    @classmethod
    def _create_from_loaded(cls, pathes, labels, noise, prepare_fun, min_amp, max_amp):
        """
        Создаёт новый датасет из уже подгруженного.
        Используется при сплите датасета на train и val.
        """
        obj = AudioDataset(
            None, prepare_fun=prepare_fun, min_amp=min_amp, max_amp=max_amp
        )
        obj._pathes = pathes
        obj._labels = labels
        obj._noise = noise
        return obj

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
        return_dict = {}
        if self._noise is not None:
            # Если в класс передавались шумы, то тут происходит зашумление
            noise_idx = np.random.randint(0, len(self._noise))
            noise_audio = add_noise(
                audio, self._noise[noise_idx], self._min_amp, self._max_amp
            )
            return_dict["noise_feats"] = self._prepare_fun(noise_audio)[0]
        else:
            # Если же нет, то вместо шумов выдаётся чистый звук
            return_dict["noise_feats"] = self._prepare_fun(audio)[0]

        if self._augment_fn is not None:
            if np.random.random() < 0.5:
                return_dict["noise_feats"] = self._augment_fn(
                    return_dict["noise_feats"]
                )

        if self._labels[idx] is not None:
            return_dict["label"] = self._labels[idx]

        return return_dict

    def split(self, train_part=0.9, seed=42, ft=False):
        """
        Функция для того чтобы разделить датасет на 2 части, например, на train и val.
        """
        np.random.seed(seed)
        train_pathes = []
        train_labels = []
        val_pathes = []
        val_labels = []
        for idx in range(len(self._pathes)):
            if np.random.rand() < train_part:
                train_pathes.append(self._pathes[idx])
                train_labels.append(self._labels[idx])
            else:
                val_pathes.append(self._pathes[idx])
                val_labels.append(self._labels[idx])

        train_dataset = AudioDataset._create_from_loaded(
            train_pathes,
            train_labels,
            self._noise,
            self._prepare_fun,
            self._min_amp,
            self._max_amp,
        )
        if not ft:
            train_dataset._augment_fn = self._augment_fn

        val_dataset = AudioDataset._create_from_loaded(
            val_pathes,
            val_labels,
            self._noise,
            self._prepare_fun,
            self._min_amp,
            self._max_amp,
        )
        if not ft:
            val_dataset._augment_fn = self._augment_fn

        return train_dataset, val_dataset


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


# Дальше функции для того, чтобы склеить записи в один батч на все случаи жизни
def spotter_collate_fn(batch):
    X = join_in_batch(batch, "noise_feats")
    X = X.unsqueeze(1)

    Y = torch.tensor([item["label"] for item in batch])
    return X, Y


def denoiser_collate_fn(batch):
    X = join_in_batch(batch, "noise_feats")
    X = X.unsqueeze(1)

    Y = join_in_batch(batch, "clean_feats")
    return X, Y


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, size=x.size(0))
        lam = x.new_tensor(lam).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    else:
        lam = 1

    bs = x.size(0)
    index = torch.randperm(bs)
    mixed_x = lam * x + (1 - lam) * x[index, :]

    return mixed_x, y, y[index], lam


def spotter_eval(model, val_loader, criterion, local_rank=0):
    """
    Фунция подсчёта качества на валидационных модели на валидационных данных.
    """
    model.eval()
    val_loss = val_acc = val_batches = 0

    if local_rank == 0:
        pbar = tqdm.tqdm(
            total=len(val_loader),
            desc=" [ Validataing.. ] ",
            mininterval=2,
            leave=False,
        )

    for x, y in val_loader:
        x = x.cuda(local_rank, non_blocking=True)
        y = y.cuda(local_rank, non_blocking=True)

        bs = x.size(0)

        logits = model.forward(x)

        loss = criterion(logits, y)

        val_loss += loss.mean().item() * bs

        val_acc += (torch.argmax(logits, dim=-1) == y).sum().item()
        val_batches += bs

        torch.cuda.synchronize()

        if local_rank == 0:
            postfix = {
                "loss": f"{val_loss / val_batches:.3f}",
                "acc": f"{val_acc / val_batches:.3f}",
            }
            pbar.set_postfix(**postfix)
            pbar.update()

    if local_rank == 0:
        pbar.close()

    return val_loss / val_batches, val_acc / val_batches, val_batches


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_dist(args):
    # to autotune convolutions and other algorithms
    # to pick the best for current configuration
    torch.backends.cudnn.benchmark = True

    if args.deterministic:
        set_seed(args.random_state)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_printoptions(precision=10)

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def save_jit(model, args, model_path):
    model.eval()
    if args.model_name.startswith("efficientnet-b"):
        if hasattr(model, "module"):
            model.module.set_swish(memory_efficient=False)
        else:
            model.set_swish(memory_efficient=False)

    inp = torch.rand(1, args.in_channels, 128, 128, device="cuda")

    with torch.no_grad():
        traced_model = torch.jit.trace(model, inp)

    traced_model.save(model_path)

    if args.model_name.startswith("efficientnet-b"):
        if hasattr(model, "module"):
            model.module.set_swish(memory_efficient=True)
        else:
            model.set_swish(memory_efficient=True)


def train(args):
    if args.distributed:
        init_dist(args)

    torch.backends.cudnn.benchmark = True

    checkpoint_dir = Path(args.checkpoint_dir)
    summary_writer = None
    if args.local_rank == 0:
        summary_writer = SummaryWriter(checkpoint_dir / "logs")

    model = build_model(args)
    model = model.cuda(args.gpu)

    checkpoint = None
    if args.load:
        path_to_resume = Path(args.load).expanduser()
        if path_to_resume.is_file():
            print(f"=> loading resume checkpoint '{path_to_resume}'")
            checkpoint = torch.load(
                path_to_resume,
                map_location=lambda storage, loc: storage.cuda(args.gpu),
            )
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:] if k.startswith("module") else k
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
            print(
                f"=> resume from checkpoint '{path_to_resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            print(f"=> no checkpoint found at '{path_to_resume}'")

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    weight_decay = args.weight_decay
    if weight_decay > 0:  # and filter_bias_and_bn:
        skip = {}
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()

        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    optimizer = build_optimizer(parameters, args)

    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        if args.syncbn:
            model = apex.parallel.convert_syncbn_model(model)

        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

    # Подгрузим тренировочный сет в двух вариантах: чистый и с зашумлением
    data_dir = Path(args.data_dir)
    prepare_fn = get_prepare_fn(args)
    img_augment = get_img_augment(args)
    fullnoiseset = AudioDataset(
        data_dir / "speech_commands_train",
        prepare_fn,
        noise_path=data_dir / "noises",
        augment_fn=img_augment,
    )
    train_dataset, val_dataset = fullnoiseset.split(seed=args.random_state, ft=args.ft)

    if args.ft:
        print("ft")
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

    if args.pseudo:
        test_dataset = AudioDataset(
            data_dir / "speech_commands_test_pseudo",
            prepare_fn,
            augment_fn=img_augment,
        )
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    args.num_workers = min(args.batch_size, 16)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=spotter_collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True,
        # worker_init_fn=worker_init_fn,
    )
    val_batch_size = 4 * args.batch_size
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=spotter_collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
    )

    scheduler = build_scheduler(optimizer, args)

    scaler = None
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    criterion = nn.CrossEntropyLoss(reduction="none" if args.mixup else "mean")

    def saver(path, score):
        torch.save(
            {
                "epoch": epoch,
                "best_score": best_score,
                "score": score,
                "state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "sched_state_dict": scheduler.state_dict()
                if scheduler is not None
                else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "args": args,
            },
            path,
        )

    start_epoch = 0
    best_score = 0
    if args.resume and checkpoint is not None:
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        if checkpoint["sched_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["sched_state_dict"])

        optimizer.load_state_dict(checkpoint["opt_state_dict"])

        if checkpoint["scaler"] is not None:
            scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(start_epoch, args.num_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc, train_batches = 0, 0, 0
        model.train()
        if args.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(train_loader),
                desc=f" [ Training {epoch}/{args.num_epochs}..] ",
                leave=False,
                mininterval=2,
            )

        for x, y in train_loader:
            bs = x.size(0)
            x = x.cuda(args.local_rank, non_blocking=True)
            y = y.cuda(args.local_rank, non_blocking=True)

            optimizer.zero_grad()

            if args.mixup:
                x, Y_A, Y_B, lam = mixup_data(x, y, alpha=args.alpha)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    if args.mixup:
                        lam = lam.squeeze()
                        loss_a = criterion(logits, Y_A)
                        loss_b = criterion(logits, Y_B)

                        loss = lam * loss_a + (1 - lam) * loss_b
                        loss = loss.mean()
                    else:
                        loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 2.0
                )  # , error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model.forward(x)

                if args.mixup:
                    lam = lam.squeeze()
                    loss_a = criterion(logits, Y_A)
                    loss_b = criterion(logits, Y_B)

                    loss = lam * loss_a + (1 - lam) * loss_b
                    loss = loss.mean()
                else:
                    loss = criterion(logits, y)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 2.0
                )  # , error_if_nonfinite=False)
                optimizer.step()

            train_loss += loss.item() * bs
            train_acc += (torch.argmax(logits, dim=-1) == y).sum().item()
            train_batches += bs

            torch.cuda.synchronize()

            if args.local_rank == 0:
                postfix = {
                    "loss": f"{train_loss / train_batches:.3f}",
                    "acc": f"{train_acc / train_batches:.3f}",
                }
                pbar.set_postfix(**postfix)
                pbar.update()

        train_loss /= train_batches
        train_acc /= train_batches

        with torch.no_grad():
            val_loss, val_acc, n = spotter_eval(
                model, val_loader, criterion, local_rank=args.local_rank
            )

        if args.distributed:
            val_accs = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(val_accs, val_acc)
            ns = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(ns, n)
            val_acc = sum([a * n for a, n in zip(val_accs, ns)]) / sum(ns)

        if scheduler is not None:
            scheduler.step()

        if args.local_rank == 0:

            for idx, param_group in enumerate(optimizer.param_groups):
                lr = param_group["lr"]
                summary_writer.add_scalar(
                    "group{}/lr".format(idx), float(lr), global_step=epoch
                )

            summary_writer.add_scalar("loss/train", train_loss, global_step=epoch)
            summary_writer.add_scalar("loss/dev", val_loss, global_step=epoch)
            summary_writer.add_scalar("accuracy/train", train_acc, global_step=epoch)
            summary_writer.add_scalar("accuracy/dev", val_acc, global_step=epoch)

            score = val_acc

            if score > best_score:
                best_score = score

                saver(checkpoint_dir / "model_best.pth", best_score)
                save_jit(
                    model, args, checkpoint_dir / f"model_best_{best_score:.3f}.pt"
                )

            saver(checkpoint_dir / "model_last.pth", score)
            save_jit(model, args, checkpoint_dir / "model_last.pt")

            if epoch % (2 * args.T_max) == (args.T_max - 1):
                saver(checkpoint_dir / f"model_last_{epoch + 1}.pth", score)
                save_jit(model, args, checkpoint_dir / f"model_last_{epoch + 1}.pt")

    if args.local_rank == 0:
        summary_writer.close()


def build_model(args):
    if args.model_name.startswith("efficientnet-b"):
        model = efficientnet_pytorch.EfficientNet.from_pretrained(
            model_name=args.model_name,
            in_channels=args.in_channels,
            num_classes=args.n_classes,
        )
    else:
        model = timm.create_model(
            args.model_name,  # "efficientnetv2_l",
            pretrained=True,
            in_chans=args.in_channels,
            num_classes=args.n_classes,
        )

    return model


def build_optimizer(parameters, args):
    if args.optim.lower() == "fusedadam":
        optimizer = apex.optimizers.FusedAdam(
            parameters,
            adam_w_mode=True,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optim.lower() == "fusedsgd":
        optimizer = apex.optimizers.FusedSGD(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optim.lower() == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError(f"not yet implemented {args.optim}")

    return optimizer


def build_scheduler(optimizer, args):
    scheduler = None

    if args.scheduler.lower() == "cosa":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.T_max,
            eta_min=max(args.learning_rate * 1e-2, 1e-7),
        )
    elif args.scheduler.lower() == "cosawr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_max,
            T_mult=2,
            eta_min=max(args.learning_rate * 1e-2, 1e-7),
        )
    else:
        print("No scheduler")

    return scheduler


def main():
    args = parse_args()
    if args.local_rank == 0:
        print(args)

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    if args.local_rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
