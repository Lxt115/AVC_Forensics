"""Implementation of ResNetAudio+MS-TCN"""

import json
import math
import torch
import torch.nn as nn

from .resnet import ResNet, BasicBlock
from .tcn import MultibranchTemporalConvNet
from .audio_model import ResNetAudio, GRU, BasicBlockAudio
from utils import calculateNorm2


def load_json(json_fp):
    with open(json_fp, "r") as f:
        json_content = json.load(f)
    return json_content


def get_model(weights_forgery_path, device, mode):
    """ "
    Get Resnet+MS-TCN model, optionally with pre-trained weights
    加载模型参数
    Parameters
    ----------
    mode
    weights_forgery_path : str
        Path to file with network weights
    device : str
        Device to put model on
    """
    args_loaded = load_json("./models/configs/lrw_resnet18_mstcn.json")
    relu_type = args_loaded["relu_type"]
    tcn_options = {
        "num_layers": args_loaded["tcn_num_layers"],
        "kernel_size": args_loaded["tcn_kernel_size"],
        "dropout": args_loaded["tcn_dropout"],
        "dwpw": args_loaded["tcn_dwpw"],
        "width_mult": args_loaded["tcn_width_mult"],
    }
    model = SelectModel(mode, inputDim=256, hiddenDim=512, nClasses=2, frameLen=29,
                        relu_type=relu_type, tcn_options=tcn_options)
    new_dict = model.state_dict()
    # load weights learned during face forgery detection
    if weights_forgery_path is not None:
        checkpoint_dict = torch.load(weights_forgery_path, map_location=lambda storage, loc: storage.cuda(device))
        old_dict = checkpoint_dict["model"]
        old_dict = {k: v for k, v in old_dict.items() if k in new_dict}
        new_dict.update(old_dict)
        model.load_state_dict(new_dict)
        print("Face forgery weights loaded.")
    else:
        print("Randomly initialised weights.")

    calculateNorm2(model)
    model.to(device)
    return model


def reshape_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


def _average_batch(x, lengths):
    return torch.stack([torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0)


class MultiscaleMultibranchTCN(nn.Module):
    def __init__(self, input_size, num_channels, tcn_options, dropout, relu_type, dwpw=False):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options["kernel_size"]
        self.num_kernels = len(self.kernel_sizes)

        self.mb_ms_tcn = MultibranchTemporalConvNet(
            input_size, num_channels, tcn_options, dropout=dropout, relu_type=relu_type, dwpw=dwpw
        )

    def forward(self, x, lengths):
        # 32, 25, 512
        x = x.transpose(1, 2)  # (32, 512, 25)
        out = self.mb_ms_tcn(x)
        return out


class SelectModel(nn.Module):
    def __init__(self, mode, inputDim=256, hiddenDim=512, nClasses=2, frameLen=29, relu_type="prelu",
                 tcn_options={}):
        super(SelectModel, self).__init__()
        self.mode = mode
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.nLayers = 2
        self.frontend_nout = 64  # 前向输出通道数
        self.backend_out = 512  # 后向输出通道数
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)  # 残差网络

        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == "prelu" else nn.ReLU()  # 参数化修正线性单元
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            # 三维卷积 (32, 64, 25, 44, 44), batch, channel, depth, width, height
            nn.BatchNorm3d(self.frontend_nout),  # 批标准化
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # (32, 64, 25, 22, 22)
        )
        self.tcn = MultiscaleMultibranchTCN(
            input_size=self.backend_out,  # 512
            num_channels=[inputDim * len(tcn_options["kernel_size"]) * tcn_options["width_mult"]]
                         * tcn_options["num_layers"],  # [256 * 3 * 1] * 4
            tcn_options=tcn_options,
            dropout=tcn_options["dropout"],
            relu_type=relu_type,
            dwpw=tcn_options["dwpw"],
        )
        # frontend1D
        self.fronted1D = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        # resnet
        self.resnet18 = ResNetAudio(BasicBlockAudio, [2, 2, 2, 2], num_classes=self.inputDim)
        # backend_gru
        self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers)
        # mode output
        self.fc = nn.Linear(768 if self.mode == 'visual' else self.hiddenDim * 2, self.nClasses)
        # merge to the same dimension
        self.match = nn.Linear(768, self.hiddenDim * 2)
        # visual output
        self.consensus_func = _average_batch
        self._initialize_weights()

        self.encoder_audio = nn.ModuleList([torch.nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=1024) for _ in range(4)])
        self.encoder_image = nn.ModuleList([torch.nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=1024) for _ in range(4)])
        self.fusion = nn.Embedding(num_embeddings=1, embedding_dim=1024)
        self.down1 = nn.Linear(self.hiddenDim * 4, self.hiddenDim * 2)
        self.down2 = nn.Linear(self.hiddenDim * 2, self.hiddenDim)
        self.down3 = nn.Linear(self.hiddenDim, self.nClasses)


    def forward(self, image_data, audio_data, lengths):
        if self.mode == "visual":
            # frontend  (32, 1, 25, 88, 88)
            image_data = self.frontend3D(image_data)  # (32, 64, 25, 22, 22)
            t_new = image_data.shape[2]  # 25
            image_data = reshape_tensor(image_data)  # (32*25, 64, 22, 22)
            image_data = self.trunk(image_data)  # (32*25, 512)
            # backend
            image_data = image_data.view(-1, t_new, image_data.size(1))  # (32, 25, 512)
            image_data = self.tcn(image_data, lengths)
            image_data = self.consensus_func(image_data, lengths)
            return self.fc(image_data)
        elif self.mode == "audio":
            audio_data = audio_data.view(-1, 1, audio_data.size(1))
            audio_data = self.fronted1D(audio_data)
            audio_data = audio_data.contiguous()
            audio_data = self.resnet18(audio_data)
            audio_data = audio_data.view(-1, self.frameLen, self.inputDim)
            audio_data = self.gru(audio_data)
            return self.fc(audio_data[:, -1, :])
        else:
            # frontend  (B, 1, 25, 88, 88)
            image_data = self.frontend3D(image_data)  # (B, 64, 25, 22, 22)
            t_new = image_data.shape[2]  # 25
            image_data = reshape_tensor(image_data)  # (B*25, 64, 22, 22)
            image_data = self.trunk(image_data)  # (B*25, 512)
            # backend
            image_data = image_data.view(-1, t_new, image_data.size(1))  # (B, 25, 512)
            image_data = self.tcn(image_data, lengths)
            image_data = image_data.permute([0, 2, 1])
            image_data = self.match(image_data)

            audio_data = audio_data.view(-1, 1, audio_data.size(1))
            audio_data = self.fronted1D(audio_data)
            audio_data = audio_data.contiguous()
            audio_data = self.resnet18(audio_data)
            audio_data = audio_data.view(-1, self.frameLen, self.inputDim)
            audio_data = self.gru(audio_data)
            bs = image_data.shape[0]
            fusion = self.fusion(torch.tensor([0]).to('cuda'))  # [1024]
            fusion = fusion.view(1, 1, -1)
            fusion = fusion.expand(bs, 1, -1)

            audio_data = torch.cat([audio_data, fusion], dim=1)
            image_data = torch.cat([image_data, fusion], dim=1)
            for i in range(4):
                audio_data = self.encoder_audio[i](audio_data)
                image_data = self.encoder_image[i](audio_data)
                fusion_audio = audio_data[:, -1, :]
                fusion_image = image_data[:, -1, :]
                fusion = fusion_image + fusion_audio
                audio_data.data[:, -1, :] = fusion
                image_data.data[:, -1, :] = fusion

            input_data = torch.cat((audio_data, image_data), dim=2)
            avg_pool = torch.nn.AvgPool2d(kernel_size=(input_data.size(1), 1))
            input_data = input_data.unsqueeze(1)
            fusion_vector = avg_pool(input_data)
            fusion_vector = fusion_vector.squeeze(1).squeeze(1)
            output = self.down1(fusion_vector)
            output = self.down2(output)
            return self.down3(output)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

