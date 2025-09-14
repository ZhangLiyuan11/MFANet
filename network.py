import math
import torch
import torch.nn as nn

class Alignment(nn.Module):
    def __init__(self):
        super(Alignment, self).__init__()
        self.clip_alignment = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.clap_alignment = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),  # 比ELU更平滑
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

    def forward(self, clip_text_fea, clip_video_fea, clap_text_fea, clap_audio_fea):
        clip_text_out = self.clip_alignment(clip_text_fea)
        clap_text_out = self.clap_alignment(clap_text_fea)
        clip_img_out = self.clip_alignment(clip_video_fea)
        clap_audio_out = self.clap_alignment(clap_audio_fea)
        return clip_text_out, clip_img_out, clap_text_out, clap_audio_out

class Fusion_layer(nn.Module):
    def __init__(self,input_dim,out_dim):
        super(Fusion_layer, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.correlation_layer = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, fea_victor1, fea_victor2):
        input1 = fea_victor1.unsqueeze(2)
        input2 = fea_victor2.unsqueeze(1)
        feature_dim = fea_victor1.shape[1]
        similarity = torch.matmul(input1, input2) / math.sqrt(feature_dim)
        fusion = self.correlation_layer(self.pooling(similarity).squeeze())
        return fusion



class FeatureMask(nn.Module):
    def __init__(self, projection_dim):
        super(FeatureMask, self).__init__()
        self.mask = nn.Parameter(torch.zeros(int(projection_dim)))

    def forward(self):
        return torch.sigmoid(torch.ones_like(self.mask) * self.mask)

class Finalfusion(nn.Module):
    def __init__(self,fea_dim):
        super(Finalfusion, self).__init__()
        self.fea_dim = fea_dim
        self.senet = nn.Sequential(
            nn.Linear(3, 3),
            nn.GELU(),
            nn.Linear(3, 3),
        )
        # 平均池化和最大池化
        self.avepooling =  nn.AvgPool1d(self.fea_dim, stride=1)
        self.maxpooling =  nn.MaxPool1d(self.fea_dim, stride=1)
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self,text_fea,video_fea,audio_fea):
        # Combine features
        final_feature = torch.cat([text_fea.unsqueeze(1), video_fea.unsqueeze(1), audio_fea.unsqueeze(1)], 1)

        # Pooling and transformation
        s1 = self.avepooling(final_feature)
        s2 = self.maxpooling(final_feature)
        s1 = s1.view(s1.size(0), -1)
        s2 = s2.view(s2.size(0), -1)
        s1 = self.senet(s1)
        s2 = self.senet(s2)
        s = self.sigmoid(s1 + s2)
        s = s.view(s.size(0), s.size(1), 1)

        # Apply pooling weights
        final_feature = torch.mean(s * final_feature, dim=1)
        return final_feature


class MulModel(torch.nn.Module):
    def __init__(self, fea_dim, dropout):
        super(MulModel, self).__init__()
        self.dim = fea_dim
        self.dropout = dropout

        self.alignment = Alignment()
        self.text_fusion_layer = Fusion_layer(self.dim, self.dim)
        self.tv_fusion_layer = Fusion_layer(self.dim, self.dim)
        self.ta_fusion_layer = Fusion_layer(self.dim, self.dim)

        # 生成残差
        self.text_clip = nn.Sequential(nn.Linear(1024, self.dim),nn.ReLU())
        self.text_clap = nn.Sequential(nn.Linear(512, self.dim), nn.ReLU())
        self.video_clip = nn.Sequential(nn.Linear(1024, self.dim), nn.ReLU())
        self.audio_clap = nn.Sequential(nn.Linear(512, self.dim), nn.ReLU())

        self.final_fusion_layer = Finalfusion(self.dim)

        self.classifier = nn.Sequential(
            nn.Linear(fea_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, **kwargs):
        clip_text_fea = kwargs['clip_text_fea']
        clip_video_fea = kwargs['clip_video_fea']
        clap_text_fea = kwargs['clap_text_fea']
        clap_audio_fea = kwargs['clap_audio_fea']
        clip_text_out, clip_img_out, clap_text_out, clap_audio_out = self.alignment(clip_text_fea, clip_video_fea, clap_text_fea, clap_audio_fea)
        text_fea = clip_text_out
        tv_fea = self.tv_fusion_layer(text_fea,clip_img_out) + self.video_clip(clip_video_fea)
        ta_fea = self.ta_fusion_layer(text_fea,clap_audio_out) + self.audio_clap(clap_audio_fea)
        text_fea = text_fea + self.text_clip(clip_text_fea)
        final_fea = self.final_fusion_layer(text_fea,tv_fea,ta_fea)
        output = self.classifier(final_fea)
        return output,clip_text_out,clap_text_out
