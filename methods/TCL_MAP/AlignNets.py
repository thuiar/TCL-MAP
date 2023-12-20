import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

__all__ = ['CTCModule', 'AlignSubNet', 'SimModule']

class CTCModule(nn.Module):
    def __init__(self, in_dim, out_seq_len, args):
        '''
        This module is performing alignment from A (e.g., audio) to B (e.g., text).
        :param in_dim: Dimension for input modality A
        :param out_seq_len: Sequence length for output modality B
        From: https://github.com/yaohungt/Multimodal-Transformer
        '''
        super(CTCModule, self).__init__()
        # Use LSTM for predicting the position from A to B
        self.pred_output_position_inclu_blank = nn.LSTM(in_dim, out_seq_len+1, num_layers=2, batch_first=True) # 1 denoting blank
        self.out_seq_len = out_seq_len
        
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        '''
        :input x: Input with shape [batch_size x in_seq_len x in_dim]
        '''
        # NOTE that the index 0 refers to blank. 
        
        pred_output_position_inclu_blank, _ = self.pred_output_position_inclu_blank(x)

        prob_pred_output_position_inclu_blank = self.softmax(pred_output_position_inclu_blank) # batch_size x in_seq_len x out_seq_len+1
        prob_pred_output_position = prob_pred_output_position_inclu_blank[:, :, 1:] # batch_size x in_seq_len x out_seq_len
        prob_pred_output_position = prob_pred_output_position.transpose(1,2) # batch_size x out_seq_len x in_seq_len
        pseudo_aligned_out = torch.bmm(prob_pred_output_position, x) # batch_size x out_seq_len x in_dim
        
        # pseudo_aligned_out is regarded as the aligned A (w.r.t B)
        # return pseudo_aligned_out, (pred_output_position_inclu_blank)
        return pseudo_aligned_out

# similarity-based modality alignment
class SimModule(nn.Module):
    def __init__(self, in_dim_x, in_dim_y, shared_dim, out_seq_len, args):
        '''
        This module is performing alignment from A (e.g., audio) to B (e.g., text).
        :param in_dim: Dimension for input modality A
        :param out_seq_len: Sequence length for output modality B
        '''
        super(SimModule, self).__init__()
        # Use LSTM for predicting the position from A to B
        self.ctc = CTCModule(in_dim_x, out_seq_len, args)
        self.eps = args.eps
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.proj_x = nn.Linear(in_features=in_dim_x, out_features=shared_dim)
        self.proj_y = nn.Linear(in_features=in_dim_y, out_features=shared_dim)

        self.fc1 = nn.Linear(in_features=out_seq_len, out_features=round(out_seq_len / 2))
        self.fc2 = nn.Linear(in_features=round(out_seq_len / 2), out_features=out_seq_len)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        '''
        :input x: Input with shape [batch_size x in_seq_len x in_dim]
        '''

        pseudo_aligned_out = self.ctc(x)

        x_common = self.proj_x(pseudo_aligned_out)
        x_n = x_common.norm(dim=-1, keepdim=True)
        x_norm = x_common / torch.max(x_n, self.eps * torch.ones_like(x_n))
        
        y_common = self.proj_y(y)
        y_n = y_common.norm(dim=-1, keepdim=True)
        y_norm = y_common / torch.max(y_n, self.eps * torch.ones_like(y_n))
            
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        similarity_matrix = logit_scale * torch.bmm(y_norm, x_norm.permute(0, 2, 1))
        
        logits = similarity_matrix.softmax(dim=-1)
        logits = self.fc1(logits)
        logits = self.relu(logits)
        logits = self.fc2(logits)
        logits = self.sigmoid(logits)
        
        aligned_out = torch.bmm(logits, pseudo_aligned_out)

        return aligned_out



class AlignSubNet(nn.Module):
    def __init__(self, args, mode):
        """
        mode: the way of aligning
            avg_pool, ctc, conv1d
        """
        super(AlignSubNet, self).__init__()
        assert mode in ['avg_pool', 'ctc', 'conv1d', 'sim']

        in_dim_t, in_dim_v, in_dim_a = args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim

        seq_len_t, seq_len_v, seq_len_a = args.max_cons_seq_length, args.video_seq_len, args.audio_seq_len
        self.dst_len = seq_len_t
        self.dst_dim = in_dim_t
        self.mode = mode

        self.ALIGN_WAY = {
            'avg_pool': self.__avg_pool,
            'ctc': self.__ctc,
            'conv1d': self.__conv1d,
            'sim': self.__sim,
        }

        if mode == 'conv1d':
            self.conv1d_t = nn.Conv1d(seq_len_t, self.dst_len, kernel_size=1, bias=False)
            self.conv1d_v = nn.Conv1d(seq_len_v, self.dst_len, kernel_size=1, bias=False)
            self.conv1d_a = nn.Conv1d(seq_len_a, self.dst_len, kernel_size=1, bias=False)
        elif mode == 'ctc':
            self.ctc_t = CTCModule(in_dim_t, self.dst_len, args)
            self.ctc_v = CTCModule(in_dim_v, self.dst_len, args)
            self.ctc_a = CTCModule(in_dim_a, self.dst_len, args)
        elif mode == 'sim':
            self.shared_dim = args.shared_dim
            self.sim_t = SimModule(in_dim_t, self.dst_dim,  self.shared_dim, self.dst_len, args)
            self.sim_v = SimModule(in_dim_v, self.dst_dim, self.shared_dim, self.dst_len, args)
            self.sim_a = SimModule(in_dim_a, self.dst_dim, self.shared_dim, self.dst_len, args)

    def get_seq_len(self):
        return self.dst_len
    
    def __ctc(self, text_x, video_x, audio_x):
        text_x = self.ctc_t(text_x) if text_x.size(1) != self.dst_len else text_x
        video_x = self.ctc_v(video_x) if video_x.size(1) != self.dst_len else video_x
        audio_x = self.ctc_a(audio_x) if audio_x.size(1) != self.dst_len else audio_x
        return text_x, video_x, audio_x

    def __avg_pool(self, text_x, video_x, audio_x):
        def align(x):
            raw_seq_len = x.size(1)
            if raw_seq_len == self.dst_len:
                return x
            if raw_seq_len // self.dst_len == raw_seq_len / self.dst_len:
                pad_len = 0
                pool_size = raw_seq_len // self.dst_len
            else:
                pad_len = self.dst_len - raw_seq_len % self.dst_len
                pool_size = raw_seq_len // self.dst_len + 1
            pad_x = x[:, -1, :].unsqueeze(1).expand([x.size(0), pad_len, x.size(-1)])
            x = torch.cat([x, pad_x], dim=1).view(x.size(0), pool_size, self.dst_len, -1)
            x = x.mean(dim=1)
            return x
        text_x = align(text_x)
        video_x = align(video_x)
        audio_x = align(audio_x)
        return text_x, video_x, audio_x
    
    def __conv1d(self, text_x, video_x, audio_x):
        text_x = self.conv1d_t(text_x) if text_x.size(1) != self.dst_len else text_x
        video_x = self.conv1d_v(video_x) if video_x.size(1) != self.dst_len else video_x
        audio_x = self.conv1d_a(audio_x) if audio_x.size(1) != self.dst_len else audio_x
        return text_x, video_x, audio_x
    
    def __sim(self, text_x, video_x, audio_x):
        
        text_x = self.sim_t(text_x, text_x) if text_x.size(1) != self.dst_len else text_x
        video_x = self.sim_v(video_x, text_x) if video_x.size(1) != self.dst_len else video_x
        audio_x = self.sim_a(audio_x, text_x) if audio_x.size(1) != self.dst_len else audio_x
        return text_x, video_x, audio_x

    def forward(self, text_x, video_x, audio_x):
        # already aligned
        if text_x.size(1) == video_x.size(1) and text_x.size(1) == audio_x.size(1):
            return text_x, video_x, audio_x
        return self.ALIGN_WAY[self.mode](text_x, video_x, audio_x)
    