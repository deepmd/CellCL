from itertools import chain

import torch
from torch import nn


class ISD(nn.Module):
    def __init__(self, base_encoder, K=128000, m=0.99, T_t=0.01, T_s=0.1, projection=True, **kwargs):
        super(ISD, self).__init__()

        self.K = K
        self.m = m
        self.T_t = T_t
        self.T_s = T_s

        # both encoders should have same arch
        self.encoder_q, feat_dim = base_encoder()
        self.encoder_k, _ = base_encoder()

        if projection:
            hidden_dim = feat_dim * 2
            proj_dim = feat_dim // 4
            self.projection_q = self._get_mlp(feat_dim, hidden_dim, proj_dim)
            self.projection_k = self._get_mlp(feat_dim, hidden_dim, proj_dim)
        else:
            hidden_dim = feat_dim
            proj_dim = feat_dim
            self.projection_q = nn.Identity()
            self.projection_k = nn.Identity()

        self.predict_q = self._get_mlp(proj_dim, hidden_dim, proj_dim)

        # copy query encoder weights to key encoder
        for param_q, param_k in zip(chain(self.encoder_q.parameters(), self.projection_q.parameters()),
                                    chain(self.encoder_k.parameters(), self.projection_k.parameters())):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # setup queue
        self.register_buffer('queue', torch.randn(self.K, proj_dim))
        # normalize the queue
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # setup the queue pointer
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(chain(self.encoder_q.parameters(), self.projection_q.parameters()),
                                    chain(self.encoder_k.parameters(), self.projection_k.parameters())):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def _get_mlp(self, inp_dim, hidden_dim, out_dim):
        mlp = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        return mlp

    def _get_shuffle_ids(self, bsz):
        """generate shuffle ids for ShuffleBN"""
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds

    def forward(self, im_q, im_k=None):
        # if only im_q is provided, just return output of encoder_q
        if im_k is None:
            return self.encoder_q(im_q)

        # compute query features
        feat_q = self.projection_q(self.encoder_q(im_q))
        # compute prediction queries
        q = self.predict_q(feat_q)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():
            # update the key encoder
            self._momentum_update_key_encoder()

            # shuffle keys
            shuffle_ids, reverse_ids = self._get_shuffle_ids(im_k.shape[0])
            im_k = im_k[shuffle_ids]

            # forward through the key encoder
            k = self.projection_k(self.encoder_k(im_k))
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = k[reverse_ids]

        # calculate similarities
        queue = self.queue.clone().detach()
        sim_q = torch.mm(q, queue.t())
        sim_k = torch.mm(k, queue.t())

        # scale the similarities with temperature
        sim_q /= self.T_s
        sim_k /= self.T_t

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return sim_q, sim_k
