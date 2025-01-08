class CombinedSparseAdaptiveAttention(nn.Module):
    def __init__(self, n_state, n_head, max_rel_dist, base, sparsity_factor, max_span):
        super().__init__()
        self.n_head = n_head
        self.multihead_attn = MultiheadAttention(n_state, n_head, max_rel_dist, base)
        self.sparsity_factor = sparsity_factor
        self.max_span = max_span
        self.span_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, key, value):
        assert query.dim() == 2 or query.dim() == 3, "query should be unbatched 2D or batched 3D tensor but received {}-D tensor".format(query.dim())
        if query.dim() == 4:
            query = query.view(query.shape[0] * query.shape[1], query.shape[2], query.shape[3])  # Adjust this based on your requirements

        batch_size, seq_len, n_state = query.size()

        # Sparse Attention
        k = max(1, int(seq_len * self.sparsity_factor))  # Ensure k is at least 1
        indices = torch.topk(query.norm(dim=-1), k, dim=1).indices
        query_sparse = query.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))
        key_sparse = key.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))
        value_sparse = value.gather(1, indices.unsqueeze(-1).expand(-1, -1, n_state))

        if query_sparse.shape[1] > 0 and key_sparse.shape[1] > 0 and value_sparse.shape[1] > 0:
            query_sparse = query_sparse.view(query_sparse.shape[0], query_sparse.shape[1], self.n_head, -1)
            key_sparse = key_sparse.view(key_sparse.shape[0], key_sparse.shape[1], self.n_head, -1)
            value_sparse = value_sparse.view(value_sparse.shape[0], value_sparse.shape[1], self.n_head, -1)

        # Adaptive Span Attention
        span_length = int(self.max_span * self.span_scale.item())
        span_length = min(span_length, query.shape[1])
        query_span = query_sparse[:, :span_length, :]
        key_span = key_sparse[:, :span_length, :]
        value_span = value_sparse[:, :span_length, :]

        # Combined Attention
        attn_output, attn_weights = self.multihead_attn(query_span, key_span, value_span)
        return attn_output, attn_weights
