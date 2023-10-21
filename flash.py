import numpy as np

np.random.seed(0)   # 可复现

f = 3
t = 3
h = 2

q = np.random.random(size=(f, h))
k = np.random.random(size=(t, h))
v = np.random.random(size=(t, h))
do = np.random.random(size=(f, h))
head_scale = 1 / np.sqrt(float(h))
dropout_prob = 0.3
dropout_mask = np.random.random(size=(f, t)) >= dropout_prob


def dropout(array, ratio, mask):
    assert (array.shape == mask.shape)
    scale = 1 / (1 - float(ratio))
    array_dp = array * scale
    zero = np.zeros(array.shape, dtype=array.dtype)
    output = np.where(mask, array_dp, zero)
    return output


def flash_attention(q, k, v, is_train=False):
    output = np.zeros(q.shape, dtype=np.float32)
    m = np.zeros(f, dtype=np.float32)   # (16,)
    l = np.zeros(f, dtype=np.float32)   # (16,)

    block_m = 1     # 2
    block_n = 1     # 2
    block_head = h  # 8
    assert (f % block_m == 0)
    assert (t % block_n == 0)
    for start_m in range(0, f, block_m):    # for q
        m_prev = np.zeros([block_m], dtype=np.float32) - float("inf")
        l_prev = np.zeros([block_m], dtype=np.float32)
        acc = np.zeros([block_m, block_head], dtype=np.float32) # (1,2)
        q_sub = q[start_m: start_m + block_m, :]                # (1,2)
        for start_n in range(0, t, block_n):
            k_sub = k[start_n: start_n+block_n, :]              # (1,2)
            v_sub = v[start_n: start_n+block_n, :]              # (1,2)
            dropout_mask_sub = dropout_mask[start_m: start_m +
                                            block_m, start_n: start_n+block_n]
            qk = np.matmul(q_sub, k_sub.T)
            qk *= head_scale
            m_cur = np.maximum(np.amax(qk, -1), m_prev)     # 求x数据的最大值
            l_prev *= np.exp(m_prev - m_cur)                # 乘以系数
            p = np.exp(qk - m_cur.reshape(-1, 1))           # softmax的分子
            l_cur = np.sum(p, -1) + l_prev
            l_rcp = 1 / l_cur
            s = p * l_rcp.reshape(-1, 1)                    # sub_softmax
            acc *= (l_prev * l_rcp).reshape(-1, 1)          # 需要乘以系数，从而更新数据
            # Below commeneted part is from flash attention2
            # s = p
            # acc *= np.exp(m_prev - m_cur).reshape(-1, 1)
            dp_s = dropout(s, dropout_prob, dropout_mask_sub)
            acc += np.matmul(dp_s, v_sub)                   # o
            m_prev = m_cur
            l_prev = l_cur
        # acc /= l_prev.reshape(-1, 1)
        output[start_m: start_m+block_m, :] = acc
        m[start_m: start_m+block_m] = m_prev
        l[start_m: start_m+block_m] = l_prev

    if is_train:
        return output, m, l
    else:
        return output


def naive_attention(q, k, v, is_train=False):
    score = np.matmul(q, k.T)
    score *= head_scale
    row_max = np.amax(score, -1).reshape(-1, 1)
    row_sum = np.sum(np.exp(score - row_max), -1).reshape(-1, 1)
    prob = np.exp(score - row_max) / row_sum
    prob_dp = dropout(prob, dropout_prob, dropout_mask)
    output = np.matmul(prob_dp, v)
    if is_train:
        return output, prob, prob_dp
    else:
        return output


def forward_test(q, k, v):
    desired = naive_attention(q, k, v)
    actual = flash_attention(q, k, v)
    print("desired = {}".format(desired))  # [16,8]
    print("actual = {}".format(actual))
    np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=1e-5)
    
if __name__ == "__main__":
    print("q = {}".format(q))
    print("k = {}".format(k))
    print('v = {}'.format(v))
    print('----------------------')
    forward_test(q,k,v)