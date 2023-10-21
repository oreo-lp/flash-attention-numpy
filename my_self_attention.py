import numpy as np

qk = np.array([1.0,2.0,3.0])
v  = np.array([[2.0,3.0,4.0],[1.0,3.0,5.0],[2.0,4.0,6.0]])



def standard_attention():
    max_val = np.amax(qk)
    softmax_qk = np.exp(qk - max_val) / np.sum(np.exp(qk - max_val))
    qkv = np.matmul(softmax_qk, v)
    print("standard attention output = {}".format(qkv))
    
    
def flash_attention():
    pre_max_val = np.amin(qk) - 1
    pre_exp_sum = 0.0
    result = np.zeros(shape=(1,3), dtype=np.float32)
    for idx in range(len(qk)):
        sub_qk = qk[idx]
        sub_v = v[idx,:]
        cur_max_val = max(pre_max_val, sub_qk)
        
        cur_exp_sum = np.sum(np.exp(qk[:(idx+1)] - cur_max_val))
        a = pre_exp_sum / cur_exp_sum * np.exp(pre_max_val - cur_max_val)
        result = a * result + np.exp(sub_qk - cur_max_val)/cur_exp_sum * sub_v
        
        pre_max_val = cur_max_val
        pre_exp_sum = cur_exp_sum
    print(result)
    
if __name__ == "__main__":
    # standard_attention()  # [1.75527153 3.66524096 5.57521038]
    flash_attention()   # [[1.75527153 3.66524096 5.57521038]]