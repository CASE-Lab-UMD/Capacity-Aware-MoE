# 参考： https://blog.csdn.net/weixin_26704853/article/details/108495242

import numpy as np
import ot
# ot.gromov

data = np.load('manhattan.npz') # https://github.com/PythonOT/POT/raw/master/data/
bakery_pos = data['bakery_pos']
bakery_prod = data['bakery_prod']
cafe_pos = data['cafe_pos']
cafe_prod = data['cafe_prod']
Imap = data['Imap']
print('Bakery production: {}'.format(bakery_prod))
print('Cafe sale: {}'.format(cafe_prod))
print('Total croissants : {}'.format(cafe_prod.sum()))

# 计算面包店和咖啡馆之间的成本矩阵，它将是运输成本矩阵 。 可以使用ot.dist函数完成此操作，该函数默认为平方的欧几里得距离，但可以返回诸如cityblock(或Manhattan距离)之类的其他信息。
M = ot.dist(bakery_pos, cafe_pos)
print(M.shape)

# 得到运输矩阵
gamma_emd = ot.emd(bakery_prod, cafe_prod, M)
print(gamma_emd)

# 运输成本
print(np.multiply(M, gamma_emd).sum())

# 运输过程：bakery -> caffe
gamma_emd_norm = gamma_emd/np.sum(gamma_emd, axis=1, keepdims=True)
caffe_prod_trans = np.dot(bakery_prod, gamma_emd_norm)
print(caffe_prod_trans-cafe_prod)