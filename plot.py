from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np
# lines = []
# for i in [2, 20, 100]:
#     data = pd.read_csv(f"data/heads{i}.csv")
#     data = data.iloc[:, 1:2]
#     data = np.array(data)
#     print(data.shape)
#     data = data.reshape(1, len(data)).tolist()
#     print(data[0])
#     line, = plt.plot(list(range(250)), gaussian_filter1d(data[0], sigma=2), label=str(i))
#     lines.append(line)
# plt.legend(['2', '20', '100'])
# plt.title("Uncertainty by epoch")
# plt.savefig(f"data/all_uncertainty")


# disrewards = pd.read_csv(f"data/heads5_disrewards.csv")
# accrewards = disrewards.iloc[:, 2:3]
# accrewards = np.array(accrewards)
# accrewards = accrewards.reshape(1, len(accrewards)).tolist()
# plt.plot(list(range(1000)), accrewards[0])
# plt.savefig(f"data/accdiscount_rewards")

x = []
for i in range(26):
    x.append(40*i)
disrewards = pd.read_csv(f"5x6_mapdata/test/heads5_disrewards_test.csv")
accrewards = disrewards.iloc[:, 1:2]
accrewards = np.array(accrewards)
accrewards = accrewards.reshape(1, len(accrewards)).tolist()
plt.plot(x,accrewards[0], c='g')

disrewards = pd.read_csv(f"5x6_mapdata/random2/heads5_disrewards_random.csv")
accrewards = disrewards.iloc[:, 1:2]
accrewards = np.array(accrewards)
accrewards = accrewards.reshape(1, len(accrewards)).tolist()
plt.plot(x, accrewards[0], c='r')

disrewards = pd.read_csv(f"5x6_mapdata/RCMP/heads5_disrewards_RCMP.csv")
accrewards = disrewards.iloc[:, 1:2]
accrewards = np.array(accrewards)
accrewards = accrewards.reshape(1, len(accrewards)).tolist()
plt.plot(x, accrewards[0], c='y')

disrewards = pd.read_csv(f"5x6_mapdata/IMP/heads5_disrewards_IMP.csv")
accrewards = disrewards.iloc[:, 1:2]
accrewards = np.array(accrewards)
accrewards = accrewards.reshape(1, len(accrewards)).tolist()
plt.plot(x, accrewards[0], c='b')

plt.title("discount rewards")
plt.margins(x=0, y=0)
plt.legend(["no advice", "random", "RCMP", "importance"])
plt.savefig(f"5x6_mapdata/discount_rewards")

# x = []
# for i in range(26):
#     x.append(40*i)
# disrewards = pd.read_csv(f"5x6_mapdata/test/heads5_disrewards_test.csv")
# accrewards = disrewards.iloc[:, 2:3]
# accrewards = np.array(accrewards)
# accrewards = accrewards.reshape(1, len(accrewards)).tolist()
# plt.plot(x,accrewards[0], c='g')

# disrewards = pd.read_csv(f"5x6_mapdata/random2/heads5_disrewards_random.csv")
# accrewards = disrewards.iloc[:, 2:3]
# accrewards = np.array(accrewards)
# accrewards = accrewards.reshape(1, len(accrewards)).tolist()
# plt.plot(x, accrewards[0], c='r')

# disrewards = pd.read_csv(f"5x6_mapdata/RCMP/heads5_disrewards_RCMP.csv")
# accrewards = disrewards.iloc[:, 2:3]
# accrewards = np.array(accrewards)
# accrewards = accrewards.reshape(1, len(accrewards)).tolist()
# plt.plot(x, accrewards[0], c='y')

# disrewards = pd.read_csv(f"5x6_mapdata/IMP/heads5_disrewards_IMP.csv")
# accrewards = disrewards.iloc[:, 2:3]
# accrewards = np.array(accrewards)
# accrewards = accrewards.reshape(1, len(accrewards)).tolist()
# plt.plot(x, accrewards[0], c='b')

# plt.title("accumulate discount rewards")
# plt.margins(x=0, y=0)
# plt.legend(["no advice", "random", "RCMP", "importance"])
# plt.savefig(f"5x6_mapdata/acc_discount_rewards")

