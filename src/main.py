from loader import DataLoader
from helper import *

dl = DataLoader()

m1, m2 = train_stage_0(dl)
mf = train_stage_1(dl, [m1, m2])
mf_final = train_stage_2(dl, mf)
