from DeepPurpose import utils, dataset, CompoundPred
import warnings

import numpy as np
import os
import sys
import pandas as pd

from .utils import weighted_BH, weighted_CS, eval_sel
from wcs import elond, rlond, ulond, urlond, elond_wcs, ulond_wcs

warnings.filterwarnings("ignore")

X_drugs, y, drugs_index = dataset.load_HIV(path='./data')
drug_encoding = 'Morgan'

seed = int(sys.argv[1])
q = int(sys.argv[2]) / 10

np.random.seed(seed)
n = len(y)
reind = np.random.permutation(n)

X_drugs_train = X_drugs[reind[0:int(n * 0.4 + 1)]]
y_train = y[reind[0:int(n * 0.4 + 1)]]
X_drugs_other = X_drugs[reind[int(1 + n * 0.4):n]]
y_other = y[reind[int(1 + n * 0.4):n]]

# =============================================================================
# # train prediction model on the training data
# =============================================================================

ttrain, tval, ttest = utils.data_process(X_drug=X_drugs_train,
                                         y=y_train,
                                         drug_encoding=drug_encoding,
                                         split_method='random',
                                         frac=[0.7, 0.1, 0.2],
                                         random_seed=seed)

# small neural network
config = utils.generate_config(drug_encoding=drug_encoding,
                               cls_hidden_dims=[1024, 1024, 512],
                               train_epoch=3,
                               LR=0.001,
                               batch_size=128,
                               hidden_dim_drug=128,
                               mpnn_hidden_size=128,
                               mpnn_depth=3)
model = CompoundPred.model_initialize(**config)
model.train(ttrain, tval, ttest)

# =============================================================================
# # weighted split into calibration and test data
# =============================================================================

d_, d__, dother = utils.data_process(X_drug=X_drugs_other,
                                     y=y_other,
                                     drug_encoding=drug_encoding,
                                     split_method='random',
                                     frac=[0, 0, .05],
                                     random_seed=seed)
# print(len(X_drugs_other))
# print(len(y_other))
# print(len(dother))
all_pred = model.predict(dother)
# all_pred = model.predict(X_drugs_other)
train_pred = model.predict(ttrain)

p_x = np.minimum(
    0.8,
    np.exp(all_pred - np.mean(train_pred)) /
    (1 + np.exp(all_pred - np.mean(train_pred))))
in_calib = np.random.binomial(1, p_x, size=len(p_x))

dcalib = dother[pd.Series(in_calib == 1).values].reset_index()
# print(dother)
# print(dcalib)
dtest = dother[pd.Series(in_calib == 0).values].reset_index()

hat_mu_calib = np.array(model.predict(dcalib))
hat_mu_test = np.array(model.predict(dtest))
y_calib = np.array(dcalib["Label"])
w_calib = np.array(1 / p_x[in_calib == 1] - 1)
y_test = np.array(dtest['Label'])
w_test = np.array(1 / p_x[in_calib == 0] - 1)

# =============================================================================
# # run testing procedures
# =============================================================================

c = 0

calib_scores_res = y_calib - hat_mu_calib
# calib_scores_sub = -hat_mu_calib
# calib_scores_clip = 100 * (y_calib > c) + c * (y_calib <= c) - hat_mu_calib

# TODO single test scores is buggy
test_scores = c - hat_mu_test

# =========================
# ## weighted BH procedure
# =========================

# use scores res, sub, and clip
# BH_res = weighted_BH(calib_scores_res, w_calib, test_scores, w_test, q)
# BH_sub = weighted_BH(calib_scores_sub[y_calib <= c], w_calib[y_calib <= c],
#                      test_scores, w_test, q)
# BH_clip = weighted_BH(calib_scores_clip, w_calib, test_scores, w_test, q)
# print('BH done')
# ====================================
# ## weighted conformalized selection
# ====================================

# use scores res, sub, and clip
# WCS_res_0, WCS_res_hete, WCS_res_homo, WCS_res_dtm = weighted_CS(
#     calib_scores_res, w_calib, test_scores, w_test, q)
# print('res0 done')
# WCS_sub_0, WCS_sub_hete, WCS_sub_homo, WCS_sub_dtm = weighted_CS(
#     calib_scores_sub[y_calib <= c], w_calib[y_calib <= c], test_scores, w_test,
#     q)
# print('sub0 done')
# WCS_clip_0, WCS_clip_hete, WCS_clip_homo, WCS_clip_dtm = weighted_CS(
#     calib_scores_clip, w_calib, test_scores, w_test, q)
# print('clip0 done')

s_calib_idxs = np.argsort(calib_scores_res)
data_map = {
    'res': {
        'calib_scores': calib_scores_res,
        'calib_weights': w_calib,
        'test_scores': test_scores,
        'test_weights': w_test,
        's_calib_scores': calib_scores_res[s_calib_idxs],
        's_calib_cum_weights': np.cumsum(w_calib[s_calib_idxs])
    },
    # 'sub': {
    #     'calib_scores': calib_scores_sub[y_calib <= c],
    #     'calib_weights': w_calib[y_calib <= c],
    #     'test_scores': test_scores,
    #     'test_weights': w_test,
    # },
    # 'clip': {
    #     'calib_scores': calib_scores_clip,
    #     'calib_weights': w_calib,
    #     'test_scores': test_scores,
    #     'test_weights': w_test,
    # }
}

algs = [elond, rlond, ulond, urlond]
alg_names = ['e-LOND', 'r-LOND', 'U-LOND', 'Ur-LOND']
omt_results = {
    key: [alg(alpha=q, seed=seed, **value) for alg in algs]
    for key, value in data_map.items()
}
print('omt done')

# =============================================================================
# # summarize FDP, power and selection sizes
# =============================================================================
# BH_res_fdp, BH_res_power = eval_sel(BH_res, y_test,
#                                     np.array([c] * len(y_test)))
# BH_sub_fdp, BH_sub_power = eval_sel(BH_sub, y_test,
#                                     np.array([c] * len(y_test)))
# BH_clip_fdp, BH_clip_power = eval_sel(BH_clip, y_test,
#                                       np.array([c] * len(y_test)))
#
# all_BH = [BH_res, BH_sub, BH_clip]
# all_sel = [[WCS_res_hete, WCS_res_homo, WCS_res_dtm] + omt_results['res'],
#            [WCS_sub_hete, WCS_sub_homo, WCS_sub_dtm] + omt_results['sub'],
#            [WCS_clip_hete, WCS_clip_homo, WCS_clip_dtm] + omt_results['clip']]
all_sel = [omt_results['res']]  #, omt_results['sub'], omt_results['clip']]
# fdp = [BH_res_fdp, BH_sub_fdp, BH_clip_fdp]
# power = [BH_res_power, BH_sub_power, BH_clip_power]
# ndiff = [0] * 3
# nsel = [len(BH_res), len(BH_sub), len(BH_clip)]
# nsame = [len(BH_res), len(BH_sub), len(BH_clip)]
fdp, power, nsel = [], [], []

rows = []
for ii in range(len(all_sel)):
    sels = all_sel[ii]
    tpowers = []
    tfdps = []
    tnsels = []
    # tndiffs = []
    # tnsames = []
    for jj in range(len(algs)):
        tfdp, tpower = eval_sel(sels[jj], y_test, np.array([c] * len(y_test)))
        tpowers.append(tpower)
        tfdps.append(tfdp)
        tnsels.append(len(sels[jj]))
        # tndiffs.append(len(np.setxor1d(all_BH[ii], sels[jj])))
        # tnsames.append(len(np.intersect1d(all_BH[ii], sels[jj])))
        # rows.append({
        #     'FDP': tfdp,
        #     'power': tpower,
        #     'nsel': len(sels[jj]),
        #     'score': 'res',
        #     'method': alg_names[ii]
        # })
    fdp += tfdps
    power += tpowers
    # ndiff += tndiffs
    nsel += tnsels
    # nsame += tnsames

# method_ct = 3 + len(algs)
method_ct = len(algs)
res = pd.DataFrame({
    "FDP": fdp,
    "power": power,
    "nsel": nsel,
    # "ndiff":
    # ndiff,
    # "nsame":
    # nsame,
    # "score": ["res", "sub", "clip"] + ["res"] * method_ct +
    # ["sub"] * method_ct + ["clip"] * method_ct,
    "score":
    ["res"] * method_ct,  #+ ["sub"] * method_ct + ["clip"] * method_ct,
    #  "method": ["WBH"] * 3 + (['WCS.hete', 'WCS.homo', "WCS.dtm"] + algs) * 3,
    # "method": ["WBH"] * 3 + algs * 3,
    "method": alg_names,
    "q": q,
    "T": len(y_test),
    "seed": seed
})
save_path = "./DPP_results"
if not os.path.exists(save_path):
    os.makedirs(save_path)
res.to_csv("./DPP_results/seed" + str(seed) + "q" + str(q) + ".csv")
print(res)
