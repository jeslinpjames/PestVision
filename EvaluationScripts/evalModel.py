

# import os
# from ultralytics import YOLO

# # Set the environment variable
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# def evaluate_model_performance(model_path, data_path):
#     # Load the trained model
#     model = YOLO(model_path)

#     # Evaluate the model on the evaluation data
#     results = model.val(data=data_path)

#     # Print the results
#     print("Evaluation Results:")
#     print(results)

# if __name__ == "__main__":
#     # Define model path and evaluation data path
#     model_path = "D:\git\PestVisionChallenge\yolov4-pytorch.pt"
#     data_path = "config2.yaml"

#     # Evaluate the model performance
#     evaluate_model_performance(model_path, data_path)



# """"
# model 1
# "C:/Users/james/Downloads/best.pt"
#  New cache created: D:\git\Data\PestVisionEvalImages\Data\val\labels.cache
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 84/84 [28:39<00:00
#                    all       1337       3605       0.15       0.12     0.0425     0.0121
# Speed: 2.0ms preprocess, 1146.2ms inference, 0.0ms loss, 3.9ms postprocess per image
# Results saved to runs\detect\val2
# Evaluation Results:
# ultralytics.utils.metrics.DetMetrics object with attributes:

# ap_class_index: array([0])
# box: ultralytics.utils.metrics.Metric object
# confusion_matrix: <ultralytics.utils.metrics.ConfusionMatrix object at 0x0000026BE4F4A110>
# curves: ['Precision-Recall(B)', 'F1-Confidence(B)', 'Precision-Confidence(B)', 'Recall-Confidence(B)']
# curves_results: [[array([          0,    0.001001,    0.002002,    0.003003,    0.004004,    0.005005,    0.006006,    0.007007,    0.008008,    0.009009,     0.01001,    0.011011,    0.012012,    0.013013,    0.014014,    0.015015,    0.016016,    0.017017,    0.018018,    0.019019,     0.02002,    0.021021,    0.022022,    0.023023,
#           0.024024,    0.025025,    0.026026,    0.027027,    0.028028,    0.029029,     0.03003,    0.031031,    0.032032,    0.033033,    0.034034,    0.035035,    0.036036,    0.037037,    0.038038,    0.039039,     0.04004,    0.041041,    0.042042,    0.043043,    0.044044,    0.045045,    0.046046,    0.047047,
#           0.048048,    0.049049,     0.05005,    0.051051,    0.052052,    0.053053,    0.054054,    0.055055,    0.056056,    0.057057,    0.058058,    0.059059,     0.06006,    0.061061,    0.062062,    0.063063,    0.064064,    0.065065,    0.066066,    0.067067,    0.068068,    0.069069,     0.07007,    0.071071,
#           0.072072,    0.073073,    0.074074,    0.075075,    0.076076,    0.077077,    0.078078,    0.079079,     0.08008,    0.081081,    0.082082,    0.083083,    0.084084,    0.085085,    0.086086,    0.087087,    0.088088,    0.089089,     0.09009,    0.091091,    0.092092,    0.093093,    0.094094,    0.095095,
#           0.096096,    0.097097,    0.098098,    0.099099,      0.1001,      0.1011,      0.1021,      0.1031,      0.1041,     0.10511,     0.10611,     0.10711,     0.10811,     0.10911,     0.11011,     0.11111,     0.11211,     0.11311,     0.11411,     0.11512,     0.11612,     0.11712,     0.11812,     0.11912,
#            0.12012,     0.12112,     0.12212,     0.12312,     0.12412,     0.12513,     0.12613,     0.12713,     0.12813,     0.12913,     0.13013,     0.13113,     0.13213,     0.13313,     0.13413,     0.13514,     0.13614,     0.13714,     0.13814,     0.13914,     0.14014,     0.14114,     0.14214,     0.14314,
#            0.14414,     0.14515,     0.14615,     0.14715,     0.14815,     0.14915,     0.15015,     0.15115,     0.15215,     0.15315,     0.15415,     0.15516,     0.15616,     0.15716,     0.15816,     0.15916,     0.16016,     0.16116,     0.16216,     0.16316,     0.16416,     0.16517,     0.16617,     0.16717,
#            0.16817,     0.16917,     0.17017,     0.17117,     0.17217,     0.17317,     0.17417,     0.17518,     0.17618,     0.17718,     0.17818,     0.17918,     0.18018,     0.18118,     0.18218,     0.18318,     0.18418,     0.18519,     0.18619,     0.18719,     0.18819,     0.18919,     0.19019,     0.19119,
#            0.19219,     0.19319,     0.19419,      0.1952,      0.1962,      0.1972,      0.1982,      0.1992,      0.2002,      0.2012,      0.2022,      0.2032,      0.2042,     0.20521,     0.20621,     0.20721,     0.20821,     0.20921,     0.21021,     0.21121,     0.21221,     0.21321,     0.21421,     0.21522,
#            0.21622,     0.21722,     0.21822,     0.21922,     0.22022,     0.22122,     0.22222,     0.22322,     0.22422,     0.22523,     0.22623,     0.22723,     0.22823,     0.22923,     0.23023,     0.23123,     0.23223,     0.23323,     0.23423,     0.23524,     0.23624,     0.23724,     0.23824,     0.23924,
#            0.24024,     0.24124,     0.24224,     0.24324,     0.24424,     0.24525,     0.24625,     0.24725,     0.24825,     0.24925,     0.25025,     0.25125,     0.25225,     0.25325,     0.25425,     0.25526,     0.25626,     0.25726,     0.25826,     0.25926,     0.26026,     0.26126,     0.26226,     0.26326,
#            0.26426,     0.26527,     0.26627,     0.26727,     0.26827,     0.26927,     0.27027,     0.27127,     0.27227,     0.27327,     0.27427,     0.27528,     0.27628,     0.27728,     0.27828,     0.27928,     0.28028,     0.28128,     0.28228,     0.28328,     0.28428,     0.28529,     0.28629,     0.28729,
#            0.28829,     0.28929,     0.29029,     0.29129,     0.29229,     0.29329,     0.29429,      0.2953,      0.2963,      0.2973,      0.2983,      0.2993,      0.3003,      0.3013,      0.3023,      0.3033,      0.3043,     0.30531,     0.30631,     0.30731,     0.30831,     0.30931,     0.31031,     0.31131,
#            0.31231,     0.31331,     0.31431,     0.31532,     0.31632,     0.31732,     0.31832,     0.31932,     0.32032,     0.32132,     0.32232,     0.32332,     0.32432,     0.32533,     0.32633,     0.32733,     0.32833,     0.32933,     0.33033,     0.33133,     0.33233,     0.33333,     0.33433,     0.33534,
#            0.33634,     0.33734,     0.33834,     0.33934,     0.34034,     0.34134,     0.34234,     0.34334,     0.34434,     0.34535,     0.34635,     0.34735,     0.34835,     0.34935,     0.35035,     0.35135,     0.35235,     0.35335,     0.35435,     0.35536,     0.35636,     0.35736,     0.35836,     0.35936,
#            0.36036,     0.36136,     0.36236,     0.36336,     0.36436,     0.36537,     0.36637,     0.36737,     0.36837,     0.36937,     0.37037,     0.37137,     0.37237,     0.37337,     0.37437,     0.37538,     0.37638,     0.37738,     0.37838,     0.37938,     0.38038,     0.38138,     0.38238,     0.38338,
#            0.38438,     0.38539,     0.38639,     0.38739,     0.38839,     0.38939,     0.39039,     0.39139,     0.39239,     0.39339,     0.39439,      0.3954,      0.3964,      0.3974,      0.3984,      0.3994,      0.4004,      0.4014,      0.4024,      0.4034,      0.4044,     0.40541,     0.40641,     0.40741,
#            0.40841,     0.40941,     0.41041,     0.41141,     0.41241,     0.41341,     0.41441,     0.41542,     0.41642,     0.41742,     0.41842,     0.41942,     0.42042,     0.42142,     0.42242,     0.42342,     0.42442,     0.42543,     0.42643,     0.42743,     0.42843,     0.42943,     0.43043,     0.43143,
#            0.43243,     0.43343,     0.43443,     0.43544,     0.43644,     0.43744,     0.43844,     0.43944,     0.44044,     0.44144,     0.44244,     0.44344,     0.44444,     0.44545,     0.44645,     0.44745,     0.44845,     0.44945,     0.45045,     0.45145,     0.45245,     0.45345,     0.45445,     0.45546,
#            0.45646,     0.45746,     0.45846,     0.45946,     0.46046,     0.46146,     0.46246,     0.46346,     0.46446,     0.46547,     0.46647,     0.46747,     0.46847,     0.46947,     0.47047,     0.47147,     0.47247,     0.47347,     0.47447,     0.47548,     0.47648,     0.47748,     0.47848,     0.47948,
#            0.48048,     0.48148,     0.48248,     0.48348,     0.48448,     0.48549,     0.48649,     0.48749,     0.48849,     0.48949,     0.49049,     0.49149,     0.49249,     0.49349,     0.49449,      0.4955,      0.4965,      0.4975,      0.4985,      0.4995,      0.5005,      0.5015,      0.5025,      0.5035,
#             0.5045,     0.50551,     0.50651,     0.50751,     0.50851,     0.50951,     0.51051,     0.51151,     0.51251,     0.51351,     0.51451,     0.51552,     0.51652,     0.51752,     0.51852,     0.51952,     0.52052,     0.52152,     0.52252,     0.52352,     0.52452,     0.52553,     0.52653,     0.52753,
#            0.52853,     0.52953,     0.53053,     0.53153,     0.53253,     0.53353,     0.53453,     0.53554,     0.53654,     0.53754,     0.53854,     0.53954,     0.54054,     0.54154,     0.54254,     0.54354,     0.54454,     0.54555,     0.54655,     0.54755,     0.54855,     0.54955,     0.55055,     0.55155,
#            0.55255,     0.55355,     0.55455,     0.55556,     0.55656,     0.55756,     0.55856,     0.55956,     0.56056,     0.56156,     0.56256,     0.56356,     0.56456,     0.56557,     0.56657,     0.56757,     0.56857,     0.56957,     0.57057,     0.57157,     0.57257,     0.57357,     0.57457,     0.57558,
#            0.57658,     0.57758,     0.57858,     0.57958,     0.58058,     0.58158,     0.58258,     0.58358,     0.58458,     0.58559,     0.58659,     0.58759,     0.58859,     0.58959,     0.59059,     0.59159,     0.59259,     0.59359,     0.59459,      0.5956,      0.5966,      0.5976,      0.5986,      0.5996,
#             0.6006,      0.6016,      0.6026,      0.6036,      0.6046,     0.60561,     0.60661,     0.60761,     0.60861,     0.60961,     0.61061,     0.61161,     0.61261,     0.61361,     0.61461,     0.61562,     0.61662,     0.61762,     0.61862,     0.61962,     0.62062,     0.62162,     0.62262,     0.62362,
#            0.62462,     0.62563,     0.62663,     0.62763,     0.62863,     0.62963,     0.63063,     0.63163,     0.63263,     0.63363,     0.63463,     0.63564,     0.63664,     0.63764,     0.63864,     0.63964,     0.64064,     0.64164,     0.64264,     0.64364,     0.64464,     0.64565,     0.64665,     0.64765,
#            0.64865,     0.64965,     0.65065,     0.65165,     0.65265,     0.65365,     0.65465,     0.65566,     0.65666,     0.65766,     0.65866,     0.65966,     0.66066,     0.66166,     0.66266,     0.66366,     0.66466,     0.66567,     0.66667,     0.66767,     0.66867,     0.66967,     0.67067,     0.67167,
#            0.67267,     0.67367,     0.67467,     0.67568,     0.67668,     0.67768,     0.67868,     0.67968,     0.68068,     0.68168,     0.68268,     0.68368,     0.68468,     0.68569,     0.68669,     0.68769,     0.68869,     0.68969,     0.69069,     0.69169,     0.69269,     0.69369,     0.69469,      0.6957,
#             0.6967,      0.6977,      0.6987,      0.6997,      0.7007,      0.7017,      0.7027,      0.7037,      0.7047,     0.70571,     0.70671,     0.70771,     0.70871,     0.70971,     0.71071,     0.71171,     0.71271,     0.71371,     0.71471,     0.71572,     0.71672,     0.71772,     0.71872,     0.71972,
#            0.72072,     0.72172,     0.72272,     0.72372,     0.72472,     0.72573,     0.72673,     0.72773,     0.72873,     0.72973,     0.73073,     0.73173,     0.73273,     0.73373,     0.73473,     0.73574,     0.73674,     0.73774,     0.73874,     0.73974,     0.74074,     0.74174,     0.74274,     0.74374,
#            0.74474,     0.74575,     0.74675,     0.74775,     0.74875,     0.74975,     0.75075,     0.75175,     0.75275,     0.75375,     0.75475,     0.75576,     0.75676,     0.75776,     0.75876,     0.75976,     0.76076,     0.76176,     0.76276,     0.76376,     0.76476,     0.76577,     0.76677,     0.76777,
#            0.76877,     0.76977,     0.77077,     0.77177,     0.77277,     0.77377,     0.77477,     0.77578,     0.77678,     0.77778,     0.77878,     0.77978,     0.78078,     0.78178,     0.78278,     0.78378,     0.78478,     0.78579,     0.78679,     0.78779,     0.78879,     0.78979,     0.79079,     0.79179,
#            0.79279,     0.79379,     0.79479,      0.7958,      0.7968,      0.7978,      0.7988,      0.7998,      0.8008,      0.8018,      0.8028,      0.8038,      0.8048,     0.80581,     0.80681,     0.80781,     0.80881,     0.80981,     0.81081,     0.81181,     0.81281,     0.81381,     0.81481,     0.81582,
#            0.81682,     0.81782,     0.81882,     0.81982,     0.82082,     0.82182,     0.82282,     0.82382,     0.82482,     0.82583,     0.82683,     0.82783,     0.82883,     0.82983,     0.83083,     0.83183,     0.83283,     0.83383,     0.83483,     0.83584,     0.83684,     0.83784,     0.83884,     0.83984,
#            0.84084,     0.84184,     0.84284,     0.84384,     0.84484,     0.84585,     0.84685,     0.84785,     0.84885,     0.84985,     0.85085,     0.85185,     0.85285,     0.85385,     0.85485,     0.85586,     0.85686,     0.85786,     0.85886,     0.85986,     0.86086,     0.86186,     0.86286,     0.86386,
#            0.86486,     0.86587,     0.86687,     0.86787,     0.86887,     0.86987,     0.87087,     0.87187,     0.87287,     0.87387,     0.87487,     0.87588,     0.87688,     0.87788,     0.87888,     0.87988,     0.88088,     0.88188,     0.88288,     0.88388,     0.88488,     0.88589,     0.88689,     0.88789,
#            0.88889,     0.88989,     0.89089,     0.89189,     0.89289,     0.89389,     0.89489,      0.8959,      0.8969,      0.8979,      0.8989,      0.8999,      0.9009,      0.9019,      0.9029,      0.9039,      0.9049,     0.90591,     0.90691,     0.90791,     0.90891,     0.90991,     0.91091,     0.91191,
#            0.91291,     0.91391,     0.91491,     0.91592,     0.91692,     0.91792,     0.91892,     0.91992,     0.92092,     0.92192,     0.92292,     0.92392,     0.92492,     0.92593,     0.92693,     0.92793,     0.92893,     0.92993,     0.93093,     0.93193,     0.93293,     0.93393,     0.93493,     0.93594,
#            0.93694,     0.93794,     0.93894,     0.93994,     0.94094,     0.94194,     0.94294,     0.94394,     0.94494,     0.94595,     0.94695,     0.94795,     0.94895,     0.94995,     0.95095,     0.95195,     0.95295,     0.95395,     0.95495,     0.95596,     0.95696,     0.95796,     0.95896,     0.95996,
#            0.96096,     0.96196,     0.96296,     0.96396,     0.96496,     0.96597,     0.96697,     0.96797,     0.96897,     0.96997,     0.97097,     0.97197,     0.97297,     0.97397,     0.97497,     0.97598,     0.97698,     0.97798,     0.97898,     0.97998,     0.98098,     0.98198,     0.98298,     0.98398,
#            0.98498,     0


           
# model 2
# "C:\Users\james\Downloads\train4\weights\best.pt"

# :\git\Data\PestVisionEvalImages\Data\val\images\fbab8fee-900e-4975-899d-630bb1c85e0a.jpg: corrupt JPEG restoreee-900e-4975-899d-630bb1c85e0a.jpg: corrupt JPEG restored and saved
#                  Class     Images  Instances      Box(P          R                      Class     Images  Instances      Box(P          R                      Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  80%|███████▉  | 67/                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):  90%|████████                 Class     Images  Instances      Box(P          R                      Class     Images  Instances      Box(P          R                      Class     Images  Instances      Box(P          R                      Class     Images  Instances      Box(P                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 84/84 [34:47<00:00
#                    all       1337       3605      0.186      0.133     0.0652     0.0212
# Speed: 2.5ms preprocess, 1391.5ms inference, 0.0ms loss, 3.6ms postprocess per image
# Results saved to runs\detect\val3
# Evaluation Results:
# ultralytics.utils.metrics.DetMetrics object with attributes:

# ap_class_index: array([0])
# box: ultralytics.utils.metrics.Metric object
# confusion_matrix: <ultralytics.utils.metrics.ConfusionMatrix object at 0x000002A33FB80D90>
# curves: ['Precision-Recall(B)', 'F1-Confidence(B)', 'Precision-Confidence(B)', 'Recall-Confidence(B)']
# curves_results: [[array([          0,    0.001001,    0.002002,    0.003003,    0.004004,    0.005005,    0.006006,    0.007007,    0.008008,    0.009009,     0.01001,    0.011011,    0.012012,    0.013013,    0.014014,    0.015015,    0.016016,    0.017017,    0.018018,    0.019019,     0.02002,    0.021021,    0.022022,    0.023023,
#           0.024024,    0.025025,    0.026026,    0.027027,    0.028028,    0.029029,     0.03003,    0.031031,    0.032032,    0.033033,    0.034034,    0.035035,    0.036036,    0.037037,    0.038038,    0.039039,     0.04004,    0.041041,    0.042042,    0.043043,    0.044044,    0.045045,    0.046046,    0.047047,
#           0.048048,    0.049049,     0.05005,    0.051051,    0.052052,    0.053053,    0.054054,    0.055055,    0.056056,    0.057057,    0.058058,    0.059059,     0.06006,    0.061061,    0.062062,    0.063063,    0.064064,    0.065065,    0.066066,    0.067067,    0.068068,    0.069069,     0.07007,    0.071071,
#           0.072072,    0.073073,    0.074074,    0.075075,    0.076076,    0.077077,    0.078078,    0.079079,     0.08008,    0.081081,    0.082082,    0.083083,    0.084084,    0.085085,    0.086086,    0.087087,    0.088088,    0.089089,     0.09009,    0.091091,    0.092092,    0.093093,    0.094094,    0.095095,
#           0.096096,    0.097097,    0.098098,    0.099099,      0.1001,      0.1011,      0.1021,      0.1031,      0.1041,     0.10511,     0.10611,     0.10711,     0.10811,     0.10911,     0.11011,     0.11111,     0.11211,     0.11311,     0.11411,     0.11512,     0.11612,     0.11712,     0.11812,     0.11912,
#            0.12012,     0.12112,     0.12212,     0.12312,     0.12412,     0.12513,     0.12613,     0.12713,     0.12813,     0.12913,     0.13013,     0.13113,     0.13213,     0.13313,     0.13413,     0.13514,     0.13614,     0.13714,     0.13814,     0.13914,     0.14014,     0.14114,     0.14214,     0.14314,
#            0.14414,     0.14515,     0.14615,     0.14715,     0.14815,     0.14915,     0.15015,     0.15115,     0.15215,     0.15315,     0.15415,     0.15516,     0.15616,     0.15716,     0.15816,     0.15916,     0.16016,     0.16116,     0.16216,     0.16316,     0.16416,     0.16517,     0.16617,     0.16717,
#            0.16817,     0.16917,     0.17017,     0.17117,     0.17217,     0.17317,     0.17417,     0.17518,     0.17618,     0.17718,     0.17818,     0.17918,     0.18018,     0.18118,     0.18218,     0.18318,     0.18418,     0.18519,     0.18619,     0.18719,     0.18819,     0.18919,     0.19019,     0.19119,
#            0.19219,     0.19319,     0.19419,      0.1952,      0.1962,      0.1972,      0.1982,      0.1992,      0.2002,      0.2012,      0.2022,      0.2032,      0.2042,     0.20521,     0.20621,     0.20721,     0.20821,     0.20921,     0.21021,     0.21121,     0.21221,     0.21321,     0.21421,     0.21522,
#            0.21622,     0.21722,     0.21822,     0.21922,     0.22022,     0.22122,     0.22222,     0.22322,     0.22422,     0.22523,     0.22623,     0.22723,     0.22823,     0.22923,     0.23023,     0.23123,     0.23223,     0.23323,     0.23423,     0.23524,     0.23624,     0.23724,     0.23824,     0.23924,
#            0.24024,     0.24124,     0.24224,     0.24324,     0.24424,     0.24525,     0.24625,     0.24725,     0.24825,     0.24925,     0.25025,     0.25125,     0.25225,     0.25325,     0.25425,     0.25526,     0.25626,     0.25726,     0.25826,     0.25926,     0.26026,     0.26126,     0.26226,     0.26326,
#            0.26426,     0.26527,     0.26627,     0.26727,     0.26827,     0.26927,     0.27027,     0.27127,     0.27227,     0.27327,     0.27427,     0.27528,     0.27628,     0.27728,     0.27828,     0.27928,     0.28028,     0.28128,     0.28228,     0.28328,     0.28428,     0.28529,     0.28629,     0.28729,
#            0.28829,     0.28929,     0.29029,     0.29129,     0.29229,     0.29329,     0.29429,      0.2953,      0.2963,      0.2973,      0.2983,      0.2993,      0.3003,      0.3013,      0.3023,      0.3033,      0.3043,     0.30531,     0.30631,     0.30731,     0.30831,     0.30931,     0.31031,     0.31131,
#            0.31231,     0.31331,     0.31431,     0.31532,     0.31632,     0.31732,     0.31832,     0.31932,     0.32032,     0.32132,     0.32232,     0.32332,     0.32432,     0.32533,     0.32633,     0.32733,     0.32833,     0.32933,     0.33033,     0.33133,     0.33233,     0.33333,     0.33433,     0.33534,
#            0.33634,     0.33734,     0.33834,     0.33934,     0.34034,     0.34134,     0.34234,     0.34334,     0.34434,     0.34535,     0.34635,     0.34735,     0.34835,     0.34935,     0.35035,     0.35135,     0.35235,     0.35335,     0.35435,     0.35536,     0.35636,     0.35736,     0.35836,     0.35936,
#            0.36036,     0.36136,     0.36236,     0.36336,     0.36436,     0.36537,     0.36637,     0.36737,     0.36837,     0.36937,     0.37037,     0.37137,     0.37237,     0.37337,     0.37437,     0.37538,     0.37638,     0.37738,     0.37838,     0.37938,     0.38038,     0.38138,     0.38238,     0.38338,
#            0.38438,     0.38539,     0.38639,     0.38739,     0.38839,     0.38939,     0.39039,     0.39139,     0.39239,     0.39339,     0.39439,      0.3954,      0.3964,      0.3974,      0.3984,      0.3994,      0.4004,      0.4014,      0.4024,      0.4034,      0.4044,     0.40541,     0.40641,     0.40741,
#            0.40841,     0.40941,     0.41041,     0.41141,     0.41241,     0.41341,     0.41441,     0.41542,     0.41642,     0.41742,     0.41842,     0.41942,     0.42042,     0.42142,     0.42242,     0.42342,     0.42442,     0.42543,     0.42643,     0.42743,     0.42843,     0.42943,     0.43043,     0.43143,
#            0.43243,     0.43343,     0.43443,     0.43544,     0.43644,     0.43744,     0.43844,     0.43944,     0.44044,     0.44144,     0.44244,     0.44344,     0.44444,     0.44545,     0.44645,     0.44745,     0.44845,     0.44945,     0.45045,     0.45145,     0.45245,     0.45345,     0.45445,     0.45546,
#            0.45646,     0.45746,     0.45846,     0.45946,     0.46046,     0.46146,     0.46246,     0.46346,     0.46446,     0.46547,     0.46647,     0.46747,     0.46847,     0.46947,     0.47047,     0.47147,     0.47247,     0.47347,     0.47447,     0.47548,     0.47648,     0.47748,     0.47848,     0.47948,
#            0.48048,     0.48148,     0.48248,     0.48348,     0.48448,     0.48549,     0.48649,     0.48749,     0.48849,     0.48949,     0.49049,     0.49149,     0.49249,     0.49349,     0.49449,      0.4955,      0.4965,      0.4975,      0.4985,      0.4995,      0.5005,      0.5015,      0.5025,      0.5035,
#             0.5045,     0.50551,     0.50651,     0.50751,     0.50851,     0.50951,     0.51051,     0.51151,     0.51251,     0.51351,     0.51451,     0.51552,     0.51652,     0.51752,     0.51852,     0.51952,     0.52052,     0.52152,     0.52252,     0.52352,     0.52452,     0.52553,     0.52653,     0.52753,
#            0.52853,     0.52953,     0.53053,     0.53153,     0.53253,     0.53353,     0.53453,     0.53554,     0.53654,     0.53754,     0.53854,     0.53954,     0.54054,     0.54154,     0.54254,     0.54354,     0.54454,     0.54555,     0.54655,     0.54755,     0.54855,     0.54955,     0.55055,     0.55155,
#            0.55255,     0.55355,     0.55455,     0.55556,     0.55656,     0.55756,     0.55856,     0.55956,     0.56056,     0.56156,     0.56256,     0.56356,     0.56456,     0.56557,     0.56657,     0.56757,     0.56857,     0.56957,     0.57057,     0.57157,     0.57257,     0.57357,     0.57457,     0.57558,
#            0.57658,     0.57758,     0.57858,     0.57958,     0.58058,     0.58158,     0.58258,     0.58358,     0.58458,     0.58559,     0.58659,     0.58759,     0.58859,     0.58959,     0.59059,     0.59159,     0.59259,     0.59359,     0.59459,      0.5956,      0.5966,      0.5976,      0.5986,      0.5996,
#             0.6006,      0.6016,      0.6026,      0.6036,      0.6046,     0.60561,     0.60661,     0.60761,     0.60861,     0.60961,     0.61061,     0.61161,     0.61261,     0.61361,     0.61461,     0.61562,     0.61662,     0.61762,     0.61862,     0.61962,     0.62062,     0.62162,     0.62262,     0.62362,
#            0.62462,     0.62563,     0.62663,     0.62763,     0.62863,     0.62963,     0.63063,     0.63163,     0.63263,     0.63363,     0.63463,     0.63564,     0.63664,     0.63764,     0.63864,     0.63964,     0.64064,     0.64164,     0.64264,     0.64364,     0.64464,     0.64565,     0.64665,     0.64765,
#            0.64865,     0.64965,     0.65065,     0.65165,     0.65265,     0.65365,     0.65465,     0.65566,     0.65666,     0.65766,     0.65866,     0.65966,     0.66066,     0.66166,     0.66266,     0.66366,     0.66466,     0.66567,     0.66667,     0.66767,     0.66867,     0.66967,     0.67067,     0.67167,
#            0.67267,     0.67367,     0.67467,     0.67568,     0.67668,     0.67768,     0.67868,     0.67968,     0.68068,     0.68168,     0.68268,     0.68368,     0.68468,     0.68569,     0.68669,     0.68769,     0.68869,     0.68969,     0.69069,     0.69169,     0.69269,     0.69369,     0.69469,      0.6957,
#             0.6967,      0.6977,      0.6987,      0.6997,      0.7007,      0.7017,      0.7027,      0.7037,      0.7047,     0.70571,     0.70671,     0.70771,     0.70871,     0.70971,     0.71071,     0.71171,     0.71271,     0.71371,     0.71471,     0.71572,     0.71672,     0.71772,     0.71872,     0.71972,
#            0.72072,     0.72172,     0.72272,     0.72372,     0.72472,     0.72573,     0.72673,     0.72773,     0.72873,     0.72973,     0.73073,     0.73173,     0.73273,     0.73373,     0.73473,     0.73574,     0.73674,     0.73774,     0.73874,     0.73974,     0.74074,     0.74174,     0.74274,     0.74374,
#            0.74474,     0.74575,     0.74675,     0.74775,     0.74875,     0.74975,     0.75075,     0.75175,     0.75275,     0.75375,     0.75475,     0.75576,     0.75676,     0.75776,     0.75876,     0.75976,     0.76076,     0.76176,     0.76276,     0.76376,     0.76476,     0.76577,     0.76677,     0.76777,
#            0.76877,     0.76977,     0.77077,     0.77177,     0.77277,     0.77377,     0.77477,     0.77578,     0.77678,     0.77778,     0.77878,     0.77978,     0.78078,     0.78178,     0.78278,     0.78378,     0.78478,     0.78579,     0.78679,     0.78779,     0.78879,     0.78979,     0.79079,     0.79179,
#            0.79279,     0.79379,     0.79479,      0.7958,      0.7968,      0.7978,      0.7988,      0.7998,      0.8008,      0.8018,      0.8028,      0.8038,      0.8048,     0.80581,     0.80681,     0.80781,     0.80881,     0.80981,     0.81081,     0.81181,     0.81281,     0.81381,     0.81481,     0.81582,
#            0.81682,     0.81782,     0.81882,     0.81982,     0.82082,     0.82182,     0.82282,     0.82382,     0.82482,     0.82583,     0.82683,     0.82783,     0.82883,     0.82983,     0.83083,     0.83183,     0.83283,     0.83383,     0.83483,     0.83584,     0.83684,     0.83784,     0.83884,     0.83984,
#            0.84084,     0.84184,     0.84284,     0.84384,     0.84484,     0.84585,     0.84685,     0.84785,     0.84885,     0.84985,     0.85085,     0.85185,     0.85285,     0.85385,     0.85485,     0.85586,     0.85686,     0.85786,     0.85886,     0.85986,     0.86086,     0.86186,     0.86286,     0.86386,
#            0.86486,     0.86587,     0.86687,     0.86787,     0.86887,     0.86987,     0.87087,     0.87187,     0.87287,     0.87387,     0.87487,     0.87588,     0.87688,     0.87788,     0.87888,     0.87988,     0.88088,     0.88188,     0.88288,     0.88388,     0.88488,     0.88589,     0.88689,     0.88789,
#            0.88889,     0.88989,     0.89089,     0.89189,     0.89289,     0.89389,     0.89489,      0.8959,      0.8969,      0.8979,      0.8989,      0.8999,      0.9009,      0.9019,      0.9029,      0.9039,      0.9049,     0.90591,     0.90691,     0.90791,     0.90891,     0.90991,     0.91091,     0.91191,
#            0.91291,     0.91391,     0.91491,     0.91592,     0.91692,     0.91792,     0.91892,     0.91992,     0.92092,     0.92192,     0.92292,     0.92392,     0.92492,     0.92593,     0.92693,     0.92793,     0.92893,     0.92993,     0.93093,     0.93193,     0.93293,     0.93393,     0.93493,     0.93594,
#            0.93694,     0.93794,     0.93894,     0.93994,     0.94094,     0.94194,     0.94294,     0.94394,     0.94494,     0.94595,     0.94695,     0.94795,     0.94895,     0.94995,     0.95095,     0.95195,     0.95295,     0.95395,     0.95495,     0.95596,     0.95696,     0.95796,     0.95896,     0.95996,
#            0.96096,     0.96196,     0.96296,     0.96396,     0.96496,     0.96597,     0.96697,     0.96797,     0.96897,     0.96997,     0.97097,     0.97197,     0.97297,     0.97397,     0.97497,     0.97598,     0.97698,     0.97798,     0.97898,     0.97998,     0.98098,     0.98198,     0.98298,     0.98398,
#            0.98498,     0.
# PS D:\git\PestVisionChallenge>







# """