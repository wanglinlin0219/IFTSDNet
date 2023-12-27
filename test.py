"""
test process by test datasets
"""
import warnings
import torch.utils.data
from tqdm import tqdm
from utils.parser import get_parser_with_args
from utils.Related import get_test_loaders
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test_loader = get_test_loaders(opt)

"""load the weighted file and model"""
for i in range(66, 67):
    path = './tmp/epoch'+'_{}'.format(i) + '.pt'
    model = torch.load(path)

    c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    model.eval()

    with torch.no_grad():
        tbar = tqdm(test_loader)
        for batch_img1, batch_img2, labels in tbar:
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)
            # labels = labels[:, :, :, 0]

            cd_preds1, cd_preds2, cd_preds3 = model(batch_img1, batch_img2)
            cd_preds1 = cd_preds1[-1]
            _, cd_preds1 = torch.max(cd_preds1, 1)

            tn, fp, fn, tp = confusion_matrix(labels.data.cpu().numpy().flatten(),
                                              cd_preds1.data.cpu().numpy().flatten(), labels=[0, 1]).ravel()

            c_matrix['tn'] += tn
            c_matrix['fp'] += fp
            c_matrix['fn'] += fn
            c_matrix['tp'] += tp

    tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * P * R / (R + P)
    OA = (tp + tn) / (tp + tn + fp + fn)
    Pe = ((tn + fn) * (tn + fp) + (fp + tp) * (fn + tp)) / ((tp + tn + fp + fn) * (tp + tn + fp + fn))
    KC = (OA - Pe) / (1 - Pe)

    print( i, 'Precision: {}\nRecall: {}\nF1-Score: {}\nOA: {}\nKC: {}'.format(P, R, F1, OA, KC))
