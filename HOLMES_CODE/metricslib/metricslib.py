from __future__ import division
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import random
import torch
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt

from utilslib.utilslib import imshow_t as tensor_imshow, plot_row, get_class_name

"""**Calibrated F1-score**"""

def compute_calibrated_f1score(y_test_orig, y_score_orig_max, n_classes):
    f1score_c = dict()
    for class_cntr in range(n_classes):
        y_test, y_score_max = y_test_orig, y_score_orig_max
        # one hot encoding
        if n_classes == 2:
          y_test = np.array([[1,0] if l==0 else [0,1] for l in y_test])
          y_score_max = np.array([[1,0] if l==0 else [0,1] for l in y_score_max])
        else:
          y_test = label_binarize(y_test, classes=[x for x in range(n_classes)]) 
          y_score_max = label_binarize(y_score_max, classes=[x for x in range(n_classes)]) 
        f1score_c[class_cntr] = cal_f1score(y_test[:, class_cntr], y_score_max[:, class_cntr], pi0=0.5)
        
    return f1score_c
    
def cal_f1score(y_true, y_pred, pi0=0.5):
    """
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier. (must be binary)
    pi0 : float, None by default
        The reference ratio for calibration
    """
    CM = confusion_matrix(y_true, y_pred)
    
    tn = CM[0][0]
    fn = CM[1][0]
    tp = CM[1][1]
    fp = CM[0][1] 
        
    pos = fn + tp
    
    recall = tp / float(pos)
    
    if pi0 is not None:
        pi = pos/float(tn + fn + tp + fp)
        ratio = pi*(1-pi0)/(pi0*(1-pi))
        precision = tp / float(tp + ratio*fp)
    else:
        precision = tp / float(tp + fp)
    
    if np.isnan(precision):
        precision = 0
    
    if (precision+recall)==0.0:
        f=0.0
    else:
        f = (2*precision*recall)/(precision+recall)

    return f
    
"""**Insertion/Deletion curve metrics**"""

HW = 224 * 224 # image area
n_classes = 1000

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn):
        r"""Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins', 'pres']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, img_tensor, explanation, verbose=0, save_to=None, freq=10, random_order=False, segments=None):
        r"""Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
                3 - plot only the final curve
            save_to (str): directory to save every step plots to.
        Return:
            scores (nd.array): Array containing scores at every step.
        """
        # original image prediction
        pred = self.model(img_tensor.cuda())
        # top class and score
        top, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]
        # (224x224=50176 + 224 -1) // 224 = 224
        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion'
            ylabel = 'Pixels deleted'
            # start with original image
            start = img_tensor.clone()
            # end with completely gray(zero) image
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion'
            ylabel = 'Pixels inserted'
            # start with blurred image
            start = self.substrate_fn(img_tensor)
            # end with original image
            finish = img_tensor.clone()
        elif self.mode == 'pres':
            title = 'Preservation'
            ylabel = 'Pixels deleted'
            # start with original image
            start = img_tensor.clone()
            # end with completely gray(zero) image
            finish = self.substrate_fn(img_tensor)

        # 225
        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency (or increasing in preservation mode)
        if segments is not None:
          # delete segmentation regions at random order (for the random heatmap comparison)
          salient_order = segments
          # number of segments
          num_segments = np.unique(segments)
          order_segments = num_segments
          # shuffle order
          random.shuffle(order_segments)
          # assign to segments another index
          for i, num in enumerate(order_segments):
            salient_order[ salient_order== num ] = i
          salient_order = np.argsort(salient_order)
          salient_order = np.array([list(salient_order)])
        elif random_order == False:
          if self.mode == 'pres':
            # delete from less salient to more salient
            salient_order = np.argsort(explanation.reshape(-1, HW), axis=1)
          else:
            # delete from more salient to less salient 
            salient_order = np.flip(np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1) # 1x(224x224=50176)
        else:
          # random order
          salient_order = np.array(list(range(HW)))
          random.shuffle(salient_order) # in-place operation
          salient_order = np.array([salient_order])
        #for i in tqdm(range(n_steps+1)):
        for i in range(n_steps+1):
            # prediction of starting point image
            pred = self.model(start.cuda())
            # pass trough softmax 
            pred = F.softmax(pred, dim=1)
            # get score and class
            pr, cl = torch.topk(pred, 2)
            if verbose == 2:
                print('{}: {:.3f}'.format(get_class_name(cl[0][0]), float(pr[0][0])))
                print('{}: {:.3f}'.format(get_class_name(cl[0][1]), float(pr[0][1])))
            # register the top score in the scores
            scores[i] = pred[0, c]
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and (i == n_steps or i % freq == 0)) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')
                tensor_imshow(start[0])

                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()

            if verbose == 3 and (i == n_steps):
                plt.figure(figsize=(5, 5))
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_class_name(c))
                plt.show()
            # if not last step
            if i < n_steps:
                # 0:224 -> 224:448 -> ...
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                # take portion of end point image and plug it in the current step image
                start.cpu().numpy().reshape(1, 3, HW)[0, :, coords] = finish.cpu().numpy().reshape(1, 3, HW)[0, :, coords]
        return scores
        