from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmdet.evaluation import eval_map
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.logging import print_log

from mmdet3d.evaluation import indoor_eval
from mmdet3d.registry import METRICS
from mmdet3d.structures import get_box_type

@METRICS.register_module()
class NVSMetric(BaseMetric):
    """Indoor scene evaluation metric.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.25, 0.5].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super(NVSMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        # self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        # for data_sample in data_samples:
        #     pred_3d = data_sample['pred_instances_3d']
        #     eval_ann_info = data_sample['eval_ann_info']
        #     cpu_pred_3d = dict()
        #     for k, v in pred_3d.items():
        #         if hasattr(v, 'to'):
        #             cpu_pred_3d[k] = v.to('cpu')
        #         else:
        #             cpu_pred_3d[k] = v
        #     self.results.append((eval_ann_info, cpu_pred_3d))
        for data_sample in data_samples:
            ssim = data_sample['ssim'] # already cpu numpy
            psnr = data_sample['psnr']
            rmse = data_sample['rmse']
            self.results.append((psnr,ssim,rmse))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        final_psnr = []
        final_ssim = []
        final_rmse = []

        # for eval_ann, sinlge_pred_results in results:
        #     ann_infos.append(eval_ann)
        #     pred_results.append(sinlge_pred_results)

        # # some checkpoints may not record the key "box_type_3d"
        # box_type_3d, box_mode_3d = get_box_type(
        #     self.dataset_meta.get('box_type_3d', 'depth'))

        # ret_dict = indoor_eval(
        #     ann_infos,
        #     pred_results,
        #     self.iou_thr,
        #     self.dataset_meta['classes'],
        #     logger=logger,
        #     box_mode_3d=box_mode_3d) # is this a dict? each key is an evaluation result?

        for psnr, ssim, rmse in results:
            final_psnr.append(psnr)
            final_ssim.append(ssim)
            final_rmse.append(rmse)
            
        
        final_psnr = sum(final_psnr) / len(final_psnr)
        final_ssim = sum(final_ssim) / len(final_ssim)
        final_rmse = sum(final_rmse) / len(final_rmse)
        ret_dict = {'psnr': final_psnr, 'ssim': final_ssim, 'rmse': final_rmse}
        print_log('NVS results: ssim: ' + str(final_ssim) + " psnr: " + str(final_psnr) + ' rmse: ' + str(final_rmse) , logger=logger)
        
        return ret_dict
    


# evaluate src depth prediction
@METRICS.register_module()
class GaussianDepthMetric(BaseMetric):
    '''
    evaluate src depth prediction parameterized by gaussian
    '''

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super(GaussianDepthMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        # self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            mu_gt_gap = data_sample['mu_gt_gap'].cpu().numpy() # already cpu numpy
            avg_sigma = data_sample['avg_sigma'].cpu().numpy()
            avg_mu = data_sample['avg_mu'].cpu().numpy()
            weight_gap = data_sample['weight_gap'].cpu().numpy()
            self.results.append((mu_gt_gap, avg_sigma, avg_mu, weight_gap))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        final_mu_gt_gap = []
        final_avg_sigma = []
        final_avg_mu = []
        final_weight_gap = []
    
        for mu_gt_gap, avg_sigma, avg_mu, weight_gap in results:
            final_mu_gt_gap.append(mu_gt_gap)
            final_avg_sigma.append(avg_sigma)
            final_avg_mu.append(avg_mu)
            final_weight_gap.append(weight_gap)
            
        final_mu_gt_gap = sum(final_mu_gt_gap) / len(final_mu_gt_gap)
        final_avg_sigma = sum(final_avg_sigma) / len(final_avg_sigma)
        final_avg_mu = sum(final_avg_mu) / len(final_avg_mu)
        final_weight_gap = sum(final_weight_gap) / len(final_weight_gap)
        
        ret_dict = {'mu_gt_gap': final_mu_gt_gap, 'avg_sigma': final_avg_sigma, 'avg_mu': final_avg_mu, 'weight_gap': weight_gap}
        print_log('depth prediction: mu_gt_gap: ' + str(final_mu_gt_gap) + " avg_sigma: " + str(final_avg_sigma) + 
                  " avg_mu: "+str(final_avg_mu) + "weight_gap: " + str(weight_gap), logger=logger)
        
        return ret_dict
    
    
@METRICS.register_module()
class WeightGapMetric(BaseMetric):
    '''
    evaluate weight accuracy in the feature volume
    '''

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super(WeightGapMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        # self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            weight_gap = data_sample['weight_gap'].cpu().numpy()
            true_vox_score = data_sample['true_vox_score'].cpu().numpy()
            self.results.append((weight_gap, true_vox_score))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        final_weight_gap = []
        final_true_vox_score = []
    
        for weight_gap, true_vox_score in results:
            final_weight_gap.append(weight_gap)
            final_true_vox_score.append(true_vox_score)
            
        final_weight_gap = sum(final_weight_gap) / len(final_weight_gap)
        final_true_vox_score = sum(final_true_vox_score) / len(final_true_vox_score)
        
        ret_dict = {'weight_gap': weight_gap, 'true_vox_score': final_true_vox_score}
        print_log("weight_gap: " + str(weight_gap) + " true_vox_score: " + str(final_true_vox_score), logger=logger)
        
        return ret_dict
    
    

@METRICS.register_module()
class MVSMetric(BaseMetric):
    '''
    evaluate weight accuracy in the feature volume
    '''

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super(MVSMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        # self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            weight_gap = data_sample['weight_gap'].cpu().numpy()
            src_rmse = data_sample['src_rmse'].cpu().numpy()
            self.results.append((weight_gap, src_rmse))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        final_weight_gap = []
        final_src_rmse = []
    
        for weight_gap, src_rmse in results:
            final_weight_gap.append(weight_gap)
            final_src_rmse.append(src_rmse)
            
        final_weight_gap = sum(final_weight_gap) / len(final_weight_gap)
        final_src_rmse = sum(final_src_rmse) / len(final_src_rmse)
        
        ret_dict = {'weight_gap': weight_gap, 'src_rmse': final_src_rmse}
        print_log("weight_gap: " + str(weight_gap) + " src_rmse: " + str(final_src_rmse), logger=logger)
        
        return ret_dict