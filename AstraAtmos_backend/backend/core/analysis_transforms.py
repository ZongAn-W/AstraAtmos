"""
数据预处理转换器 (分析专用)
负责物理变换和基于 StandardScaler 的标准化。
注意：此版本仅用于数据分析和可视化展示，不保证与训练分布对齐。
"""
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import MCD_VARIABLES
from services.data_service import DataService

logger = logging.getLogger("astraatmos.core.analysis_transforms")


class AnalysisTransforms:
    def __init__(self, data_service: DataService):
        self.data_service = data_service
        self.scalers = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

        self._compute_scalers()

    def _compute_scalers(self):
        try:
            # 数据分析只使用 MY27 作为基准
            om = self.data_service.get_openmars_data(27)
            am = self.data_service.get_aligned_mcd_data(27)

            o3 = om["o3col"]
            T, H, W = o3.shape

            self.y_mean = float(np.nanmean(o3))
            self.y_std = float(np.nanstd(o3))
            if self.y_std < 1e-10:
                self.y_std = 1.0

            self.scalers = []

            scaler_o3 = StandardScaler()
            scaler_o3.fit(np.nan_to_num(o3, nan=0.0).reshape(T, -1))
            self.scalers.append(scaler_o3)

            for var_name in MCD_VARIABLES:
                if var_name in am:
                    data = np.nan_to_num(am[var_name], nan=0.0)
                    if var_name == "Dust_Optical_Depth":
                        data = np.where(data < 0, 0.0, data)
                        data = np.log1p(data)
                    elif var_name == "Solar_Flux_DN":
                        # 分析端使用 sample-wise max 即可
                        max_val = np.max(np.abs(data))
                        if max_val > 1e-6:
                            data = data / max_val
                    scaler = StandardScaler()
                    scaler.fit(data.reshape(T, -1))
                    self.scalers.append(scaler)
                else:
                    dummy = StandardScaler()
                    dummy.mean_ = np.zeros(H * W)
                    dummy.scale_ = np.ones(H * W)
                    dummy.var_ = np.ones(H * W)
                    dummy.n_features_in_ = H * W
                    self.scalers.append(dummy)

            logger.info(f"[Analysis] 标准化参数计算完成: y_mean={self.y_mean:.4f}, y_std={self.y_std:.4f}")

        except Exception as e:
            logger.warning(f"[Analysis] 计算标准化参数失败: {e}，将跳过标准化")
            self.scalers = None
            self.y_mean = 0.0
            self.y_std = 1.0

    def apply_physical_preprocess(self, input_arr: np.ndarray) -> np.ndarray:
        result = input_arr.copy()

        dust = result[:, 5, :, :]
        dust = np.where(dust < 0, 0.0, dust)
        result[:, 5, :, :] = np.log1p(dust)

        flux = result[:, 6, :, :]
        max_val = np.max(np.abs(flux))
        if max_val > 1e-6:
            result[:, 6, :, :] = flux / max_val

        return result

    def preprocess_input(self, input_arr: np.ndarray) -> np.ndarray:
        result = self.apply_physical_preprocess(input_arr)

        if self.scalers is None:
            return result

        window, C, H, W = result.shape
        for c in range(C):
            if self.scalers[c] is not None:
                flat = result[:, c, :, :].reshape(window, -1)
                result[:, c, :, :] = self.scalers[c].transform(flat).reshape(window, H, W)

        return result

    def postprocess_output(self, pred_scaled: np.ndarray) -> np.ndarray:
        return pred_scaled * self.y_std + self.y_mean
