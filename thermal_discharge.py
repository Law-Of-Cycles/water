import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from scipy.special import erf
from skimage import measure, morphology


DEFAULT_PARAMS = {
    "background": {
        "method": "percentile",
        "window_size_px": 90,
        "percentile": 25,
        "smooth_sigma": 5.0,
    },
    "spatial_stats": {
        "neighbor_mode": "queen",
        "kernel_radius_px": 2,
        "row_standardize": True,
    },
    "thresholds": {
        "dt_min": 3.0,
        "gistar_z_core_min": 2.3,
        "moran_z_core_min": 2.3,
        "shore_score_min": 0.4,
        "elongation_score_min": 0.4,
        "decay_score_min": 0.2,
        "min_area": 30,
    },
    "confidence": {
        "w_temp": 0.3,
        "w_gistar": 0.3,
        "w_moran": 0.2,
        "w_shape": 0.2,
    },
}


class ThermalDischargeDetector:
    def __init__(self, params: dict | None = None):
        self.params = self._merge_params(DEFAULT_PARAMS, params or {})

    @staticmethod
    def _merge_params(base: dict, override: dict) -> dict:
        result = {k: v.copy() if isinstance(v, dict) else v for k, v in base.items()}
        for key, val in override.items():
            if isinstance(val, dict) and key in result:
                result[key] = ThermalDischargeDetector._merge_params(result[key], val)
            else:
                result[key] = val
        return result

    def detect(
        self,
        T_obs: np.ndarray,
        water_mask: np.ndarray,
        modules: dict | None = None,
        return_intermediate: bool = False,
    ) -> dict:
        modules = modules or {
            "use_gistar": True,
            "use_moran": True,
            "use_shape_shore": True,
            "use_shape_elongation": True,
            "use_shape_decay": True,
        }
        params = self.params
        valid_water_mask = np.asarray(water_mask).astype(bool) & np.isfinite(T_obs)

        T_bg, delta_T = self._background_correction(T_obs, valid_water_mask)
        spatial_stats = self._compute_spatial_stats(delta_T, valid_water_mask, modules)
        shape_scores, candidate_mask = self._compute_shape_scores(
            delta_T, valid_water_mask, spatial_stats, modules
        )
        anomaly_mask, confidence_map = self._final_decision(
            delta_T, valid_water_mask, spatial_stats, shape_scores, modules
        )
        stats = self._compute_stats(anomaly_mask, valid_water_mask)

        result = {
            "anomaly_mask": anomaly_mask,
            "confidence_map": confidence_map,
            "stats": stats,
        }
        if return_intermediate:
            result["intermediate"] = {
                "T_bg": T_bg,
                "delta_T": delta_T,
                "spatial_stats": spatial_stats,
                "shape_scores": shape_scores,
                "candidate_mask": candidate_mask,
            }
        return result

    def _background_correction(self, T_obs: np.ndarray, valid_water_mask: np.ndarray):
        cfg = self.params["background"]
        window = cfg.get("window_size_px", 90)
        percentile = cfg.get("percentile", 25)
        smooth_sigma = cfg.get("smooth_sigma", 0.0)
        nan_filled = np.where(valid_water_mask, T_obs, np.nan)

        def nan_percentile(values):
            return np.nanpercentile(values, percentile)

        footprint = np.ones((window, window), dtype=bool)
        T_bg = ndimage.generic_filter(
            nan_filled,
            function=nan_percentile,
            footprint=footprint,
            mode="nearest",
            cval=np.nan,
        )

        if smooth_sigma and smooth_sigma > 0:
            finite_mask = np.isfinite(T_bg)
            T_bg_filled = np.where(finite_mask, T_bg, 0.0)
            weight = ndimage.gaussian_filter(finite_mask.astype(float), smooth_sigma)
            smoothed = ndimage.gaussian_filter(T_bg_filled, smooth_sigma)
            T_bg = np.where(weight > 0, smoothed / (weight + 1e-6), np.nan)

        delta_T = np.where(valid_water_mask, T_obs - T_bg, np.nan)
        return T_bg, delta_T

    def _compute_spatial_stats(self, delta_T, valid_water_mask, modules):
        cfg = self.params["spatial_stats"]
        r = int(cfg.get("kernel_radius_px", 2))
        row_standardize = cfg.get("row_standardize", True)
        eps = 1e-6

        kernel_size = 2 * r + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=bool)
        center = (kernel_size // 2, kernel_size // 2)
        kernel_moran = kernel.copy()
        kernel_moran[center] = False

        delta_masked = np.where(valid_water_mask, delta_T, np.nan)
        x_bar = np.nanmean(delta_masked)
        s = np.nanstd(delta_masked) + eps
        n = np.sum(np.isfinite(delta_masked))

        diff = delta_masked - x_bar

        def neighbor_mean(arr):
            vals = arr[~np.isnan(arr)]
            if vals.size == 0:
                return np.nan
            return np.mean(vals)

        neighbor_mean_diff = ndimage.generic_filter(
            diff,
            function=neighbor_mean,
            footprint=kernel_moran,
            mode="nearest",
            cval=np.nan,
        )

        moran_I = ((diff) / (s**2)) * neighbor_mean_diff
        moran_mean = np.nanmean(moran_I)
        moran_std = np.nanstd(moran_I) + eps
        moran_z = (moran_I - moran_mean) / moran_std

        def sum_and_count(arr):
            vals = arr[~np.isnan(arr)]
            if vals.size == 0:
                return np.nan, 0
            if row_standardize:
                mean_val = np.mean(vals)
                w_sum = 1.0
                w_sq_sum = 1.0 / vals.size
                return mean_val, (w_sum, w_sq_sum, vals.size)
            return np.sum(vals), (vals.size, vals.size, vals.size)

        def sum_only(arr):
            vals = arr[~np.isnan(arr)]
            return np.sum(vals) if vals.size > 0 else np.nan

        sum_values = ndimage.generic_filter(
            delta_masked,
            function=sum_only,
            footprint=kernel,
            mode="nearest",
            cval=np.nan,
        )
        count_values = ndimage.generic_filter(
            (~np.isnan(delta_masked)).astype(float),
            function=np.sum,
            footprint=kernel,
            mode="nearest",
            cval=0,
        )

        if row_standardize:
            local_mean = ndimage.generic_filter(
                delta_masked,
                function=neighbor_mean,
                footprint=kernel,
                mode="nearest",
                cval=np.nan,
            )
            gistar_raw = local_mean
            w_sum = np.ones_like(local_mean)
            w_sq_sum = 1.0 / np.maximum(count_values, 1)
        else:
            gistar_raw = sum_values
            w_sum = count_values
            w_sq_sum = count_values

        denom = s * np.sqrt(np.maximum((n * w_sq_sum) - (w_sum**2), 0) / max(n - 1, 1))
        gistar_z = (gistar_raw - x_bar * w_sum) / (denom + eps)

        return {
            "moran_I": moran_I,
            "moran_z": np.where(valid_water_mask, moran_z, np.nan),
            "gistar": gistar_raw,
            "gistar_z": np.where(valid_water_mask, gistar_z, np.nan),
        }

    def _compute_shape_scores(self, delta_T, valid_water_mask, spatial_stats, modules):
        thr = self.params["thresholds"]
        dt_min = thr.get("dt_min", 3.0)
        gistar_min = thr.get("gistar_z_core_min", 2.3) - 0.3
        moran_min = thr.get("moran_z_core_min", 2.3) - 0.3
        min_area = max(5, int(thr.get("min_area", 30) / 2))

        candidate_mask = (
            (delta_T >= dt_min - 0.5)
            & (spatial_stats["gistar_z"] >= gistar_min)
            & (spatial_stats["moran_z"] >= moran_min)
            & valid_water_mask
        )
        candidate_mask = morphology.remove_small_objects(candidate_mask, min_size=min_area)
        candidate_mask = morphology.binary_closing(candidate_mask, morphology.disk(2))
        candidate_mask = morphology.binary_opening(candidate_mask, morphology.disk(1))

        shore_mask = valid_water_mask & (~morphology.binary_erosion(valid_water_mask))
        shore_dist = distance_transform_edt(~shore_mask)

        labeled = measure.label(candidate_mask, connectivity=2)
        shore_scores = np.zeros_like(delta_T, dtype=float)
        elong_scores = np.zeros_like(delta_T, dtype=float)
        decay_scores = np.zeros_like(delta_T, dtype=float)

        for region in measure.regionprops(labeled, intensity_image=delta_T):
            coords = region.coords
            rows, cols = coords[:, 0], coords[:, 1]
            region_delta = delta_T[rows, cols]

            mean_dist = float(np.mean(shore_dist[rows, cols])) if coords.size > 0 else np.inf
            shore_score_val = np.exp(-mean_dist / 20.0) if modules.get("use_shape_shore", True) else 0.5
            shore_scores[rows, cols] = shore_score_val

            maj = region.major_axis_length or 0.0
            minor = region.minor_axis_length or 0.0
            ar = maj / (minor + 1e-6)
            if not modules.get("use_shape_elongation", True):
                elong_score_val = 0.5
            elif ar <= 1:
                elong_score_val = 0.0
            elif ar >= 3:
                elong_score_val = 1.0
            else:
                elong_score_val = (ar - 1) / (3 - 1)
            elong_scores[rows, cols] = elong_score_val

            if modules.get("use_shape_decay", True):
                if region_delta.size == 0:
                    decay_score_val = 0.0
                else:
                    top_thresh = np.nanpercentile(region_delta, 95)
                    source_candidates = np.column_stack(np.where((labeled == region.label) & (delta_T >= top_thresh)))
                    if source_candidates.size == 0:
                        decay_score_val = 0.2
                    else:
                        dists_to_shore = shore_dist[source_candidates[:, 0], source_candidates[:, 1]]
                        src_idx = int(np.argmin(dists_to_shore))
                        src_r, src_c = source_candidates[src_idx]
                        d_from_source = np.sqrt((rows - src_r) ** 2 + (cols - src_c) ** 2)
                        valid = np.isfinite(region_delta)
                        if np.sum(valid) < 3:
                            decay_score_val = 0.2
                        else:
                            corr = np.corrcoef(d_from_source[valid], region_delta[valid])[0, 1]
                            if corr <= -0.5:
                                decay_score_val = 1.0
                            elif corr >= 0.2:
                                decay_score_val = 0.0
                            else:
                                decay_score_val = (0.2 - corr) / (0.7)
                decay_scores[rows, cols] = decay_score_val
            else:
                decay_scores[rows, cols] = 0.5

        shape_score = np.zeros_like(delta_T, dtype=float)
        shape_score = (
            0.4 * shore_scores + 0.3 * elong_scores + 0.3 * decay_scores
        )
        shape_score = np.clip(shape_score, 0, 1)

        return {
            "shore_score": shore_scores,
            "elongation_score": elong_scores,
            "decay_score": decay_scores,
            "shape_score": shape_score,
        }, candidate_mask

    def _final_decision(self, delta_T, valid_water_mask, spatial_stats, shape_scores, modules):
        thr = self.params["thresholds"]
        dt_min = thr.get("dt_min", 3.0)
        gistar_min = thr.get("gistar_z_core_min", 2.3)
        moran_min = thr.get("moran_z_core_min", 2.3)
        shape_min = min(
            thr.get("shore_score_min", 0.4),
            thr.get("elongation_score_min", 0.4),
            thr.get("decay_score_min", 0.2),
        )

        anomaly_mask = (
            (delta_T >= dt_min)
            & (spatial_stats["gistar_z"] >= gistar_min if modules.get("use_gistar", True) else True)
            & (spatial_stats["moran_z"] >= moran_min if modules.get("use_moran", True) else True)
            & (shape_scores["shape_score"] >= shape_min)
            & valid_water_mask
        )

        anomaly_mask = morphology.remove_small_objects(
            anomaly_mask, min_size=thr.get("min_area", 30)
        )
        anomaly_mask = morphology.binary_opening(anomaly_mask, morphology.disk(1))

        confidence_map = self._compute_confidence(
            delta_T, spatial_stats, shape_scores, anomaly_mask, modules
        )

        return anomaly_mask, confidence_map

    def _compute_confidence(self, delta_T, spatial_stats, shape_scores, anomaly_mask, modules):
        conf_cfg = self.params["confidence"]
        thr = self.params["thresholds"]
        dt_min = thr.get("dt_min", 3.0)
        eps = 1e-6

        valid = np.isfinite(delta_T)
        dt_ref = np.nanpercentile(delta_T[valid], 95) if np.any(valid) else dt_min + 1
        conf_temp = np.clip((delta_T - dt_min) / (dt_ref - dt_min + eps), 0, 1)

        def z_to_conf(z):
            return 0.5 * (1 + erf(z / np.sqrt(2)))

        gistar_z = spatial_stats["gistar_z"]
        moran_z = spatial_stats["moran_z"]
        conf_gistar = z_to_conf(np.where(np.isfinite(gistar_z), gistar_z, 0))
        conf_moran = z_to_conf(np.where(np.isfinite(moran_z), moran_z, 0))
        conf_shape = np.clip(shape_scores["shape_score"], 0, 1)

        w_temp = conf_cfg.get("w_temp", 0.3)
        w_gistar = conf_cfg.get("w_gistar", 0.3) if modules.get("use_gistar", True) else 0
        w_moran = conf_cfg.get("w_moran", 0.2) if modules.get("use_moran", True) else 0
        w_shape = conf_cfg.get("w_shape", 0.2)
        total_w = w_temp + w_gistar + w_moran + w_shape + eps

        conf = (
            w_temp * conf_temp
            + w_gistar * conf_gistar
            + w_moran * conf_moran
            + w_shape * conf_shape
        ) / total_w

        conf = np.clip(conf, 0, 1)
        conf = np.where(anomaly_mask, conf, 0.0)
        return conf.astype(np.float32) * 100

    def _compute_stats(self, anomaly_mask, valid_water_mask):
        labeled = measure.label(anomaly_mask, connectivity=2)
        regions = measure.regionprops(labeled)
        n_clusters = len(regions)
        areas = [r.area for r in regions]
        area_array = np.array(areas) if areas else np.array([])

        water_pixels = int(np.sum(valid_water_mask))
        n_anomaly_pixels = int(np.sum(anomaly_mask))

        stats = {
            "n_anomaly_pixels": n_anomaly_pixels,
            "water_pixels": water_pixels,
            "anomaly_ratio": n_anomaly_pixels / water_pixels if water_pixels else 0,
            "n_clusters": n_clusters,
            "mean_cluster_area": float(area_array.mean()) if area_array.size else 0.0,
            "median_cluster_area": float(np.median(area_array)) if area_array.size else 0.0,
        }
        return stats


def run_param_sensitivity_experiments(scenes: list[dict]):
    detector = ThermalDischargeDetector()
    base_params = detector.params
    param_space = {
        "thresholds.dt_min": [2.5, 3.0, 3.5, 4.0],
        "thresholds.gistar_z_core_min": [1.8, 2.0, 2.3, 2.6],
        "thresholds.moran_z_core_min": [1.8, 2.0, 2.3, 2.6],
        "thresholds.min_area": [10, 20, 30, 50],
        "thresholds.shore_score_min": [0.3, 0.4, 0.5, 0.6],
    }

    all_results = []
    for param_name, values in param_space.items():
        for val in values:
            section, key = param_name.split(".")
            custom_params = ThermalDischargeDetector._merge_params(base_params, {})
            custom_params[section][key] = val
            det = ThermalDischargeDetector(params=custom_params)
            for scene in scenes:
                res = det.detect(scene["T_obs"], scene["water_mask"], return_intermediate=False)
                stats = res["stats"]
                all_results.append(
                    {
                        "param_name": param_name,
                        "param_value": val,
                        "scene_name": scene.get("name", "unknown"),
                        **stats,
                    }
                )
        df = pd.DataFrame([r for r in all_results if r["param_name"] == param_name])
        print(f"\n==== Sensitivity on {param_name} ====")
        print(df)

    summary_df = pd.DataFrame(all_results)
    print("\nCombined sensitivity summary (all parameters):")
    print(summary_df)
    return summary_df


SENSITIVITY_NOTES = """
经验趋势预期：
- dt_min 提高时，热异常像元数会单调减少，较高阈值可抑制弱噪声但可能漏检小羽流。
- gistar_z_core_min 与 moran_z_core_min 提高，可显著减少零散伪异常，聚焦高置信度热点。
- min_area 增大可减少碎片化斑块，但过大时会漏掉面积小、形状狭长的排水羽流。
- shore_score_min 提高，会更偏向贴岸羽流，减少离岸暖水体的误报，同时可能降低对远离岸线的羽流的敏感度。
"""
