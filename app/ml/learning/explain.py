from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


def _get_feature_names(pipeline: Any) -> List[str]:
    pre = pipeline.named_steps["preprocess"]
    return list(pre.get_feature_names_out())


def explain_instance_shap(pipeline: Any, X_row_df, shap_bg=None, top_n: int = 8) -> Optional[list[dict]]:
    """Return per-instance SHAP-style contributions if shap is available."""
    try:
        import shap  # type: ignore
    except Exception:
        return None

    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    if hasattr(model, "calibrated_classifiers_") and getattr(model, "calibrated_classifiers_", None):
        base_model = model.calibrated_classifiers_[0].estimator
    else:
        base_model = model

    X_trans = pre.transform(X_row_df)
    feature_names = _get_feature_names(pipeline)

    if shap_bg is None:
        bg = np.zeros((50, X_trans.shape[1]), dtype=float)
    else:
        bg = shap_bg

    explainer = shap.TreeExplainer(base_model, data=bg, feature_names=feature_names)
    shap_vals = explainer.shap_values(X_trans)
    sv = shap_vals[0]
    vals = X_trans[0]

    idx = np.argsort(np.abs(sv))[::-1][:top_n]
    top: list[dict] = []
    for i in idx:
        top.append(
            {
                "feature": feature_names[i],
                "contribution": float(sv[i]),
                "value": float(vals[i]) if np.isscalar(vals[i]) else float(vals[i]),
            }
        )
    return top


def explain_fallback_importance(pipeline: Any, top_n: int = 8) -> list[dict]:
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    if hasattr(model, "calibrated_classifiers_") and getattr(model, "calibrated_classifiers_", None):
        base_model = model.calibrated_classifiers_[0].estimator
    else:
        base_model = model

    if not hasattr(base_model, "feature_importances_"):
        return []

    feat_names = list(pre.get_feature_names_out())
    imps = base_model.feature_importances_
    idx = np.argsort(imps)[::-1][:top_n]
    return [{"feature": feat_names[i], "importance": float(imps[i])} for i in idx]
