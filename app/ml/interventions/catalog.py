from __future__ import annotations

from typing import Any, Callable, Dict, List


CATALOG: List[Dict[str, Any]] = [
    {
        "id": "counselor_referral",
        "title": "School counselor referral",
        "stakeholder": "teacher/counselor",
        "priority": "HIGH",
        "timeframe_days": 7,
        "when": lambda s: s["emotion_risk"] >= 0.7,
        "reason": lambda s: [
            f"Distress risk is high ({s['emotion_risk']*100:.1f}%)",
            "Recommend mental-health screening and support",
        ],
    },
    {
        "id": "attendance_support",
        "title": "Attendance support plan",
        "stakeholder": "teacher/parent",
        "priority": "MEDIUM",
        "timeframe_days": 14,
        "when": lambda s: s["learning_risk"] >= 0.5,
        "reason": lambda s: [
            f"Learning risk elevated ({s['learning_risk']*100:.1f}%)",
            "Structured study plan + attendance monitoring",
        ],
    },
    {
        "id": "nutrition_check",
        "title": "Nutrition check + growth monitoring",
        "stakeholder": "health worker",
        "priority": "MEDIUM",
        "timeframe_days": 30,
        "when": lambda s: s["growth_risk"] >= 0.35,
        "reason": lambda s: [
            f"Growth risk elevated ({s['growth_risk']*100:.1f}%)",
            "Check diet diversity and follow-up measurement",
        ],
    },
]

