"""
AMPIX FastAPI Backend — Centralized Risk Service (backend/risk_service.py)
Single Source of Truth for Risk Assessment, Layman Explanations, and MCB Trip Logic
"""

# Constant Risk Mapping per Hazard Class
HAZARD_CONSTANT_RISK = {
    "normal": (5.0, "LOW", "safe"),
    "harmonic_distortion": (35.0, "MEDIUM", "warning"),
    "overcurrent_overload": (65.0, "HIGH", "warning"),
    "ground_fault_leakage": (80.0, "CRITICAL", "warning"),
    "arc_fault_series": (90.0, "CRITICAL", "critical"),
    "arc_fault_parallel": (95.0, "CRITICAL", "critical"),
    "short_circuit": (100.0, "CRITICAL", "critical"),
}


class RiskService:
    @staticmethod
    def evaluate_risk(pred_class: str):
        risk_score, risk_level, safety_type = HAZARD_CONSTANT_RISK.get(
            pred_class, HAZARD_CONSTANT_RISK["normal"]
        )
        should_trip = pred_class in ["short_circuit", "arc_fault_series", "arc_fault_parallel"] or risk_score >= 80.0
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "safety_type": safety_type,
            "should_trip": should_trip,
        }


risk_service = RiskService()
