import networkx as nx
from typing import Any


class ExplainableAIEngine:
    """
    Parses clinical agent execution states and watch sensor telemetry
    to formulate structured step-by-step reasoning paths and NetworkX evidence graphs.
    """

    @classmethod
    def generate_explanation(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Generates a complete explainability report from a CDSS/sensor run.
        """
        telemetry = payload.get("telemetry", {})
        features = payload.get("features", {})
        anomalies = payload.get("anomalies", [])
        risk_level = payload.get("risk_level", "Low")
        
        # CDSS/Clinical details can be in 'cdss_alert' or directly in payload
        cdss = payload.get("cdss_alert") or payload.get("clinical_decision") or {}
        evidence = cdss.get("evidence", {})
        recs = cdss.get("recommendations", {})

        # 1. Synthesize Reasoning Path
        reasoning_path = []

        # Step 1: Telemetry Inputs
        telemetry_text = f"Heart Rate: {telemetry.get('heart_rate', 70)} bpm, SpO2: {telemetry.get('spo2', 98)}%, Skin Temp: {telemetry.get('skin_temp', 36.6)}°C."
        reasoning_path.append({
            "step": 1,
            "title": "Raw Telemetry Ingestion",
            "description": "Wearable sensor metrics successfully read from Galaxy Watch Simulator.",
            "value": telemetry_text
        })

        # Step 2: Time-Series Feature Extraction
        features_text = (
            f"HRV RMSSD: {features.get('hrv_rmssd_ms', 25.0)}ms, "
            f"Skin Temp Trend: {features.get('temp_trend', 'Stable')}, "
            f"FFT Dominant Frequency: {features.get('dominant_frequency', 0.0)}Hz."
        )
        reasoning_path.append({
            "step": 2,
            "title": "Time-Series Feature Extraction",
            "description": "Extracted rolling statistics, spectral peak features, and heart rate variability.",
            "value": features_text
        })

        # Step 3: Anomaly & Red Flag Classification
        anoms_list = [f"{a.get('type')} ({a.get('severity')})" for a in anomalies]
        anoms_text = ", ".join(anoms_list) if anoms_list else "No active physiological anomalies detected."
        reasoning_path.append({
            "step": 3,
            "title": "Anomaly & Red Flag Detection",
            "description": "Compared extracted features against clinical parameter boundaries.",
            "value": anoms_text
        })

        # Step 4: Knowledge Graph Traversal & Rules Fired
        rules = []
        if evidence.get("red_flags"):
            rules.extend([f"Red Flag: {rf}" for rf in evidence.get("red_flags")])
        if evidence.get("contraindications"):
            rules.extend([f"Contraindicated medication for {ci.get('disease')}: {ci.get('medication')}" for ci in evidence.get("contraindications")])
        
        rules_text = "; ".join(rules) if rules else "No clinical exclusion rules or red flags fired."
        reasoning_path.append({
            "step": 4,
            "title": "Clinical Rules & Contraindications",
            "description": "Traversed Knowledge Graph pathways and verified patient safety guidelines.",
            "value": rules_text
        })

        # Step 5: Risk Stratification
        reasoning_path.append({
            "step": 5,
            "title": "Risk Stratification Outcome",
            "description": "Computed final patient risk level using rules and active anomalies severity.",
            "value": f"Risk Level: {risk_level}"
        })

        # Step 6: Diagnostic & Recommendation Dispatch
        rec_list = []
        if recs.get("recommended_labs"):
            rec_list.append(f"Labs: {', '.join(recs.get('recommended_labs'))}")
        if recs.get("recommended_medication_categories"):
            meds = [m.get("category") for m in recs.get("recommended_medication_categories") if m.get("status") != "Contraindicated"]
            rec_list.append(f"Indicated Meds: {', '.join(meds)}")
        if recs.get("referral_suggestion"):
            rec_list.append(f"Referral: {recs.get('referral_suggestion')}")

        recs_text = "; ".join(rec_list) if rec_list else "No active recommendations generated."
        reasoning_path.append({
            "step": 6,
            "title": "Therapeutic & Diagnostic Actions",
            "description": "Dispatched verified lab orders, safe drug indications, referrals, and advice.",
            "value": recs_text
        })

        # 2. Build Evidence Graph
        g = nx.DiGraph()

        # Helper to safely add nodes
        def add_node(node_id: str, label: str, node_type: str, val: Any = ""):
            g.add_node(node_id, label=label, type=node_type, value=val)

        # Base nodes
        add_node("telemetry_hr", f"HR: {telemetry.get('heart_rate', 70)} bpm", "telemetry")
        add_node("telemetry_spo2", f"SpO2: {telemetry.get('spo2', 98)}%", "telemetry")
        add_node("telemetry_temp", f"Temp: {telemetry.get('skin_temp', 36.6)}°C", "telemetry")
        add_node("risk_node", f"Risk: {risk_level}", "risk", risk_level)

        # Link anomalies
        for idx, anomaly in enumerate(anomalies):
            anom_type = anomaly.get("type", "")
            anom_node = f"anomaly_{idx}"
            add_node(anom_node, f"Anomaly: {anom_type}", "anomaly", anomaly.get("severity"))
            
            # Map telemetry to anomaly
            if "spo2" in anom_type.lower() or "hypoxia" in anom_type.lower():
                g.add_edge("telemetry_spo2", anom_node, label="triggers")
            elif "heart" in anom_type.lower() or "tachycardia" in anom_type.lower() or "bradycardia" in anom_type.lower():
                g.add_edge("telemetry_hr", anom_node, label="triggers")
            elif "fever" in anom_type.lower() or "temp" in anom_type.lower():
                g.add_edge("telemetry_temp", anom_node, label="triggers")
                
            # Map anomaly to risk
            g.add_edge(anom_node, "risk_node", label="increases")

        # Link red flags
        for idx, rf in enumerate(evidence.get("red_flags", [])):
            rf_node = f"red_flag_{idx}"
            add_node(rf_node, f"Red Flag: {rf}", "rule")
            g.add_edge(rf_node, "risk_node", label="elevates")

        # Link contraindications
        for idx, ci in enumerate(evidence.get("contraindications", [])):
            ci_node = f"contraindication_{idx}"
            add_node(ci_node, f"Contraindicated: {ci.get('medication')}", "rule")
            dis_node = f"disease_{idx}"
            add_node(dis_node, f"Disease: {ci.get('disease')}", "disease")
            
            g.add_edge(dis_node, ci_node, label="exhibits")
            g.add_edge(ci_node, "risk_node", label="complicates")

        # Link clinical guidelines
        for idx, guide in enumerate(evidence.get("guideline_sources", [])):
            g_node = f"guideline_{idx}"
            add_node(g_node, f"Guideline: {guide.get('title')}", "guideline")
            g.add_edge(g_node, "risk_node", label="validates")

        # Link recommendations
        # Lab orders
        for idx, lab in enumerate(recs.get("recommended_labs", [])):
            lab_node = f"lab_{idx}"
            add_node(lab_node, f"Order Lab: {lab}", "recommendation")
            g.add_edge("risk_node", lab_node, label="directs")

        # Indicated medications
        for idx, med in enumerate(recs.get("recommended_medication_categories", [])):
            med_node = f"med_rec_{idx}"
            med_label = f"Meds: {med.get('category')} ({med.get('status', 'Indicated')})"
            add_node(med_node, med_label, "recommendation")
            g.add_edge("risk_node", med_node, label="prescribes")

        # Referral
        if recs.get("referral_suggestion"):
            ref_node = "referral_action"
            add_node(ref_node, f"Referral: {recs.get('referral_suggestion')}", "recommendation")
            g.add_edge("risk_node", ref_node, label="routes")

        # Convert NetworkX graph to JSON-serializable list of nodes and edges
        nodes = []
        for n_id, n_data in g.nodes(data=True):
            nodes.append({
                "id": n_id,
                "label": n_data.get("label", ""),
                "type": n_data.get("type", ""),
                "value": n_data.get("value", "")
            })

        edges = []
        for u, v, e_data in g.edges(data=True):
            edges.append({
                "from": u,
                "to": v,
                "label": e_data.get("label", "")
            })

        return {
            "summary": f"Clinical decision generated with risk level '{risk_level}'. Active anomalies: {len(anomalies)}.",
            "confidence": cdss.get("confidence", 0.90),
            "reasoning_path": reasoning_path,
            "evidence_graph": {
                "nodes": nodes,
                "edges": edges
            },
            "rules_fired": rules
        }
