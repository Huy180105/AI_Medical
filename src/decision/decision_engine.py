import networkx as nx
from typing import Any

from src.graph.ontology import NodeType, EdgeType
from src.decision.clinical_rules import ClinicalRuleEngine
from src.decision.risk_engine import ClinicalRiskEngine
from src.decision.recommendation import ClinicalRecommendationEngine
from src.decision.followup import ClinicalFollowUpEngine
from src.decision.validator import ClinicalDecisionValidator


class ClinicalDecisionEngine:
    """
    Orchestrates the entire Clinical Decision Support System (CDSS).
    Coordinates rules evaluation, risk assessment, recommendations,
    follow-up scheduling, and output validation.
    """

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph

    def make_decision(
        self, entities: list[dict[str, Any]], graph_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Processes inputs and generates a validated unified clinical decision payload.
        """
        # 1. Evaluate clinical rules (Red flags and Contraindications)
        red_flags = ClinicalRuleEngine.check_red_flags(entities)
        contraindications = ClinicalRuleEngine.check_contraindications(entities, graph_results)

        # 2. Assess Risk Level
        risk_level = ClinicalRiskEngine.assess_risk(
            entities, graph_results, red_flags, contraindications
        )

        # 3. Generate clinical recommendations
        recs = ClinicalRecommendationEngine.generate_recommendations(
            entities, graph_results, risk_level, contraindications
        )

        # 4. Schedule Follow-up plan
        follow_up = ClinicalFollowUpEngine.determine_followup(
            risk_level, entities, graph_results
        )

        # 5. Extract guideline sources from graph
        guideline_sources = self._extract_guidelines(graph_results)

        # 6. Aggregate evidence and compute overall confidence
        supporting_paths = []
        for cand in graph_results:
            supporting_paths.extend(cand.get("path", []))

        # Overall confidence is the max of the candidates or a default base
        overall_confidence = max(
            (cand.get("confidence", 0.0) for cand in graph_results), default=0.3
        )
        
        # Penalize confidence slightly if there are contraindications
        if contraindications:
            overall_confidence = round(max(0.1, overall_confidence - 0.15), 2)

        # Assemble final decision payload
        payload = {
            "diagnosis_candidates": recs["diagnosis_candidates"],
            "risk_level": risk_level,
            "recommendations": {
                "recommended_labs": recs["recommended_labs"],
                "recommended_medication_categories": recs["recommended_medication_categories"],
                "referral_suggestion": recs["referral_suggestion"],
                "lifestyle_advice": recs["lifestyle_advice"],
            },
            "follow_up": follow_up,
            "evidence": {
                "supporting_paths": supporting_paths,
                "red_flags": red_flags,
                "contraindications": contraindications,
                "guideline_sources": guideline_sources,
            },
            "confidence": overall_confidence,
        }

        # 7. Validate structural output
        is_valid, validation_errors = ClinicalDecisionValidator.validate(payload)
        if not is_valid:
            # We don't fail outright in production to maintain service availability,
            # but we append warnings to the payload for clinical visibility.
            payload["validation_warnings"] = validation_errors

        return payload

    def _extract_guidelines(self, graph_results: list[dict[str, Any]]) -> list[dict[str, str]]:
        """
        Traverses the knowledge graph to extract relevant clinical guideline guidelines
        associated with suspected candidate diseases.
        """
        guidelines = []
        disease_ids = [res.get("node_id") for res in graph_results if res.get("node_id")]

        for disease_id in disease_ids:
            if not self.graph.has_node(disease_id):
                continue
            
            # Look at neighbors of the disease node in the directed graph
            # Disease -> GUIDED_BY -> Guideline
            for _, target, edge_data in self.graph.out_edges(disease_id, data=True):
                target_data = self.graph.nodes[target]
                if (
                    edge_data.get("type") == EdgeType.GUIDED_BY.value
                    or target_data.get("type") == NodeType.GUIDELINE.value
                ):
                    g_title = target_data.get("name", "Clinical Guideline")
                    g_text = target_data.get("description", "")
                    g_code = target_data.get("code", "")
                    
                    g_info = {"title": g_title, "code": g_code, "text": g_text}
                    if g_info not in guidelines:
                        guidelines.append(g_info)

        return guidelines
