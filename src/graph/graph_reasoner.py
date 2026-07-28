import networkx as nx
from typing import Any
from src.graph.ontology import MedicalOntology, NodeType, EdgeType
from src.graph.graph_query import MedicalGraphQuery


class MedicalGraphReasoner:
    """
    Graph Reasoning Engine that takes clinical entities (e.g. from NER),
    maps them to canonical graph nodes, traverses relationships, ranks
    candidate diseases, and constructs explainable node paths.
    """

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph
        self.query_engine = MedicalGraphQuery(graph)
        # Create an undirected copy for path finding
        self.undirected_graph = graph.to_undirected()

    def reason(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Runs reasoning over a list of extracted entities.
        
        Args:
            entities: List of dicts, each having 'text' and 'type' keys.
            
        Returns:
            A list of candidate disease dicts, ranked by confidence.
            Each dict contains:
                - disease: Name of candidate disease.
                - confidence: Score between 0.0 and 1.0.
                - explanation: Textual reasoning explanation.
                - path: List of node-edge transition paths from matched entities.
                - evidence: Supporting matches (symptoms, drugs, labs, contraindications).
        """
        # 1. Map input entities to canonical graph nodes
        resolved_nodes: dict[str, dict[str, Any]] = {}  # node_id -> entity_info
        for ent in entities:
            ent_text = ent.get("text", "")
            ent_type = ent.get("type", "").upper()
            
            # Map NER label to NodeType
            node_type = None
            if ent_type == "SYMPTOM":
                node_type = NodeType.SYMPTOM
            elif ent_type == "DISEASE":
                node_type = NodeType.DISEASE
            elif ent_type in ("MEDICINE", "DRUG"):
                node_type = NodeType.DRUG
            elif ent_type in ("TEST", "LAB"):
                node_type = NodeType.LAB

            if not node_type:
                continue

            node_id = self.query_engine._resolve_node(node_type, ent_text)
            if node_id and node_id in self.graph:
                node_data = self.graph.nodes[node_id]
                resolved_nodes[node_id] = {
                    "original_text": ent_text,
                    "canonical_name": node_data.get("name"),
                    "type": node_type,
                    "data": node_data
                }

        if not resolved_nodes:
            return []

        # 2. Identify candidate diseases
        # Candidate diseases are Disease nodes directly or indirectly linked to resolved nodes.
        candidate_disease_ids: set[str] = set()
        
        for node_id, info in resolved_nodes.items():
            if info["type"] == NodeType.DISEASE:
                candidate_disease_ids.add(node_id)
                
            # Look at neighbor nodes in the undirected graph
            if node_id in self.undirected_graph:
                for neighbor in self.undirected_graph.neighbors(node_id):
                    neighbor_data = self.graph.nodes[neighbor]
                    if neighbor_data.get("type") == NodeType.DISEASE.value:
                        candidate_disease_ids.add(neighbor)

        # 3. Score and construct evidence for each candidate disease
        ranked_candidates: list[dict[str, Any]] = []

        for disease_id in candidate_disease_ids:
            disease_data = self.graph.nodes[disease_id]
            disease_name = disease_data.get("name")
            
            # Gather evidence
            matched_symptoms = []
            matched_drugs_treatment = []
            contraindicated_drugs = []
            matched_labs = []
            related_diseases = []
            
            # Look at all direct edges related to this disease
            if disease_id in self.graph:
                # Outgoing edges: Disease -> HAS_SYMPTOM -> Symptom, Disease -> REQUIRES_TEST -> Lab, etc.
                for _, target, edge_idx, edge_data in self.graph.out_edges(disease_id, keys=True, data=True):
                    target_type = self.graph.nodes[target].get("type")
                    edge_type = edge_data.get("type")
                    
                    if target in resolved_nodes:
                        ent_info = resolved_nodes[target]
                        if edge_type == EdgeType.HAS_SYMPTOM.value:
                            matched_symptoms.append(ent_info["canonical_name"])
                        elif edge_type == EdgeType.REQUIRES_TEST.value:
                            matched_labs.append(ent_info["canonical_name"])
                        elif edge_type == EdgeType.HAS_COMPLICATION.value:
                            # complication mapped as target
                            pass

                # Incoming edges: Drug -> TREATS -> Disease, Drug -> CONTRAINDICATED_FOR -> Disease, etc.
                for source, _, edge_idx, edge_data in self.graph.in_edges(disease_id, keys=True, data=True):
                    source_type = self.graph.nodes[source].get("type")
                    edge_type = edge_data.get("type")
                    
                    if source in resolved_nodes:
                        ent_info = resolved_nodes[source]
                        if edge_type == EdgeType.TREATS.value:
                            matched_drugs_treatment.append(ent_info["canonical_name"])
                        elif edge_type == EdgeType.CONTRAINDICATED_FOR.value:
                            contraindicated_drugs.append(ent_info["canonical_name"])

            # Check direct relationship if the disease itself was in the input
            if disease_id in resolved_nodes:
                related_diseases.append(disease_name)

            # Let's count how many total symptoms this disease has in the graph
            total_symptoms_in_graph = 0
            if disease_id in self.graph:
                for _, target, edge_data in self.graph.out_edges(disease_id, data=True):
                    if edge_data.get("type") == EdgeType.HAS_SYMPTOM.value:
                        total_symptoms_in_graph += 1

            # Calculate confidence score
            # Score components:
            # - Symptom overlap: up to 0.4
            # - Treatment match: up to 0.3
            # - Direct match: 0.2 if the disease was explicitly mentioned
            # - Lab match: 0.1
            # - Contraindication penalty: if a contraindicated drug is given, we flag it.
            score = 0.0
            
            if total_symptoms_in_graph > 0:
                score += 0.4 * (len(matched_symptoms) / total_symptoms_in_graph)
            elif len(matched_symptoms) > 0:
                score += 0.3  # Fallback if symptom details are missing but we matched some

            if matched_drugs_treatment:
                score += min(0.3, 0.15 * len(matched_drugs_treatment))
                
            if disease_id in resolved_nodes:
                score += 0.2
                
            if matched_labs:
                score += min(0.1, 0.05 * len(matched_labs))

            confidence = round(min(0.95, max(0.1, score)), 2)

            # 4. Generate path explanations
            explanation_paths = []
            node_path_strings = []
            
            for input_node_id in resolved_nodes:
                if input_node_id == disease_id:
                    explanation_paths.append([disease_id])
                    node_path_strings.append(f"Entity is a direct match: {disease_name}")
                    continue
                    
                if nx.has_path(self.undirected_graph, input_node_id, disease_id):
                    # Find shortest path
                    p = nx.shortest_path(self.undirected_graph, source=input_node_id, target=disease_id)
                    # We limit the path length to keep it reasonable (e.g. <= 4 hops)
                    if len(p) <= 4:
                        explanation_paths.append(p)
                        # Construct a readable string path representation
                        path_elements = []
                        for i in range(len(p) - 1):
                            u, v = p[i], p[i+1]
                            u_name = self.graph.nodes[u].get("name", u)
                            u_type = self.graph.nodes[u].get("type", "")
                            v_name = self.graph.nodes[v].get("name", v)
                            v_type = self.graph.nodes[v].get("type", "")
                            
                            # Determine edge label & direction in directed graph
                            edge_label = "connected_to"
                            if self.graph.has_edge(u, v):
                                edge_data = self.graph.get_edge_data(u, v)
                                edge_label = edge_data[0].get("type", "links")
                                edge_str = f"[{u_type}:{u_name}] -({edge_label})-> [{v_type}:{v_name}]"
                            elif self.graph.has_edge(v, u):
                                edge_data = self.graph.get_edge_data(v, u)
                                edge_label = edge_data[0].get("type", "links")
                                edge_str = f"[{v_type}:{v_name}] -({edge_label})-> [{u_type}:{u_name}]"
                            else:
                                edge_str = f"[{u_type}:{u_name}] -- [{v_type}:{v_name}]"
                            path_elements.append(edge_str)
                        node_path_strings.append(" | ".join(path_elements))

            # 5. Build narrative explanation
            narrative = f"Disease '{disease_name}' identified as a candidate with {int(confidence*100)}% confidence."
            findings = []
            if matched_symptoms:
                findings.append(f"matched symptoms: {', '.join(matched_symptoms)}")
            if matched_drugs_treatment:
                findings.append(f"treatments: {', '.join(matched_drugs_treatment)}")
            if matched_labs:
                findings.append(f"required labs: {', '.join(matched_labs)}")
            if contraindicated_drugs:
                findings.append(f"CRITICAL WARNING: Contraindicated drugs detected: {', '.join(contraindicated_drugs)}")

            if findings:
                narrative += " Supporting evidence: " + "; ".join(findings) + "."
            else:
                narrative += " Found indirect connection in knowledge graph."

            ranked_candidates.append({
                "disease": disease_name,
                "node_id": disease_id,
                "confidence": confidence,
                "explanation": narrative,
                "path": node_path_strings,
                "evidence": {
                    "matched_symptoms": matched_symptoms,
                    "matched_drugs_treatment": matched_drugs_treatment,
                    "contraindicated_drugs": contraindicated_drugs,
                    "matched_labs": matched_labs,
                    "related_diseases": related_diseases,
                }
            })

        # Sort by confidence descending
        ranked_candidates.sort(key=lambda x: x["confidence"], reverse=True)
        return ranked_candidates
