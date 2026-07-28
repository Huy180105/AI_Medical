import networkx as nx
from typing import Any
from src.graph.ontology import MedicalOntology, NodeType, EdgeType


class MedicalGraphQuery:
    """
    Engine to run structured queries on the NetworkX medical knowledge graph.
    Supports queries like finding diseases by symptoms, drugs for diseases,
    contraindications, related diseases, lab tests, and complications.
    """

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph

    def _resolve_node(self, node_type: NodeType, name: str) -> str | None:
        """
        Resolves a search string/name to a node ID in the graph.
        Checks canonical node ID first, then tries attributes (name, aliases) case-insensitively.
        """
        # 1. Check canonical node ID directly
        node_id = MedicalOntology.node_id(node_type, name)
        if self.graph.has_node(node_id):
            return node_id

        # 2. Check case-insensitive match on name or aliases
        normalized = MedicalOntology.normalize_text(name)
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == node_type.value:
                if MedicalOntology.normalize_text(data.get("name", "")) == normalized:
                    return node
                aliases = data.get("aliases", [])
                if any(MedicalOntology.normalize_text(alias) == normalized for alias in aliases):
                    return node

        return None

    def find_diseases_by_symptom(self, symptom_name: str) -> list[dict[str, Any]]:
        """
        Finds all diseases associated with a given symptom.
        Traverses incoming HAS_SYMPTOM edges to find source Disease nodes.
        """
        symptom_id = self._resolve_node(NodeType.SYMPTOM, symptom_name)
        if not symptom_id:
            return []

        diseases = []
        # In a directed graph, source -> target. Edge is: Disease -> HAS_SYMPTOM -> Symptom.
        # So we look at incoming edges to the symptom node.
        for source, _, data in self.graph.in_edges(symptom_id, data=True):
            source_data = self.graph.nodes[source]
            if source_data.get("type") == NodeType.DISEASE.value and data.get("type") == EdgeType.HAS_SYMPTOM.value:
                diseases.append({
                    "id": source,
                    "name": source_data.get("name"),
                    "category": source_data.get("category", ""),
                    "description": source_data.get("description", ""),
                    "edge_data": data
                })
        return diseases

    def find_drugs_for_disease(self, disease_name: str) -> list[dict[str, Any]]:
        """
        Finds all drugs indicated/treating a given disease.
        Traverses incoming TREATS edges to find Drug nodes.
        """
        disease_id = self._resolve_node(NodeType.DISEASE, disease_name)
        if not disease_id:
            return []

        drugs = []
        # Edge is: Drug -> TREATS -> Disease
        for source, _, data in self.graph.in_edges(disease_id, data=True):
            source_data = self.graph.nodes[source]
            if source_data.get("type") == NodeType.DRUG.value and data.get("type") == EdgeType.TREATS.value:
                drugs.append({
                    "id": source,
                    "name": source_data.get("name"),
                    "code": source_data.get("code", ""),
                    "category": source_data.get("category", ""),
                    "description": source_data.get("description", ""),
                    "edge_data": data
                })
        return drugs

    def find_contraindications_for_drug(self, drug_name: str) -> list[dict[str, Any]]:
        """
        Finds all diseases or states contraindicated for a given drug.
        Traverses outgoing CONTRAINDICATED_FOR edges.
        """
        drug_id = self._resolve_node(NodeType.DRUG, drug_name)
        if not drug_id:
            return []

        contraindications = []
        # Edge is: Drug -> CONTRAINDICATED_FOR -> Disease
        for _, target, data in self.graph.out_edges(drug_id, data=True):
            target_data = self.graph.nodes[target]
            if data.get("type") == EdgeType.CONTRAINDICATED_FOR.value:
                contraindications.append({
                    "id": target,
                    "name": target_data.get("name"),
                    "type": target_data.get("type"),
                    "description": target_data.get("description", ""),
                    "edge_data": data
                })
        return contraindications

    def find_contraindications_for_disease(self, disease_name: str) -> list[dict[str, Any]]:
        """
        Finds all drugs contraindicated for a given disease.
        Traverses incoming CONTRAINDICATED_FOR edges.
        """
        disease_id = self._resolve_node(NodeType.DISEASE, disease_name)
        if not disease_id:
            return []

        drugs = []
        # Edge is: Drug -> CONTRAINDICATED_FOR -> Disease
        for source, _, data in self.graph.in_edges(disease_id, data=True):
            source_data = self.graph.nodes[source]
            if source_data.get("type") == NodeType.DRUG.value and data.get("type") == EdgeType.CONTRAINDICATED_FOR.value:
                drugs.append({
                    "id": source,
                    "name": source_data.get("name"),
                    "code": source_data.get("code", ""),
                    "description": source_data.get("description", ""),
                    "edge_data": data
                })
        return drugs

    def find_related_diseases(self, disease_name: str) -> list[dict[str, Any]]:
        """
        Finds related diseases for a given disease.
        Traverses RELATED_TO edges.
        """
        disease_id = self._resolve_node(NodeType.DISEASE, disease_name)
        if not disease_id:
            return []

        related = []
        # Edge is: Disease -> RELATED_TO -> Disease
        for _, target, data in self.graph.out_edges(disease_id, data=True):
            target_data = self.graph.nodes[target]
            if target_data.get("type") == NodeType.DISEASE.value and data.get("type") == EdgeType.RELATED_TO.value:
                related.append({
                    "id": target,
                    "name": target_data.get("name"),
                    "category": target_data.get("category", ""),
                    "shared_symptoms": data.get("shared_symptoms", []),
                    "edge_data": data
                })
        return related

    def find_required_tests(self, disease_name: str) -> list[dict[str, Any]]:
        """
        Finds laboratory tests required/recommended for a disease.
        Traverses outgoing REQUIRES_TEST edges.
        """
        disease_id = self._resolve_node(NodeType.DISEASE, disease_name)
        if not disease_id:
            return []

        tests = []
        # Edge is: Disease -> REQUIRES_TEST -> Lab
        for _, target, data in self.graph.out_edges(disease_id, data=True):
            target_data = self.graph.nodes[target]
            if target_data.get("type") == NodeType.LAB.value and data.get("type") == EdgeType.REQUIRES_TEST.value:
                tests.append({
                    "id": target,
                    "name": target_data.get("name"),
                    "description": target_data.get("description", ""),
                    "edge_data": data
                })
        return tests

    def find_disease_complications(self, disease_name: str) -> list[dict[str, Any]]:
        """
        Finds complications associated with a disease.
        Traverses HAS_COMPLICATION edges.
        """
        disease_id = self._resolve_node(NodeType.DISEASE, disease_name)
        if not disease_id:
            return []

        complications = []
        # Edge is: Disease -> HAS_COMPLICATION -> Complication
        for _, target, data in self.graph.out_edges(disease_id, data=True):
            target_data = self.graph.nodes[target]
            if target_data.get("type") == NodeType.COMPLICATION.value and data.get("type") == EdgeType.HAS_COMPLICATION.value:
                complications.append({
                    "id": target,
                    "name": target_data.get("name"),
                    "description": target_data.get("description", ""),
                    "edge_data": data
                })
        return complications
