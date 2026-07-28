import csv
import json
from pathlib import Path
from typing import Any

import networkx as nx

from src.graph.ontology import EdgeType, MedicalOntology, NodeType
from src.utils.config import Config


class MedicalGraphBuilder:
    def __init__(self, knowledge_base_dir: str | None = None) -> None:
        self.knowledge_base_dir = Path(knowledge_base_dir or Config.KNOWLEDGE_BASE_DIR)
        self.ontology = MedicalOntology()

    def build(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        self._add_static_clinical_nodes(graph)
        self._load_icd10(graph)
        self._load_rxnorm(graph)
        self._load_guidelines(graph)
        self._infer_related_diseases(graph)
        return graph

    def _load_icd10(self, graph: nx.MultiDiGraph) -> None:
        path = self.knowledge_base_dir / "icd10.csv"
        if not path.exists():
            return
        with path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                title = row.get("title", "").strip()
                code = row.get("code", "").strip()
                description = row.get("description", "")
                keywords = row.get("keywords", "")
                category = row.get("category", "")
                disease_id = self._add_node(
                    graph,
                    NodeType.DISEASE,
                    title,
                    code=code,
                    category=category,
                    description=description,
                    source="icd10.csv",
                )
                icd10_id = self._add_node(graph, NodeType.ICD10, code, title=title, source="icd10.csv")
                self._add_edge(graph, disease_id, icd10_id, EdgeType.MAPPED_TO, source="icd10.csv")
                self._link_mentions(graph, disease_id, f"{title} {description} {keywords}", source="icd10.csv")

    def _load_rxnorm(self, graph: nx.MultiDiGraph) -> None:
        path = self.knowledge_base_dir / "rxnorm.csv"
        if not path.exists():
            return
        with path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                name = row.get("name", "").strip()
                code = row.get("code", "").strip()
                description = row.get("description", "")
                keywords = row.get("keywords", "")
                drug_id = self._add_node(
                    graph,
                    NodeType.DRUG,
                    name,
                    code=code,
                    category=row.get("category", ""),
                    description=description,
                    source="rxnorm.csv",
                )
                self._link_drug_semantics(graph, drug_id, f"{name} {description} {keywords}", source="rxnorm.csv")

    def _load_guidelines(self, graph: nx.MultiDiGraph) -> None:
        path = self.knowledge_base_dir / "medical_guideline.json"
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("documents", [])
        for row in rows:
            title = str(row.get("title", row.get("id", "Guideline")))
            body = str(row.get("text", ""))
            keywords = " ".join(row.get("keywords", []))
            guideline_id = self._add_node(
                graph,
                NodeType.GUIDELINE,
                title,
                code=str(row.get("id", "")),
                category=row.get("category", ""),
                description=body,
                source="medical_guideline.json",
            )
            mentioned_diseases = self._link_mentions(graph, guideline_id, f"{title} {body} {keywords}", source="medical_guideline.json")
            for disease_id in mentioned_diseases:
                self._add_edge(graph, disease_id, guideline_id, EdgeType.GUIDED_BY, source="medical_guideline.json")

    def _add_static_clinical_nodes(self, graph: nx.MultiDiGraph) -> None:
        for name in MedicalOntology.SYMPTOM_ALIASES:
            self._add_node(graph, NodeType.SYMPTOM, name, aliases=MedicalOntology.SYMPTOM_ALIASES[name], source="ontology")
        for name in MedicalOntology.DISEASE_ALIASES:
            self._add_node(graph, NodeType.DISEASE, name, aliases=MedicalOntology.DISEASE_ALIASES[name], source="ontology")
        for name in MedicalOntology.LAB_ALIASES:
            self._add_node(graph, NodeType.LAB, name, aliases=MedicalOntology.LAB_ALIASES[name], source="ontology")
        for name in MedicalOntology.PROCEDURE_ALIASES:
            self._add_node(graph, NodeType.PROCEDURE, name, aliases=MedicalOntology.PROCEDURE_ALIASES[name], source="ontology")
        for name in MedicalOntology.COMPLICATION_ALIASES:
            self._add_node(graph, NodeType.COMPLICATION, name, aliases=MedicalOntology.COMPLICATION_ALIASES[name], source="ontology")

    def _link_mentions(self, graph: nx.MultiDiGraph, source_id: str, text: str, source: str) -> list[str]:
        disease_ids: list[str] = []
        for disease, aliases in MedicalOntology.DISEASE_ALIASES.items():
            if MedicalOntology.contains_any(text, aliases + [disease]):
                disease_id = MedicalOntology.node_id(NodeType.DISEASE, disease)
                disease_ids.append(disease_id)
                if source_id != disease_id:
                    self._add_edge(graph, source_id, disease_id, EdgeType.RELATED_TO, source=source)
        for symptom, aliases in MedicalOntology.SYMPTOM_ALIASES.items():
            if MedicalOntology.contains_any(text, aliases + [symptom]):
                symptom_id = MedicalOntology.node_id(NodeType.SYMPTOM, symptom)
                self._add_edge(graph, source_id, symptom_id, EdgeType.HAS_SYMPTOM, source=source)
        for lab, aliases in MedicalOntology.LAB_ALIASES.items():
            if MedicalOntology.contains_any(text, aliases + [lab]):
                lab_id = MedicalOntology.node_id(NodeType.LAB, lab)
                self._add_edge(graph, source_id, lab_id, EdgeType.REQUIRES_TEST, source=source)
        for procedure, aliases in MedicalOntology.PROCEDURE_ALIASES.items():
            if MedicalOntology.contains_any(text, aliases + [procedure]):
                procedure_id = MedicalOntology.node_id(NodeType.PROCEDURE, procedure)
                self._add_edge(graph, source_id, procedure_id, EdgeType.RELATED_TO, source=source)
        for complication, aliases in MedicalOntology.COMPLICATION_ALIASES.items():
            if MedicalOntology.contains_any(text, aliases + [complication]):
                complication_id = MedicalOntology.node_id(NodeType.COMPLICATION, complication)
                self._add_edge(graph, source_id, complication_id, EdgeType.HAS_COMPLICATION, source=source)
        return disease_ids

    def _link_drug_semantics(self, graph: nx.MultiDiGraph, drug_id: str, text: str, source: str) -> None:
        for disease, aliases in MedicalOntology.DISEASE_ALIASES.items():
            if MedicalOntology.contains_any(text, aliases + [disease]):
                self._add_edge(graph, drug_id, MedicalOntology.node_id(NodeType.DISEASE, disease), EdgeType.TREATS, source=source)
        for symptom, aliases in MedicalOntology.SYMPTOM_ALIASES.items():
            if MedicalOntology.contains_any(text, aliases + [symptom]):
                self._add_edge(graph, drug_id, MedicalOntology.node_id(NodeType.SYMPTOM, symptom), EdgeType.TREATS, source=source)
        for disease in ["Kidney disease", "Liver disease", "Pregnancy", "Gastric ulcer"]:
            if MedicalOntology.contains_any(text, MedicalOntology.DISEASE_ALIASES[disease] + [disease]):
                self._add_edge(graph, drug_id, MedicalOntology.node_id(NodeType.DISEASE, disease), EdgeType.CONTRAINDICATED_FOR, source=source)

    def _infer_related_diseases(self, graph: nx.MultiDiGraph) -> None:
        disease_nodes = [node for node, data in graph.nodes(data=True) if data.get("type") == NodeType.DISEASE.value]
        for left in disease_nodes:
            left_symptoms = set(self._neighbors_by_edge(graph, left, EdgeType.HAS_SYMPTOM))
            for right in disease_nodes:
                if left >= right:
                    continue
                right_symptoms = set(self._neighbors_by_edge(graph, right, EdgeType.HAS_SYMPTOM))
                overlap = left_symptoms.intersection(right_symptoms)
                if overlap:
                    self._add_edge(graph, left, right, EdgeType.RELATED_TO, source="inference", shared_symptoms=list(overlap))
                    self._add_edge(graph, right, left, EdgeType.RELATED_TO, source="inference", shared_symptoms=list(overlap))

    def _neighbors_by_edge(self, graph: nx.MultiDiGraph, node_id: str, edge_type: EdgeType) -> list[str]:
        return [
            target
            for _, target, data in graph.out_edges(node_id, data=True)
            if data.get("type") == edge_type.value
        ]

    def _add_node(self, graph: nx.MultiDiGraph, node_type: NodeType, name: str, **attributes: Any) -> str:
        if not name:
            return ""
        payload = MedicalOntology.node_payload(node_type, name, **attributes)
        node_id = payload.pop("id")
        existing = dict(graph.nodes[node_id]) if graph.has_node(node_id) else {}
        existing.update({key: value for key, value in payload.items() if value not in (None, "")})
        graph.add_node(node_id, **existing)
        return node_id

    def _add_edge(self, graph: nx.MultiDiGraph, source_id: str, target_id: str, edge_type: EdgeType, **attributes: Any) -> None:
        if not source_id or not target_id or source_id == target_id:
            return
        graph.add_edge(source_id, target_id, type=edge_type.value, **attributes)
