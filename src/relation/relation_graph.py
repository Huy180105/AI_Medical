import networkx as nx
from typing import Any
from src.relation.relation_models import ClinicalRelation


class ClinicalRelationGraphBuilder:
    """
    Builds document-level NetworkX directed graphs from extracted clinical relations.
    """

    @staticmethod
    def build_graph(relations: list[ClinicalRelation]) -> nx.DiGraph:
        """
        Constructs a NetworkX DiGraph where nodes are entities and edges represent clinical relations.
        """
        g = nx.DiGraph()

        for rel in relations:
            sub = rel.subject
            obj = rel.object

            sub_node_id = f"{sub.type}:{sub.text}"
            obj_node_id = f"{obj.type}:{obj.text}"

            g.add_node(sub_node_id, label=sub.text, type=sub.type)
            g.add_node(obj_node_id, label=obj.text, type=obj.type)

            g.add_edge(
                sub_node_id,
                obj_node_id,
                relation_type=rel.relation_type,
                confidence=rel.confidence
            )

        return g
