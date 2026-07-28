from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from src.graph.graph_builder import MedicalGraphBuilder
from src.utils.config import Config


class MedicalGraphLoader:
    def __init__(
        self,
        graph_path: str | None = None,
        builder: MedicalGraphBuilder | None = None,
    ) -> None:
        self.graph_path = Path(graph_path or Path(Config.KNOWLEDGE_BASE_DIR) / "medical_graph.json")
        self.builder = builder or MedicalGraphBuilder()

    def load_or_build(self, rebuild: bool = False) -> nx.MultiDiGraph:
        if self.graph_path.exists() and not rebuild:
            return self.load()
        graph = self.builder.build()
        self.save(graph)
        return graph

    def load(self) -> nx.MultiDiGraph:
        payload = json_graph.node_link_graph(
            self.graph_path.read_text(encoding="utf-8"),
            directed=True,
            multigraph=True,
        )
        return nx.MultiDiGraph(payload)

    def save(self, graph: nx.MultiDiGraph) -> Path:
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json_graph.node_link_data(graph)
        import json

        self.graph_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return self.graph_path
