import os
from pathlib import Path
import pytest
import networkx as nx

from src.graph.ontology import NodeType, EdgeType
from src.graph.graph_builder import MedicalGraphBuilder
from src.graph.graph_query import MedicalGraphQuery
from src.graph.graph_reasoner import MedicalGraphReasoner
from src.graph.visualizer import GraphVisualizer
from src.utils.config import Config


@pytest.fixture
def temp_kb_dir(tmp_path) -> Path:
    """Creates a temporary knowledge base directory with mock clinical files matching the ontology."""
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    
    # 1. Create mock icd10.csv using canonical names/aliases
    icd_csv = kb_dir / "icd10.csv"
    icd_csv.write_text(
        "code,title,description,category,keywords\n"
        "J06.9,Acute upper respiratory infection,Acute upper respiratory tract infection with fever cough sore throat runny nose and fatigue,respiratory,fever;cough;sore throat;runny nose\n"
        "G43.9,Migraine unspecified,Recurrent headache often with nausea light sensitivity or vomiting,neurology,headache;nausea;vomiting\n",
        encoding="utf-8"
    )

    # 2. Create mock rxnorm.csv using descriptions that trigger relations
    rx_csv = kb_dir / "rxnorm.csv"
    rx_csv.write_text(
        "code,name,description,category,keywords\n"
        "RX-PARA,Paracetamol,Used for acute upper respiratory infection and fever. Check liver disease,medicine,paracetamol;fever;pain\n"
        "RX-IBU,Ibuprofen,Used for pain and fever. Avoid in kidney disease,medicine,ibuprofen;fever;pain\n",
        encoding="utf-8"
    )

    # 3. Create mock medical_guideline.json
    guideline_json = kb_dir / "medical_guideline.json"
    guideline_json.write_text(
        "[\n"
        "  {\n"
        "    \"id\": \"GL-RESP-001\",\n"
        "    \"title\": \"Respiratory infection initial guidance\",\n"
        "    \"category\": \"guideline\",\n"
        "    \"text\": \"For fever and cough, assess respiratory rate and oxygen saturation. Recommends paracetamol.\",\n"
        "    \"keywords\": [\"fever\", \"cough\"]\n"
        "  }\n"
        "]\n",
        encoding="utf-8"
    )
    
    return kb_dir


def test_graph_builder(temp_kb_dir):
    """Verifies that MedicalGraphBuilder properly parses files and populates nodes/edges."""
    builder = MedicalGraphBuilder(knowledge_base_dir=str(temp_kb_dir))
    graph = builder.build()
    
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0

    # Check for specific nodes
    disease_node = "Disease:acute_upper_respiratory_infection"
    assert graph.has_node(disease_node)
    assert graph.nodes[disease_node]["name"] == "Acute upper respiratory infection"
    assert graph.nodes[disease_node]["type"] == NodeType.DISEASE.value

    drug_node = "Drug:paracetamol"
    assert graph.has_node(drug_node)
    assert graph.nodes[drug_node]["name"] == "Paracetamol"
    
    # Check edges
    # Drug treats Disease
    assert graph.has_edge(drug_node, disease_node)
    edge_types = [d["type"] for _, _, d in graph.out_edges(drug_node, data=True)]
    assert EdgeType.TREATS.value in edge_types


def test_graph_query_engine(temp_kb_dir):
    """Verifies query operations on the graph query engine."""
    builder = MedicalGraphBuilder(knowledge_base_dir=str(temp_kb_dir))
    graph = builder.build()
    query_engine = MedicalGraphQuery(graph)

    # 1. Find disease by symptom
    diseases = query_engine.find_diseases_by_symptom("fever")
    assert len(diseases) > 0
    disease_names = [d["name"] for d in diseases]
    assert "Acute upper respiratory infection" in disease_names

    # 2. Find drugs for disease
    drugs = query_engine.find_drugs_for_disease("Acute upper respiratory infection")
    assert len(drugs) > 0
    drug_names = [dr["name"] for dr in drugs]
    assert "Paracetamol" in drug_names

    # 3. Find contraindications for drug
    contra = query_engine.find_contraindications_for_drug("Ibuprofen")
    assert len(contra) > 0
    assert "Kidney disease" in [c["name"] for c in contra]


def test_graph_reasoner(temp_kb_dir):
    """Verifies that MedicalGraphReasoner traverses paths and ranks candidates correctly."""
    builder = MedicalGraphBuilder(knowledge_base_dir=str(temp_kb_dir))
    graph = builder.build()
    reasoner = MedicalGraphReasoner(graph)

    # Mock NER entities using canonical aliases
    entities = [
        {"text": "sot", "type": "SYMPTOM"},  # sot = fever
        {"text": "ho", "type": "SYMPTOM"}   # ho = cough
    ]

    results = reasoner.reason(entities)
    assert len(results) > 0
    
    # The top candidate should be Acute upper respiratory infection
    top_candidate = results[0]
    assert top_candidate["disease"] == "Acute upper respiratory infection"
    assert top_candidate["confidence"] > 0.0
    assert len(top_candidate["path"]) > 0
    assert "fever" in top_candidate["evidence"]["matched_symptoms"]
    assert "cough" in top_candidate["evidence"]["matched_symptoms"]


def test_graph_visualizer(temp_kb_dir, tmp_path):
    """Verifies that GraphVisualizer successfully exports PNG and HTML files."""
    builder = MedicalGraphBuilder(knowledge_base_dir=str(temp_kb_dir))
    graph = builder.build()

    png_path = tmp_path / "graph.png"
    html_path = tmp_path / "graph.html"

    GraphVisualizer.export_png(graph, png_path)
    GraphVisualizer.export_html(graph, html_path)

    assert png_path.exists()
    assert html_path.exists()
    assert png_path.stat().st_size > 0
    assert html_path.stat().st_size > 0
