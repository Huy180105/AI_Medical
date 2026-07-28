import json
from pathlib import Path
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ExplanationVisualizer:
    """
    Visualization engine for clinical decision explanations.
    Generates static PNG structural diagrams and interactive Vis.js HTML dashboards.
    """

    # Node color mapping for visual consistency
    NODE_COLORS = {
        "telemetry": "#60A5FA",       # Blue
        "anomaly": "#FB923C",         # Orange
        "risk": "#EF4444",            # Red
        "disease": "#A78BFA",         # Purple
        "rule": "#F472B6",            # Pink
        "guideline": "#34D399",       # Mint Green
        "recommendation": "#10B981",  # Emerald Green
    }

    @classmethod
    def export_png(cls, evidence_graph: nx.DiGraph, output_path: str) -> None:
        """
        Generates a static color-coded PNG diagram of the evidence graph.
        """
        plt.figure(figsize=(10, 8))
        
        # Determine node colors
        colors = []
        labels = {}
        for node, data in evidence_graph.nodes(data=True):
            n_type = data.get("type", "telemetry")
            colors.append(cls.NODE_COLORS.get(n_type, "#9CA3AF"))
            labels[node] = data.get("label", node)

        # Calculate layout
        pos = nx.spring_layout(evidence_graph, k=1.2, seed=42)
        
        # Draw nodes, edges, and labels
        nx.draw_networkx_nodes(evidence_graph, pos, node_color=colors, node_size=1800, alpha=0.95)
        nx.draw_networkx_edges(evidence_graph, pos, arrowstyle="->", arrowsize=15, edge_color="#9CA3AF", width=1.5)
        nx.draw_networkx_labels(evidence_graph, pos, labels, font_size=8, font_family="sans-serif", font_color="#1F2937")

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(evidence_graph, "label")
        nx.draw_networkx_edge_labels(evidence_graph, pos, edge_labels=edge_labels, font_size=7, font_color="#4B5563")

        plt.title("Clinical Decision Support - Evidence Reasoning Graph", fontsize=12, fontweight="bold", pad=15)
        plt.axis("off")
        plt.tight_layout()
        
        # Ensure parent directories exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    @classmethod
    def export_html(cls, evidence_graph: nx.DiGraph, output_path: str) -> None:
        """
        Generates a stunning, interactive Vis.js HTML dashboard to explore the evidence graph.
        """
        # Map nodes
        nodes = []
        for node_id, data in evidence_graph.nodes(data=True):
            n_type = data.get("type", "telemetry")
            color = cls.NODE_COLORS.get(n_type, "#9CA3AF")
            
            # Format nodes with distinct shapes
            shape = "dot"
            if n_type == "risk":
                shape = "diamond"
            elif n_type == "recommendation":
                shape = "box"
                
            nodes.append({
                "id": node_id,
                "label": data.get("label", ""),
                "color": color,
                "shape": shape,
                "size": 25 if n_type in ("risk", "anomaly") else 18,
                "font": {"color": "#FFFFFF" if shape == "box" else "#333333", "size": 12},
                "title": f"Type: {n_type.capitalize()}<br>Value: {data.get('value', '')}"
            })

        # Map edges
        edges = []
        for u, v, data in evidence_graph.edges(data=True):
            edges.append({
                "from": u,
                "to": v,
                "label": data.get("label", ""),
                "arrows": "to",
                "color": {"color": "#9CA3AF", "highlight": "#4F46E5"},
                "font": {"align": "top", "size": 9}
            })

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical Decision Support System - Explanation Graph</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            margin: 0;
            font-family: 'Outfit', sans-serif;
            background-color: #0F172A;
            color: #E2E8F0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }}
        header {{
            background: linear-gradient(135deg, #1E1B4B 0%, #0F172A 100%);
            padding: 15px 30px;
            border-bottom: 1px solid #1E293B;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        h1 {{
            margin: 0;
            font-size: 1.5rem;
            background: linear-gradient(to right, #60A5FA, #A78BFA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .badge {{
            background-color: #312E81;
            color: #C084FC;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid #4338CA;
        }}
        #container {{
            display: flex;
            flex: 1;
            position: relative;
        }}
        #mynetwork {{
            flex: 1;
            height: 100%;
            background-color: #0F172A;
        }}
        #sidebar {{
            width: 320px;
            background-color: #1E293B;
            border-left: 1px solid #334155;
            padding: 20px;
            box-shadow: -4px 0 15px rgba(0,0,0,0.2);
            display: flex;
            flex-direction: column;
            gap: 20px;
            overflow-y: auto;
        }}
        h3 {{
            margin-top: 0;
            color: #F8FAFC;
            font-size: 1.1rem;
            border-bottom: 1px solid #334155;
            padding-bottom: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }}
        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .instruction {{
            font-size: 0.9rem;
            color: #94A3B8;
            line-height: 1.5;
        }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>Clinical Decision Swarm</h1>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 2px;">Explainable AI Reasoning Path Dashboard</div>
        </div>
        <div class="badge">Active Session</div>
    </header>
    <div id="container">
        <div id="mynetwork"></div>
        <div id="sidebar">
            <div>
                <h3>Legend</h3>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {cls.NODE_COLORS['telemetry']};"></div>
                    <span>Sensors (Telemetry)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {cls.NODE_COLORS['anomaly']};"></div>
                    <span>Physiological Anomalies</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {cls.NODE_COLORS['risk']};"></div>
                    <span>Patient Risk Classification</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {cls.NODE_COLORS['disease']};"></div>
                    <span>KG Candidates (Disease)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {cls.NODE_COLORS['rule']};"></div>
                    <span>Clinical Safety Rules</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {cls.NODE_COLORS['guideline']};"></div>
                    <span>RAG Guidelines</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {cls.NODE_COLORS['recommendation']};"></div>
                    <span>Clinical Recommendations</span>
                </div>
            </div>
            <div>
                <h3>How to interact</h3>
                <div class="instruction">
                    • Drag nodes to reposition them for better layout.<br>
                    • Hover over any node to view exact parameter values and diagnostics.<br>
                    • Use scroll wheel to zoom in or zoom out on the reasoning tree.
                </div>
            </div>
        </div>
    </div>
    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});

        var container = document.getElementById('mynetwork');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        var options = {{
            nodes: {{
                font: {{
                    face: 'Outfit'
                }}
            }},
            edges: {{
                font: {{
                    face: 'Outfit'
                }},
                smooth: {{
                    type: 'cubicBezier',
                    forceDirection: 'none',
                    roundness: 0.5
                }}
            }},
            physics: {{
                forceAtlas2Based: {{
                    gravitationalConstant: -26,
                    centralGravity: 0.005,
                    springLength: 230,
                    springConstant: 0.18
                }},
                maxVelocity: 146,
                solver: 'forceAtlas2Based',
                timestep: 0.35,
                stabilization: {{ iterations: 150 }}
            }}
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>
"""
        # Ensure parent directories exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_template)
