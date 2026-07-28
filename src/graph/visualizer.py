import json
from pathlib import Path
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GraphVisualizer:
    """
    Visualization engine for the Medical Knowledge Graph.
    Supports exporting to static PNG diagrams and interactive, premium HTML dashboards.
    """

    NODE_COLORS = {
        "Disease": "#FF6B6B",        # Coral Red
        "Drug": "#4D96FF",           # Neon Blue
        "Symptom": "#FFD93D",        # Bright Gold
        "Lab": "#6BCB77",            # Emerald Green
        "Guideline": "#B983FF",      # Soft Purple
        "ICD10": "#6E85B7",          # Steel Blue
        "Complication": "#FF8E9E",   # Rose Pink
        "Procedure": "#A2B5BB",      # Slate Grey
        "Condition": "#E8AA42",      # Ochre Yellow
    }

    DEFAULT_COLOR = "#94A3B8"

    @classmethod
    def export_png(cls, graph: nx.MultiDiGraph, output_path: str | Path) -> Path:
        """
        Generates and saves a static PNG visualization of the graph using matplotlib.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build node color list
        node_colors = []
        for node, data in graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            node_colors.append(cls.NODE_COLORS.get(node_type, cls.DEFAULT_COLOR))

        # Size nodes based on degree
        degrees = dict(graph.degree())
        node_sizes = [300 + (degrees[n] * 100) for n in graph.nodes()]

        plt.figure(figsize=(14, 10), facecolor="#0F172A")
        ax = plt.gca()
        ax.set_facecolor("#0F172A")

        # Layout
        pos = nx.spring_layout(graph, k=0.5, seed=42)

        # Draw nodes and edges
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors="#FFFFFF",
            linewidths=1.0,
            ax=ax,
        )

        # Draw directed edges with curves
        nx.draw_networkx_edges(
            graph,
            pos,
            edge_color="#334155",
            width=1.2,
            alpha=0.6,
            arrows=True,
            arrowsize=12,
            connectionstyle="arc3,rad=0.15",
            ax=ax,
        )

        # Draw labels
        labels = {n: graph.nodes[n].get("name", n) for n in graph.nodes()}
        nx.draw_networkx_labels(
            graph,
            pos,
            labels=labels,
            font_size=8,
            font_color="#F8FAFC",
            font_weight="bold",
            alpha=0.9,
            ax=ax,
        )

        # Create custom legend
        legend_elements = [
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                label=group,
            )
            for group, color in cls.NODE_COLORS.items()
        ]
        plt.legend(
            handles=legend_elements,
            loc="upper right",
            facecolor="#1E293B",
            edgecolor="#334155",
            labelcolor="#F8FAFC",
            framealpha=0.9,
            title="Entity Types",
            title_fontsize="medium",
        )

        plt.title(
            "Medical Knowledge Graph Engine",
            color="#F8FAFC",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.axis("off")
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#0F172A")
        plt.close()

        return output_path

    @classmethod
    def export_html(cls, graph: nx.MultiDiGraph, output_path: str | Path) -> Path:
        """
        Generates an interactive, responsive HTML visualization using Vis.js.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        nodes_data = []
        for node, data in graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            color = cls.NODE_COLORS.get(node_type, cls.DEFAULT_COLOR)
            name = data.get("name", node)
            code = data.get("code", "")
            desc = data.get("description", "")
            
            # Simple tooltip formatting
            title_parts = [f"<b>Type:</b> {node_type}", f"<b>Name:</b> {name}"]
            if code:
                title_parts.append(f"<b>Code:</b> {code}")
            if desc:
                title_parts.append(f"<b>Description:</b> {desc}")
            
            tooltip = "<br>".join(title_parts)

            nodes_data.append({
                "id": node,
                "label": name,
                "title": tooltip,
                "group": node_type,
                "color": {
                    "background": color,
                    "border": "#FFFFFF",
                    "highlight": {
                        "background": color,
                        "border": "#E2E8F0"
                    }
                },
                "font": {"color": "#F8FAFC", "face": "Inter", "size": 14, "bold": True},
                "shape": "dot",
                "size": 25,
            })

        edges_data = []
        for u, v, data in graph.edges(data=True):
            edge_type = data.get("type", "links")
            edges_data.append({
                "from": u,
                "to": v,
                "label": edge_type,
                "font": {"size": 11, "color": "#94A3B8", "align": "horizontal", "face": "Inter"},
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
                "color": {"color": "#475569", "highlight": "#64748B", "hover": "#64748B"},
                "width": 1.5,
                "smooth": {"type": "curvedCW", "roundness": 0.15}
            })

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Knowledge Graph Explorer</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
    <!-- Vis.js CDN -->
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #0B0F19;
            color: #E2E8F0;
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        header {{
            background: rgba(17, 24, 39, 0.7);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }}
        header h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60A5FA, #A78BFA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .header-subtitle {{
            font-size: 0.85rem;
            color: #94A3B8;
        }}
        #app-container {{
            flex: 1;
            display: flex;
            position: relative;
            height: calc(100vh - 70px);
        }}
        #network-container {{
            flex: 1;
            height: 100%;
            background-color: #080C14;
        }}
        /* Sidebar Glassmorphism */
        .glass-panel {{
            position: absolute;
            top: 20px;
            left: 20px;
            width: 350px;
            max-height: calc(100% - 40px);
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 24px;
            color: #F8FAFC;
            z-index: 5;
            display: flex;
            flex-direction: column;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            overflow-y: auto;
        }}
        .panel-section {{
            margin-bottom: 20px;
        }}
        .panel-section:last-child {{
            margin-bottom: 0;
        }}
        .panel-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 12px;
            color: #93C5FD;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 6px;
        }}
        /* Search Box */
        .search-box {{
            width: 100%;
            padding: 10px 14px;
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            color: #FFF;
            font-family: inherit;
            outline: none;
            transition: all 0.2s;
        }}
        .search-box:focus {{
            border-color: #60A5FA;
            box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2);
        }}
        /* Filter Pills */
        .filter-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .filter-pill {{
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            border: 1px solid transparent;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .filter-pill.active {{
            opacity: 1;
            border-color: rgba(255, 255, 255, 0.3);
        }}
        .filter-pill:not(.active) {{
            opacity: 0.4;
            background: rgba(30, 41, 59, 0.4) !important;
            color: #94A3B8 !important;
        }}
        /* Details Card */
        #details-card {{
            background: rgba(30, 41, 59, 0.3);
            border-radius: 12px;
            padding: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.9rem;
            line-height: 1.5;
        }}
        #details-card h3 {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            margin-bottom: 8px;
            color: #F8FAFC;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .property-row {{
            margin-top: 8px;
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}
        .property-label {{
            font-size: 0.75rem;
            color: #94A3B8;
            font-weight: 600;
        }}
        .property-value {{
            color: #E2E8F0;
        }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>🧠 Medical Knowledge Graph</h1>
            <span class="header-subtitle">Vietnamese Clinical Reasoning Engine</span>
        </div>
        <div style="font-size: 0.85rem; color: #64748B;">NetworkX Engine • Interactive Visualizer</div>
    </header>

    <div id="app-container">
        <!-- Floating Details Panel -->
        <div class="glass-panel">
            <div class="panel-section">
                <div class="panel-title">Search Entities</div>
                <input type="text" id="search-input" class="search-box" placeholder="Search node by name...">
            </div>

            <div class="panel-section">
                <div class="panel-title">Filter by Category</div>
                <div class="filter-container" id="filters">
                    <!-- Dynamic Filters added via JS -->
                </div>
            </div>

            <div class="panel-section">
                <div class="panel-title">Entity Details</div>
                <div id="details-card">
                    <div style="color: #94A3B8; text-align: center; padding: 20px 0;">Click on a node or edge in the graph to view clinical details.</div>
                </div>
            </div>
        </div>

        <!-- Network Container -->
        <div id="network-container"></div>
    </div>

    <script>
        // Graph raw data embedded from Python
        const nodesData = {json.dumps(nodes_data)};
        const edgesData = {json.dumps(edges_data)};
        const nodeColors = {json.dumps(cls.NODE_COLORS)};

        const container = document.getElementById('network-container');
        const searchInput = document.getElementById('search-input');
        const filtersContainer = document.getElementById('filters');
        const detailsCard = document.getElementById('details-card');

        // Setup Vis.js datasets
        const nodes = new vis.DataSet(nodesData);
        const edges = new vis.DataSet(edgesData);

        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            nodes: {{
                scaling: {{
                    min: 10,
                    max: 40
                }}
            }},
            edges: {{
                smooth: {{
                    enabled: true,
                    type: "dynamic"
                }}
            }},
            physics: {{
                solver: "forceAtlas2Based",
                forceAtlas2Based: {{
                    gravitationalConstant: -70,
                    centralGravity: 0.01,
                    springLength: 130,
                    springConstant: 0.08
                }},
                stabilization: {{
                    iterations: 150,
                    updateInterval: 25
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 150
            }}
        }};

        const network = new vis.Network(container, data, options);

        // Track active filters
        const activeFilters = new Set(Object.keys(nodeColors));

        // Initialize Filter pills
        function initFilters() {{
            filtersContainer.innerHTML = '';
            Object.keys(nodeColors).forEach(type => {{
                const pill = document.createElement('div');
                pill.className = 'filter-pill active';
                pill.style.backgroundColor = nodeColors[type];
                pill.style.color = '#0B0F19';
                pill.innerHTML = `<span>●</span> ${{type}}`;
                pill.onclick = () => toggleFilter(type, pill);
                filtersContainer.appendChild(pill);
            }});
        }}

        function toggleFilter(type, element) {{
            if (activeFilters.has(type)) {{
                activeFilters.delete(type);
                element.classList.remove('active');
            }} else {{
                activeFilters.add(type);
                element.classList.add('active');
            }}
            applyFiltersAndSearch();
        }}

        function applyFiltersAndSearch() {{
            const searchVal = searchInput.value.toLowerCase().trim();
            const filteredNodes = nodesData.filter(node => {{
                const typeMatches = activeFilters.has(node.group);
                const searchMatches = !searchVal || node.label.toLowerCase().includes(searchVal);
                return typeMatches && searchMatches;
            }});

            const hiddenNodeIds = nodesData
                .filter(node => !filteredNodes.includes(node))
                .map(node => node.id);

            nodes.forEach(node => {{
                const isHidden = hiddenNodeIds.includes(node.id);
                nodes.update({{ id: node.id, hidden: isHidden }});
            }});
        }}

        // Search trigger
        searchInput.addEventListener('input', applyFiltersAndSearch);

        // Network selection handler
        network.on("selectNode", function (params) {{
            const nodeId = params.nodes[0];
            const node = nodesData.find(n => n.id === nodeId);
            if (!node) return;

            // Get connected relationships
            const connectedEdges = edgesData.filter(e => e.from === nodeId || e.to === nodeId);
            let relsHtml = '';
            connectedEdges.forEach(e => {{
                const fromNode = nodesData.find(n => n.id === e.from);
                const toNode = nodesData.find(n => n.id === e.to);
                const relationColor = nodeColors[toNode.group] || '#FFF';
                relsHtml += `
                    <div style="margin-top: 6px; padding: 6px; background: rgba(255,255,255,0.03); border-radius: 6px; font-size: 0.8rem;">
                        <strong>${{fromNode.label}}</strong> 
                        <span style="color: #94A3B8;">${{e.label}}</span> 
                        <strong style="color: ${{relationColor}}">${{toNode.label}}</strong>
                    </div>
                `;
            }});

            const descText = node.title.split('<b>Description:</b> ');
            const desc = descText.length > 1 ? descText[1] : 'No description available.';
            const codeText = node.title.split('<b>Code:</b> ');
            const code = codeText.length > 1 ? codeText[1].split('<br>')[0] : 'N/A';

            detailsCard.innerHTML = `
                <span class="badge" style="background-color: \${{nodeColors[node.group]}}; color: #0B0F19;">\${{node.group}}</span>
                <h3>\${{node.label}}</h3>
                <div class="property-row">
                    <div class="property-label">Canonical ID</div>
                    <div class="property-value" style="font-family: monospace; font-size: 0.8rem;">\${{node.id}}</div>
                </div>
                <div class="property-row">
                    <div class="property-label">Code / Synonym</div>
                    <div class="property-value">\${{code}}</div>
                </div>
                <div class="property-row" style="margin-top: 10px;">
                    <div class="property-label">Clinical Description</div>
                    <div class="property-value" style="font-size: 0.85rem; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 8px;">\${{desc}}</div>
                </div>
                <div class="property-row" style="margin-top: 15px;">
                    <div class="property-label" style="border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 4px;">Direct Connections (\${{connectedEdges.length}})</div>
                    \${{relsHtml || '<div style="color: #64748B; font-style: italic;">No connections</div>'}}
                </div>
            `;
        }});

        network.on("deselectNode", function () {{
            detailsCard.innerHTML = `
                <div style="color: #94A3B8; text-align: center; padding: 20px 0;">Click on a node or edge in the graph to view clinical details.</div>
            `;
        }});

        // Init page
        initFilters();
    </script>
</body>
</html>
"""
        output_path.write_text(html_template, encoding="utf-8")
        return output_path
