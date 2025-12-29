"""
Utilities for visualizing MCTS search trees.

This module converts recorded MCTS traces into
graph structures suitable for Streamlit rendering.

Supports:
- Static Graphviz rendering (DOT)
- Interactive PyVis rendering (HTML, zoomable, hierarchical)
"""

from typing import List, Dict
import networkx as nx

from pyvis.network import Network
import streamlit.components.v1 as components


# --------------------------------------------------
# Phase Color Palette
# --------------------------------------------------
PHASE_COLORS = {
    "Reconnaissance": "#1f77b4",
    "Weaponization": "#ff7f0e",
    "Delivery": "#2ca02c",
    "Exploitation": "#d62728",
    "Installation": "#9467bd",
    "Command_and_Control": "#8c564b",
    "Actions_on_Objectives": "#e377c2",
}

ROOT_COLOR = "#6a0dad"  # Purple for root


def darken(hex_color: str, factor: float = 0.7) -> str:
    """Return a darker shade of a hex color."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


# --------------------------------------------------
# Graph Construction
# --------------------------------------------------
def build_mcts_graph(
    trace: List[Dict],
    phase_order: List[str],
    ttp_label_map: Dict[str, str] | None = None,
) -> nx.DiGraph:
    """
    Build a directed graph from deduplicated MCTS trace records.
    """
    G = nx.DiGraph()

    for rec in trace:
        if 0 <= rec["phase_idx"] < len(phase_order):
            phase_name = phase_order[rec["phase_idx"]]
            level = rec["phase_idx"] + 1
        else:
            phase_name = "ROOT"
            level = 0

        ttp = rec["ttp_id"] or "START"
        ttp_name = ttp_label_map.get(ttp, ttp) if ttp_label_map else ttp

        G.add_node(
            rec["id"],
            phase=phase_name,
            level=level,
            ttp_id=ttp,
            ttp_name=ttp_name,
            visits=rec["N"],
            value=rec["Q"],
            prior=rec["P"],
        )

    for rec in trace:
        if rec["parent"] is not None:
            G.add_edge(rec["parent"], rec["id"])

    return G


# --------------------------------------------------
# Graph Summary
# --------------------------------------------------
def summarize_graph(G: nx.DiGraph) -> Dict[str, int]:
    """
    Compute basic tree statistics for UI display.
    """
    roots = [n for n in G.nodes if G.in_degree(n) == 0]

    def dfs_depth(node, depth=0):
        if G.out_degree(node) == 0:
            return depth
        return max(dfs_depth(c, depth + 1) for c in G.successors(node))

    max_depth = 0
    for r in roots:
        max_depth = max(max_depth, dfs_depth(r))

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "max_depth": max_depth,
    }


# --------------------------------------------------
# DOT Export (Static Graphviz)
# --------------------------------------------------
def to_dot(G: nx.DiGraph) -> str:
    """
    Convert the graph into a DOT string with styled nodes.
    """
    lines = [
        "digraph MCTS {",
        "rankdir=TB;",
        "ranksep=1.2;",
        "nodesep=0.6;",
        'size="10,30";',
        "dpi=200;",
    ]

    for n, data in G.nodes(data=True):
        phase = data["phase"]

        if phase == "ROOT":
            label = "MCTS Kill-Chain Inference"
            fill = ROOT_COLOR
            shape = "box"
        else:
            base = PHASE_COLORS.get(phase, "#cccccc")
            fill = darken(base) if data["visits"] > 5 else base
            shape = "circle"
            label = data["ttp_id"]

        lines.append(
            f'"{n}" [shape={shape}, style="filled", '
            f'fillcolor="{fill}", fontcolor="white", label="{label}"];'
        )

    for u, v in G.edges():
        lines.append(f'"{u}" -> "{v}" [color="gray"];')

    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------
# Interactive PyVis Export
# --------------------------------------------------
def show_interactive_tree(
    G: nx.DiGraph,
    height: str = "850px",
    width: str = "100%",
):
    """
    Render the full MCTS tree as an interactive PyVis graph
    with strict hierarchical layout.
    """

    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor="#0e1117",
        font_color="white",
    )

    net.set_options(
        """
        var options = {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "levelSeparation": 160,
              "nodeSpacing": 180
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "zoomView": true,
            "dragView": true
          },
          "physics": {
            "enabled": false
          },
          "edges": {
            "color": {
              "color": "#888888"
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.6
              }
            }
          }
        }
        """
    )

    # --- Add nodes
    for node_id, data in G.nodes(data=True):
        phase = data["phase"]

        if phase == "ROOT":
            color = ROOT_COLOR
            shape = "box"
            label = "MCTS Kill-Chain Inference"
        else:
            base = PHASE_COLORS.get(phase, "#cccccc")
            color = darken(base) if data["visits"] > 5 else base
            shape = "circle"
            label = data["ttp_id"]

        title = (
            f"Phase: {phase} | "
            f"TTP: {data['ttp_id']} | "
            f"Name: {data['ttp_name']} | "
            f"N={data['visits']} | "
            f"Q={data['value']:.3f} | "
            f"P={data['prior']:.3f}"
        )

        net.add_node(
            node_id,
            label=label,
            title=title,
            level=data["level"],
            size=30,
            shape=shape,
            color=color,
            font={"size": 14},
        )

    # --- Add edges
    for u, v in G.edges():
        net.add_edge(u, v, color="#888888")

    html_path = "mcts_tree_interactive.html"
    net.save_graph(html_path)

    components.html(
        open(html_path, "r", encoding="utf-8").read(),
        height=900,
        scrolling=True,
    )
