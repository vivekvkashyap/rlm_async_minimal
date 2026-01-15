"""
Streamlit Dashboard for RLM Execution Trace Visualization.

Run with: streamlit run visualize_traces.py
"""

import streamlit as st
import json
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import pandas as pd


def load_trace_files(traces_dir="traces"):
    """Load all available trace files."""
    trace_dir = Path(traces_dir)
    if not trace_dir.exists():
        return []
    
    trace_files = list(trace_dir.glob("trace_*.json"))
    return sorted(trace_files, key=lambda x: x.stat().st_mtime, reverse=True)


def load_trace_data(trace_file):
    """Load trace data from JSON file."""
    with open(trace_file, 'r') as f:
        return json.load(f)


def create_tree_graph(trace_data):
    """Create a professional tree graph with RECURSIVE support for any depth level."""
    nodes = trace_data['nodes']
    tree_structure = trace_data['tree_structure']
    
    # Bold, distinct color palette for different depths
    DEPTH_COLORS = {
        0: '#FF9800',  # Bold orange for Root RLM (depth=0)
        1: '#4CAF50',  # Bold green for Sub-Root RLM (depth=1)
        2: '#2196F3',  # Bold blue for depth=2
        3: '#9C27B0',  # Bold purple for depth=3
        4: '#E91E63',  # Pink for depth=4
        5: '#00BCD4',  # Cyan for depth=5
    }
    
    # Border colors for depth highlighting
    DEPTH_BORDERS = {
        0: '#E65100',  # Dark orange border for root
        1: '#1B5E20',  # Dark green border for depth 1
        2: '#0D47A1',  # Dark blue border for depth 2
        3: '#4A148C',  # Dark purple for depth 3
        4: '#880E4F',  # Dark pink for depth 4
        5: '#006064',  # Dark cyan for depth 5
    }
    
    # Layout constants
    BOX_HEIGHT = 0.7
    BOX_SPACING = 0.3
    ROOT_BOX_WIDTH = 1.6
    SUB_BOX_WIDTH = 1.8
    HORIZONTAL_SPACING = 2.3
    
    # This will hold all visual elements
    all_boxes = []
    
    # First, extract the root node info
    roots = tree_structure['roots']
    if not roots:
        return go.Figure(), []
    
    root_node = roots[0]
    root_id = root_node['id']
    root_data = nodes[root_id]
    root_iterations = root_data.get('iterations', [])
    num_root_iters = len(root_iterations) if root_iterations else 1
    
    def get_parent_iteration_from_id(node_id):
        """Parse spawn_id to get parent iteration."""
        if '.' in node_id:
            parent_part = node_id.split('.')[0]
            parts = parent_part.split('-')
        else:
            parts = node_id.split('-')
        if len(parts) >= 1:
            try:
                return int(parts[0])
            except:
                return 0
        return 0
    
    def get_children_by_parent_iter(tree_node):
        """Group children of a node by which iteration spawned them."""
        children = tree_node.get('children', [])
        children_by_iter = {}
        for child in children:
            spawn_iter = get_parent_iteration_from_id(child['id'])
            if spawn_iter not in children_by_iter:
                children_by_iter[spawn_iter] = []
            children_by_iter[spawn_iter].append(child)
        return children_by_iter
    
    def process_sub_llms_recursive(sub_llm_nodes, start_x, start_y, depth_level):
        """
        Process sub-LLMs: place them HORIZONTALLY next to each other,
        with their iterations stacked VERTICALLY below.
        Returns the total height used.
        """
        if not sub_llm_nodes:
            return 0
        
        max_height = 0
        current_x = start_x
        
        for sub_node in sub_llm_nodes:
            node_id = sub_node['id']
            if node_id not in nodes:
                continue
            
            node_data = nodes[node_id]
            sub_iterations = node_data.get('iterations', [])
            num_iters = len(sub_iterations) if sub_iterations else 1
            
            # Get this sub-LLM's children grouped by iteration
            children_by_iter = get_children_by_parent_iter(sub_node)
            
            # Track height for this sub-LLM column
            column_y = start_y
            column_max_height = 0
            
            for iter_idx in range(num_iters):
                # Create box for this iteration
                box = {
                    'x': current_x,
                    'y': -column_y,
                    'name': node_data['instance_name'],
                    'iter': iter_idx,
                    'depth': depth_level,
                    'node_id': node_id,
                    'type': 'sub_iter',
                    'is_first_iter': iter_idx == 0
                }
                all_boxes.append(box)
                
                # Check if this iteration has children
                iter_children = children_by_iter.get(iter_idx, [])
                
                if iter_children:
                    # Recursively process children (place them to the right)
                    child_height = process_sub_llms_recursive(
                        iter_children,
                        current_x + HORIZONTAL_SPACING,
                        column_y,
                        depth_level + 1
                    )
                    column_y += max(child_height, BOX_HEIGHT + BOX_SPACING)
                else:
                    column_y += BOX_HEIGHT + BOX_SPACING
            
            column_max_height = column_y - start_y
            max_height = max(max_height, column_max_height)
            
            # Move to next horizontal position for next sub-LLM
            current_x += HORIZONTAL_SPACING
        
        return max_height
    
    # Get root's children grouped by iteration
    root_children_by_iter = get_children_by_parent_iter(root_node)
    
    # Current Y position tracker
    current_y = 0
    
    # For each Root LLM iteration
    for iter_idx in range(num_root_iters):
        # Create Root LLM iteration box
        root_box = {
            'x': 0,
            'y': -current_y,
            'name': 'Root LLM',
            'iter': iter_idx,
            'depth': 0,
            'node_id': root_id,
            'type': 'root_iter',
            'is_first_iter': iter_idx == 0
        }
        all_boxes.append(root_box)
        
        # Get sub-LLMs spawned in this iteration
        iter_sub_llms = root_children_by_iter.get(iter_idx, [])
        
        if iter_sub_llms:
            # Process sub-LLMs (they go to the right)
            sub_height = process_sub_llms_recursive(
                iter_sub_llms,
                HORIZONTAL_SPACING,  # Start X position for sub-LLMs
                current_y,
                1  # depth level
            )
            current_y += max(sub_height, BOX_HEIGHT + BOX_SPACING) + 0.5
        else:
            current_y += BOX_HEIGHT + BOX_SPACING + 0.3
    
    # Build node_info for hover
    node_info = {}
    for node_id, node_data in nodes.items():
        duration = node_data.get('duration', 0)
        node_info[node_id] = {
            'name': node_data['instance_name'],
            'depth': node_data['depth'],
            'max_depth': node_data['max_depth'],
            'iterations': len(node_data.get('iterations', [])),
            'duration': f"{duration:.2f}s" if duration else "N/A",
            'status': node_data['status'],
        }
    
    # Create edge traces
    edge_traces = []
    
    # Draw vertical edges (between iterations of same node)
    for node_id in set(b['node_id'] for b in all_boxes):
        node_boxes = sorted([b for b in all_boxes if b['node_id'] == node_id], key=lambda x: x['iter'])
        for i in range(len(node_boxes) - 1):
            edge_traces.append(go.Scatter(
                x=[node_boxes[i]['x'], node_boxes[i+1]['x']],
                y=[node_boxes[i]['y'] - BOX_HEIGHT/2 - 0.1, node_boxes[i+1]['y'] + BOX_HEIGHT/2 + 0.1],
                mode='lines',
                line=dict(color='#212121', width=3),
                hoverinfo='none',
                showlegend=False
            ))
    
    # Draw horizontal edges (from parent iteration to child's first iteration)
    # Root -> Sub-Root connections
    root_boxes = [b for b in all_boxes if b['type'] == 'root_iter']
    for root_box in root_boxes:
        root_iter = root_box['iter']
        # Find sub-LLMs spawned from this root iteration
        sub_first_boxes = [b for b in all_boxes if b['depth'] == 1 and b.get('is_first_iter', False)]
        for sub_box in sub_first_boxes:
            # Check if this sub-LLM was spawned from this root iteration
            spawn_iter = get_parent_iteration_from_id(sub_box['node_id'])
            if spawn_iter == root_iter:
                edge_traces.append(go.Scatter(
                    x=[root_box['x'] + ROOT_BOX_WIDTH/2 + 0.1, sub_box['x'] - SUB_BOX_WIDTH/2 - 0.1],
                    y=[root_box['y'], sub_box['y']],
                    mode='lines',
                    line=dict(color='#212121', width=3),
                    hoverinfo='none',
                    showlegend=False
                ))
    
    # Create figure
    fig = go.Figure(edge_traces)
    
    # Add boxes as shapes
    shapes = []
    annotations = []
    
    for box in all_boxes:
        depth = box['depth']
        color = DEPTH_COLORS.get(depth, DEPTH_COLORS.get(depth % 6, '#CFD8DC'))
        border_color = DEPTH_BORDERS.get(depth, DEPTH_BORDERS.get(depth % 6, '#607D8B'))
        
        current_box_width = ROOT_BOX_WIDTH if box['depth'] == 0 else SUB_BOX_WIDTH
        
        shapes.append(dict(
            type="rect",
            x0=box['x'] - current_box_width/2, y0=box['y'] - BOX_HEIGHT/2,
            x1=box['x'] + current_box_width/2, y1=box['y'] + BOX_HEIGHT/2,
            fillcolor=color,
            line=dict(color=border_color, width=3),
            opacity=1
        ))
        
        if box['depth'] == 0:
            text = f"<b style='font-size:12px;'>{box['name']}</b><br><b style='font-size:11px;'>ITER={box['iter']}</b>"
        else:
            if box.get('is_first_iter', False):
                text = f"<b style='font-size:10px;'>{box['name']}</b><br><b style='font-size:10px;'>ITER={box['iter']}</b>"
            else:
                text = f"<b style='font-size:10px;'>ITER={box['iter']}</b>"
        
        annotations.append(dict(
            x=box['x'], y=box['y'],
            text=text,
            showarrow=False,
            font=dict(size=11, color='#FFFFFF', family='Arial Black, sans-serif'),
            align='center'
        ))
    
    # Add hover points
    node_x = [box['x'] for box in all_boxes]
    node_y = [box['y'] for box in all_boxes]
    hover_text = [f"<b>{box['name']}</b><br>Iteration: {box['iter']}<br>Depth: {box['depth']}" for box in all_boxes]
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(size=20, opacity=0, color='rgba(0,0,0,0)'),
        hovertext=hover_text,
        hoverinfo='text',
        showlegend=False
    ))
    
    # Calculate x range for layout
    max_x = max([box['x'] for box in all_boxes]) if all_boxes else 5
    min_y = min([box['y'] for box in all_boxes]) if all_boxes else -5
    
    # Update layout for professional appearance
    fig.update_layout(
        title=dict(
            text="<b>RLM Execution Tree - Iteration View</b>",
            font=dict(size=22, color='#1a237e', family='Arial Black, sans-serif'),
            x=0.5,
            xanchor='center',
            y=0.97,
            yanchor='top'
        ),
        shapes=shapes,
        annotations=annotations,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=50, l=60, r=60, t=80),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            fixedrange=False,
            range=[-1.5, max_x + 2]
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            fixedrange=False,
            range=[min_y - 1, 1]
        ),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#ffffff',
        height=700,
        dragmode='pan'
    )
    
    # Return node IDs for selection
    node_ids = list(nodes.keys())
    return fig, node_ids


def display_node_details(node_id, trace_data):
    """Display detailed information about a selected node with complete text."""
    nodes = trace_data['nodes']
    if node_id not in nodes:
        st.error(f"Node {node_id} not found")
        return
    
    node = nodes[node_id]
    
    # Header with styling
    st.markdown(f"<h2 style='color: #2c3e50;'>📊 {node['instance_name']}</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Basic info with better layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔢 Depth", f"{node['depth']}/{node['max_depth']}")
    with col2:
        st.metric("🔄 Total Iterations", len(node.get('iterations', [])))
    with col3:
        duration = node.get('duration', 0)
        st.metric("⏱️ Duration", f"{duration:.2f}s" if duration else "N/A")
    with col4:
        status = node['status']
        status_emoji = "✅" if status == "completed" else "❌" if status == "error" else "🔄"
        st.metric("📊 Status", f"{status_emoji} {status.upper()}")
    
    st.markdown("---")
    
    iterations = node.get('iterations', [])
    
    # For Root LLM (depth=0): Show Query once at the top
    if node['depth'] == 0 and iterations:
        # Get query from first iteration (it's the same for all)
        first_iter = iterations[0]
        if first_iter.get('prompt'):
            st.markdown("### 📝 Query")
            with st.container():
                st.markdown(f"<div style='padding: 15px; background: #1565C0; color: #FFFFFF; border-radius: 8px; border-left: 4px solid #0D47A1; font-family: monospace; white-space: pre-wrap;'>{first_iter['prompt']}</div>", unsafe_allow_html=True)
            st.markdown("---")
    
    # Final answer with better formatting
    if node.get('final_answer'):
        st.markdown("### 🎯 Final Answer")
        with st.container():
            st.markdown(f"<div style='padding: 15px; background: #2E7D32; color: #FFFFFF; border-radius: 8px; border-left: 4px solid #1B5E20; font-family: monospace; white-space: pre-wrap;'>{node['final_answer']}</div>", unsafe_allow_html=True)
        st.markdown("---")
    
    # Iterations with complete text
    st.markdown("### 🔄 Iteration Trace")
    
    if not iterations:
        st.info("📝 No iterations recorded for this node")
        return
    
    for iter_data in iterations:
        iter_num = iter_data['iteration_number']
        with st.expander(f"🔸 Iteration {iter_num + 1} of {len(iterations)}", expanded=(iter_num == 0)):
            
            # For Sub-LLMs: Show their prompt (which is different per sub-LLM)
            if node['depth'] > 0 and iter_data.get('prompt'):
                st.markdown("#### 📝 Sub-LLM Prompt")
                with st.container():
                    st.markdown(f"<div style='padding: 10px; background: #E65100; color: #FFFFFF; border-radius: 5px; font-family: monospace; white-space: pre-wrap;'>{iter_data['prompt']}</div>", unsafe_allow_html=True)
                st.markdown("")
            
            # Response - Full text
            st.markdown("#### 💬 Model Response")
            response = iter_data.get('response', '')
            if response:
                with st.container():
                    # Display response with dark background and white text
                    st.markdown(f"<div style='padding: 12px; background: #37474F; color: #ECEFF1; border-left: 4px solid #78909C; border-radius: 5px; font-family: monospace; white-space: pre-wrap;'>{response}</div>", unsafe_allow_html=True)
            else:
                st.info("No response recorded")
            st.markdown("")
            
            # Code executions - Full text
            code_execs = iter_data.get('code_executions', [])
            if code_execs:
                st.markdown("#### 💻 Code Execution")
                for i, code_exec in enumerate(code_execs, 1):
                    st.markdown(f"**Execution Block {i} of {len(code_execs)}:**")
                    with st.container():
                        # Full code display
                        st.code(code_exec['code'], language='python')
                        
                        # Full stdout
                        if code_exec['stdout']:
                            st.markdown("**📤 Output:**")
                            with st.container():
                                st.text(code_exec['stdout'])
                        
                        # Full stderr
                        if code_exec['stderr']:
                            st.markdown("**⚠️ Error Output:**")
                            with st.container():
                                st.error(code_exec['stderr'])
                        
                        # Execution time
                        exec_time = code_exec.get('execution_time', 0)
                        st.caption(f"⏱️ Execution completed in {exec_time:.4f} seconds")
                    
                    if i < len(code_execs):
                        st.markdown("---")
            
            # Sub-LLM calls
            sub_calls = iter_data.get('sub_llm_calls', [])
            if sub_calls:
                st.markdown("#### 🔗 Spawned Sub-LLMs")
                # Display as a nice list
                for sub_call in sub_calls:
                    st.markdown(f"- `{sub_call}`")
            
            st.markdown("<br>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="RLM Execution Visualizer", layout="wide", page_icon="🌳")
    
    st.title("🌳 Recursive Language Model Execution Tree Visualizer")
    st.markdown("""
    <p style='font-size: 16px; color: #555;'>
    Interactive visualization dashboard for exploring RLM execution traces with multi-depth recursion.
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar for trace file selection
    st.sidebar.header("📁 Select Trace File")
    st.sidebar.markdown("Choose a trace file to visualize the execution tree and detailed information.")
    
    trace_files = load_trace_files()
    
    if not trace_files:
        st.warning("⚠️ No trace files found in the 'traces/' directory.")
        st.info("📋 Run your RLM with trace logging enabled to generate trace files.")
        st.markdown("### How to Generate Trace Files")
        st.code("""
from rlm.logger.trace_logger import init_trace_logger
from rlm.rlm_repl import RLM_REPL

# Initialize trace logger
init_trace_logger(output_dir="traces", enabled=True)

# Run your RLM
rlm = RLM_REPL(...)
result = rlm.completion(...)

# Save trace
from rlm.logger.trace_logger import get_trace_logger
get_trace_logger().save_trace()
        """, language='python')
        return
    
    # File selector
    trace_file_names = [f.name for f in trace_files]
    selected_file_name = st.sidebar.selectbox(
        "Select trace file:",
        trace_file_names,
        format_func=lambda x: f"{x} ({datetime.fromtimestamp(next(f for f in trace_files if f.name == x).stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})"
    )
    
    selected_file = next(f for f in trace_files if f.name == selected_file_name)
    
    # Load trace data
    trace_data = load_trace_data(selected_file)
    
    # Display session info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Session Information")
    st.sidebar.markdown(f"**🆔 Session ID:**  \n`{trace_data['session_id']}`")
    st.sidebar.markdown(f"**⏱️ Total Duration:**  \n`{trace_data['total_duration']:.2f} seconds`")
    st.sidebar.markdown(f"**🔢 Total Nodes:**  \n`{len(trace_data['nodes'])}`")
    
    depths = [n['depth'] for n in trace_data['nodes'].values()]
    max_depth = max(depths) if depths else 0
    st.sidebar.markdown(f"**📏 Maximum Depth:**  \n`{max_depth}`")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 Color Legend")
    st.sidebar.markdown("""
🟧 **Orange** — Root LLM (depth=0)

🟩 **Green** — Sub Root LLM (depth=1)

🟦 **Blue** — Sub-Sub LLM (depth=2)

🟪 **Purple** — depth=3

🩷 **Pink** — depth=4

🩵 **Cyan** — depth=5+
    """)
    st.sidebar.markdown("")
    st.sidebar.caption("💡 Each box = one iteration. Supports unlimited depth!")
    
    # Main content: Tree visualization
    st.markdown("## 🌲 Execution Tree Visualization")
    st.markdown("""
    <p style='color: #555; font-size: 14px;'>
    The tree displays the hierarchical structure of RLM execution. Root LLM iterations appear vertically, 
    while Sub-RLMs branch horizontally to the right. Hover over any node for quick information, 
    or use the dropdown below to view complete execution details.
    </p>
    """, unsafe_allow_html=True)
    
    fig, node_ids = create_tree_graph(trace_data)
    selected_points = st.plotly_chart(fig, width='stretch', on_select="rerun", key="tree")
    
    # Node selection
    st.markdown("---")
    st.markdown("## 🔍 Node Details Inspector")
    st.markdown("Select a node below to view its complete execution trace, including prompts, model responses, code executions, and outputs.")
    
    # Dropdown for manual node selection
    node_list = list(node_ids)
    node_display_names = [f"{trace_data['nodes'][nid]['instance_name']} (ID: {nid})" for nid in node_list]
    
    selected_node_display = st.selectbox(
        "🎯 Choose a node to inspect:",
        node_display_names,
        index=0
    )
    
    # Extract node ID from selection
    selected_node_id = selected_node_display.split("(ID: ")[1].rstrip(")")
    
    # Display node details
    display_node_details(selected_node_id, trace_data)
    
    # Statistics
    st.markdown("---")
    st.markdown("## 📈 Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    nodes = trace_data['nodes']
    completed_nodes = [n for n in nodes.values() if n['status'] == 'completed']
    error_nodes = [n for n in nodes.values() if n['status'] == 'error']
    
    with col1:
        st.metric("Total Nodes", len(nodes))
        st.metric("Completed", len(completed_nodes))
    
    with col2:
        total_iterations = sum(len(n['iterations']) for n in nodes.values())
        st.metric("Total Iterations", total_iterations)
        st.metric("Errors", len(error_nodes))
    
    with col3:
        depths = [n['depth'] for n in nodes.values()]
        max_depth_reached = max(depths) if depths else 0
        st.metric("Max Depth Reached", max_depth_reached)
        avg_duration = sum(n.get('duration', 0) for n in nodes.values() if n.get('duration')) / len(nodes)
        st.metric("Avg Duration/Node", f"{avg_duration:.2f}s")


if __name__ == "__main__":
    main()

