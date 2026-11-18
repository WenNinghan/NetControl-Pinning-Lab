"""
Interactive Streamlit Workbench for NetControl-Pinning-Lab:
Structural Controllability, Submodular Gramian Actuator Selection,
and Differentiable Graph Neural Pinning Control.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from netcontrol import (
    analyze_structural_controllability,
    build_continuous_system_matrix,
    classify_control_edges,
    erdos_renyi_directed,
    minimum_energy_trajectory,
    reaction_cascade_network,
    scale_free_directed,
    select_actuators_baseline,
    select_actuators_submodular,
    simulate_nonlinear_pinning,
    small_world_directed,
    solve_lqr_controller,
    train_neural_pinning_controller,
    verify_pbh_controllability,
)
from netcontrol.gramian_lqr import build_input_matrix


st.set_page_config(
    page_title="NetControl-Pinning-Lab | Complex Network Control Workbench",
    layout="wide",
)

st.title("NetControl-Pinning-Lab: 复杂网络结构可控性与神经牵制控制工作台")
st.markdown(
    "在合成有向网络上比较执行器接线、格拉姆矩阵选点和闭环反馈。"
    "结构分析使用网络的稀疏连接模式，轨迹仿真另使用带衰减的动力学矩阵。"
)

with st.sidebar:
    st.header("1. 网络拓扑参数")
    topo_type = st.selectbox(
        "选择复杂网络模型",
        ["Reaction Cascade (合成反应级联)", "Scale-Free (有向无标度网络)", "Erdős–Rényi (随机有向图)", "Small-World (Watts-Strogatz)"],
    )
    n_nodes = st.slider("节点规模 N", min_value=12, max_value=45, value=24, step=2)
    mean_deg = st.slider("平均有向度 / 分支因子 <k>", min_value=1.2, max_value=4.5, value=2.3, step=0.1)
    seed = st.number_input("随机种子 (Seed)", min_value=0, max_value=9999, value=42)

    st.header("2. 牵制控制与优化配置")
    budget_k = st.slider("牵制控制节点预算 K = |P|", min_value=2, max_value=min(16, n_nodes), value=min(7, n_nodes))
    regime = st.selectbox("非线性耦合动力学方程", ["pollutant_cascade", "bistable_sync"])
    epochs = st.slider("可微 ODE 图神经控制器训练轮数", min_value=15, max_value=80, value=35, step=5)

if topo_type.startswith("Reaction"):
    G = reaction_cascade_network(n=n_nodes, branch_factor=mean_deg, seed=int(seed))
elif topo_type.startswith("Scale-Free"):
    G = scale_free_directed(n=n_nodes, mean_degree=mean_deg, gamma_exp=2.3, seed=int(seed))
elif topo_type.startswith("Erdős"):
    G = erdos_renyi_directed(n=n_nodes, mean_degree=mean_deg, seed=int(seed))
else:
    G = small_world_directed(n=n_nodes, k_out=max(1, int(round(mean_deg))), p_rewire=0.25, seed=int(seed))

sc_res = analyze_structural_controllability(G)
edge_classes = classify_control_edges(G)
pbh_ok, pbh_rank = verify_pbh_controllability(G, sc_res.driver_nodes, seed=int(seed))
A = build_continuous_system_matrix(G, damping=0.55, coupling=0.36)

col1, col2, col3, col4 = st.columns(4)
col1.metric("节点总数 N / 有向边数 E", f"{sc_res.n_nodes} / {sc_res.n_edges}")
col2.metric("专用执行器位置（每处一路）", sc_res.n_drivers)
col3.metric("独立输入通道（允许分接）", sc_res.minimum_input_channels)
col4.metric("专用接线的 PBH 最小秩", f"{pbh_rank} / {sc_res.n_nodes}" + (" ✓" if pbh_ok else ""))
st.caption(
    "PBH 使用当前种子的边权样本，按指定位置接线。下方的牵制预算 K 是单独的仿真实验设置。"
)
with st.expander("查看执行器位置与共享信号接线"):
    st.write("专用执行器节点", sc_res.driver_nodes)
    for channel, sites in enumerate(sc_res.shared_input_sites, start=1):
        st.text(f"共享方案 · 通道 {channel} 接到节点 {', '.join(map(str, sites))}")

tab1, tab2 = st.tabs(["非线性牵制控制仿真 (Neural Pinning vs LQR)", "结构可控性图谱与子模执行器优选"])

with tab1:
    pinned_sub, m_sub = select_actuators_submodular(
        A, budget_k=budget_k, horizon=2.5, mandatory_nodes=sc_res.driver_nodes[: min(budget_k, len(sc_res.driver_nodes))]
    )
    pinned_rand, _ = select_actuators_baseline(G, A, budget_k=budget_k, strategy="random", seed=int(seed) + 1)
    B_pin = build_input_matrix(n_nodes, pinned_sub)
    K_lqr, _ = solve_lqr_controller(A, B_pin, q_weight=3.0, r_weight=0.08)

    rng = np.random.default_rng(int(seed))
    x0 = rng.uniform(1.2, 2.6, size=n_nodes)

    with st.spinner("正在通过可微 RK4 轨迹优化训练 GraphNeuralPinningController..."):
        gnn_ctrl, losses = train_neural_pinning_controller(
            G=G, A=A, pinned_nodes=pinned_sub, driver_nodes=sc_res.driver_nodes,
            x0=x0, horizon=4.0, n_steps=70, epochs=epochs, lr=0.035, regime=regime, seed=int(seed),
        )

    sim_unc = simulate_nonlinear_pinning(A, x0, pinned_sub, mode="uncontrolled", horizon=4.0, n_steps=70, regime=regime)
    sim_rand = simulate_nonlinear_pinning(A, x0, pinned_rand, mode="linear_pinning", horizon=4.0, n_steps=70, regime=regime)
    sim_lqr = simulate_nonlinear_pinning(A, x0, pinned_sub, mode="lqr_pinning", K_lqr=K_lqr, horizon=4.0, n_steps=70, regime=regime)
    sim_gnn = simulate_nonlinear_pinning(A, x0, pinned_sub, mode="gnn_pinning", gnn_controller=gnn_ctrl, horizon=4.0, n_steps=70, regime=regime)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ts = sim_gnn["t"]
    axes[0].plot(ts, sim_unc["state_norms"], label="Uncontrolled", linestyle=":", color="#7f7f7f", linewidth=2)
    axes[0].plot(ts, sim_rand["state_norms"], label="Random + Linear Pinning", linestyle="--", color="#ff7f00", linewidth=2)
    axes[0].plot(ts, sim_lqr["state_norms"], label="Submodular + CARE LQR", linestyle="-.", color="#4daf4a", linewidth=2)
    axes[0].plot(ts, sim_gnn["state_norms"], label="Submodular + Graph Neural Pinning", color="#e41a1c", linewidth=2.5)
    axes[0].set_xlabel("Time t (s)")
    axes[0].set_ylabel("State Norm ||x(t)||_2")
    axes[0].set_title("State Norm Suppression Trajectory")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].plot(range(1, len(losses) + 1), losses, color="#6a3d9a", marker="o", markersize=4)
    axes[1].set_xlabel("Training Epoch")
    axes[1].set_ylabel("Differentiable Trajectory Cost J")
    axes[1].set_title("Graph Neural Pinning Controller Convergence")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig)

    records = []
    for name, sim in [
        ("无控制", sim_unc),
        ("随机选点 + 比例反馈", sim_rand),
        ("格拉姆选点 + CARE LQR", sim_lqr),
        ("格拉姆选点 + 图神经控制器", sim_gnn),
    ]:
        records.append({
            "控制策略": name,
            "状态误差积分": sim["ise_state"],
            "实际施加的输入能量": sim["applied_input_energy"],
            "计入 J 的采样能量": sim["control_energy"],
            "总代价 J": sim["total_j"],
            "终端状态范数": sim["terminal_norm"],
            "调节时间 (s)": sim["settling_time"],
        })
    st.caption("仿真在每个时间步内保持控制输入。实际能量按保持时间累加。J 沿用原实验评分，按梯形求积累加状态误差与采样控制量，再加终端惩罚。三项权重分别是 3、0.08、12。调节时间留空表示观测期内未达到阈值。")
    table = pd.DataFrame.from_records(records).set_index("控制策略")
    st.dataframe(table.style.format("{:.3f}", na_rep="—"), use_container_width=True)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(8.5, 4.8))
    pos = nx.spring_layout(G, seed=int(seed))
    driver_set = set(sc_res.driver_nodes)
    other_nodes = [u for u in G.nodes() if u not in driver_set]
    nx.draw_networkx_edges(G, pos, edgelist=edge_classes["ordinary"], ax=ax2, edge_color="#377eb8", alpha=0.5, arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=edge_classes["critical"], ax=ax2, edge_color="#e41a1c", width=2.0, arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=edge_classes["redundant"], ax=ax2, edge_color="#bdbdbd", style="dashed", arrows=True)
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, ax=ax2, node_color="#4daf4a", node_size=260, label="Other nodes")
    nx.draw_networkx_nodes(G, pos, nodelist=sc_res.driver_nodes, ax=ax2, node_color="#ff7f00", node_shape="^", node_size=340, label="Dedicated actuator sites")
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=8, font_color="#111111")
    ax2.set_title("Dedicated actuator sites on the structural network")
    ax2.legend()
    ax2.axis("off")
    st.pyplot(fig2)
    st.caption(f"关键边 / 普通边 / 冗余边（按匹配定义）：{len(edge_classes['critical'])} / {len(edge_classes['ordinary'])} / {len(edge_classes['redundant'])}。本次闭环仿真的选点为 {pinned_sub}。")

    st.subheader("线性系统的输入能量与终端误差")
    trajectory = minimum_energy_trajectory(A, B_pin, x0, np.zeros_like(x0), horizon=2.5, reg=1e-4)
    st.caption("用上面选出的 K 个位置，在 2.5 个时间单位内向零目标施加无幅值限制的开环输入。正则目标 = 输入能量 + 终端误差平方 / 0.0001。")
    energy_col, error_col, objective_col = st.columns(3)
    energy_col.metric("实际输入能量", f"{trajectory['input_energy']:.3f}")
    error_col.metric("终端误差范数", f"{trajectory['terminal_error']:.4g}")
    objective_col.metric("含终端惩罚的正则目标", f"{trajectory['regularized_objective']:.3f}")
    exact_energy = trajectory["exact_target_energy"]
    target_note = "在数值容差下该目标不可达" if exact_energy is None else f"精确达零所需最小能量 {exact_energy:.4g}"
    st.caption(f"可达维数 {trajectory['reachability_rank']}/{n_nodes}，{target_note}。")
