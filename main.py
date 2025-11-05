import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import random
from typing import Tuple, List
from tqdm import tqdm


class BaseGraphGenerator:
    """基础图生成器 - 生成固定的基础图结构"""
    
    def __init__(self, num_classes: int = 1, num_shape_classes: int = 3, seed: int = 42):
        """
        Args:
            num_classes: 类别数量
            num_shape_classes: 形状类别数量
            seed: 随机种子
        """
        self.num_classes = num_classes
        self.num_shape_classes = num_shape_classes
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 生成5个固定的基础图
        self.base_graphs = self._generate_base_graphs()
        print(self.base_graphs)
        print(f"已生成 {len(self.base_graphs)} 个基础图")
    
    def _generate_base_graphs(self) -> List[Data]:
        """生成5个固定的基础图"""
        base_graphs = []
        
        # 为5个基础图定义不同的节点数量
        node_counts = [20, 25, 30, 35, 40]
        
        for graph_id, num_nodes in enumerate(node_counts):
            # 使用固定的种子确保每个基础图一致
            np.random.seed(42 + graph_id)
            random.seed(42 + graph_id)
            
            # 生成节点特征
            coords = np.random.randn(num_nodes, 4).astype(np.float32)
            classes = np.random.randint(0, self.num_classes, (num_nodes, 1)).astype(np.float32)
            shape_classes = np.random.randint(0, self.num_shape_classes, (num_nodes, 1)).astype(np.float32)
            
            node_features = np.concatenate([coords, classes, shape_classes], axis=1)
            node_features = torch.tensor(node_features, dtype=torch.float32)
            
            # 生成固定的边结构
            edge_list = []
            edge_features_list = []
            
            # 创建最小生成树
            nodes = list(range(num_nodes))
            np.random.shuffle(nodes)
            for i in range(1, num_nodes):
                parent = nodes[np.random.randint(0, i)]
                child = nodes[i]
                edge_list.append([parent, child])
                edge_feat = np.random.randn(2).astype(np.float32)
                edge_features_list.append(edge_feat)
            
            # 添加额外的边
            num_extra_edges = num_nodes + graph_id * 5  # 每个图有不同数量的边
            for _ in range(num_extra_edges):
                src = np.random.randint(0, num_nodes)
                dst = np.random.randint(0, num_nodes)
                if src != dst and [src, dst] not in edge_list:
                    edge_list.append([src, dst])
                    edge_feat = np.random.randn(2).astype(np.float32)
                    edge_features_list.append(edge_feat)
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features_list, dtype=torch.float32)
            
            graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
            base_graphs.append(graph)
        
        return base_graphs
    
    def get_base_graph(self, index: int) -> Data:
        """获取指定的基础图"""
        return self.base_graphs[index % len(self.base_graphs)]


class GraphDataGenerator:
    """图数据生成器 - 基于基础图添加噪声生成变体"""
    
    def __init__(self, base_generator: BaseGraphGenerator,
                 coord_noise_std: float = 0.1,
                 edge_noise_std: float = 0.1,
                 seed: int = None):
        """
        Args:
            base_generator: 基础图生成器
            coord_noise_std: 坐标高斯噪声标准差
            edge_noise_std: 边特征高斯噪声标准差
            seed: 随机种子
        """
        self.base_generator = base_generator
        self.coord_noise_std = coord_noise_std
        self.edge_noise_std = edge_noise_std
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def generate_graph(self) -> Data:
        """
        从基础图生成变体（添加高斯噪声）
        节点特征：[坐标x1, 坐标x2, 坐标x3, 坐标x4, 类别, 形状类别]
        边特征：[向量x, 向量y]
        """
        # 随机选择一个基础图
        base_idx = random.randint(0, len(self.base_generator.base_graphs) - 1)
        base_graph = self.base_generator.get_base_graph(base_idx)
        
        # 复制基础图
        x = base_graph.x.clone()
        edge_index = base_graph.edge_index.clone()
        edge_attr = base_graph.edge_attr.clone()
        
        # 对节点坐标（前4列）添加高斯噪声
        coord_noise = np.random.normal(0, self.coord_noise_std, size=(x.size(0), 4)).astype(np.float32)
        x[:, :4] = x[:, :4] + torch.tensor(coord_noise, dtype=torch.float32)
        
        # 对边特征添加高斯噪声
        edge_noise = np.random.normal(0, self.edge_noise_std, size=edge_attr.shape).astype(np.float32)
        edge_attr = edge_attr + torch.tensor(edge_noise, dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class GraphOccluder:
    """图遮挡器 - 随机移除节点和边"""
    
    def __init__(self, node_occlusion_rate_range: Tuple[float, float] = (0.0, 0.5),
                 edge_occlusion_rate_range: Tuple[float, float] = (0.0, 0.5)):
        """
        Args:
            node_occlusion_rate_range: 节点遮挡率范围
            edge_occlusion_rate_range: 边遮挡率范围
        """
        self.node_occlusion_rate_range = node_occlusion_rate_range
        self.edge_occlusion_rate_range = edge_occlusion_rate_range
    
    def occlude_graph(self, graph: Data) -> Tuple[Data, float]:
        """
        对图进行随机遮挡
        Returns:
            (遮挡后的图, 完整度)
        """
        # 随机选择遮挡率
        node_occlusion_rate = random.uniform(*self.node_occlusion_rate_range)
        edge_occlusion_rate = random.uniform(*self.edge_occlusion_rate_range)
        
        num_nodes = graph.x.size(0)
        num_edges = graph.edge_index.size(1)
        
        # 随机选择要保留的节点
        num_nodes_to_keep = max(1, int(num_nodes * (1 - node_occlusion_rate)))
        nodes_to_keep = sorted(random.sample(range(num_nodes), num_nodes_to_keep))
        
        # 创建节点映射：原节点索引 -> 新节点索引
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes_to_keep)}
        
        # 过滤节点特征
        new_node_features = graph.x[nodes_to_keep]
        
        # 过滤边：只保留两个端点都在保留节点中的边
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        
        valid_edges = []
        valid_edge_attrs = []
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in node_mapping and dst in node_mapping:
                # 随机决定是否保留这条边
                if random.random() > edge_occlusion_rate:
                    valid_edges.append([node_mapping[src], node_mapping[dst]])
                    valid_edge_attrs.append(edge_attr[i].tolist())
        
        if len(valid_edges) == 0:
            # 如果没有边，至少添加一条边确保图有效
            if len(nodes_to_keep) > 1:
                valid_edges.append([0, 1])
                valid_edge_attrs.append([0.0, 0.0])
            else:
                valid_edges.append([0, 0])
                valid_edge_attrs.append([0.0, 0.0])
        
        new_edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
        new_edge_attr = torch.tensor(valid_edge_attrs, dtype=torch.float32)
        
        # 计算完整度
        node_completeness = len(nodes_to_keep) / num_nodes
        edge_completeness = len(valid_edges) / max(num_edges, 1)
        completeness = (node_completeness + edge_completeness) / 2.0
        
        occluded_graph = Data(x=new_node_features, edge_index=new_edge_index, edge_attr=new_edge_attr)
        
        return occluded_graph, completeness


class GraphCompletenessPredictor(nn.Module):
    """图完整度预测模型"""
    
    def __init__(self, node_feature_dim: int = 6, edge_feature_dim: int = 2,
                 hidden_dim: int = 64, num_layers: int = 3):
        """
        Args:
            node_feature_dim: 节点特征维度（坐标4 + 类别1 + 形状类别1 = 6）
            edge_feature_dim: 边特征维度（2）
            hidden_dim: 隐藏层维度
            num_layers: GCN层数
        """
        super(GraphCompletenessPredictor, self).__init__()
        
        self.num_layers = num_layers
        
        # 输入层
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        
        # 中间层
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ])
        
        # 输出层
        self.conv_out = GCNConv(hidden_dim, hidden_dim)
        
        # 边特征处理
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 完整度预测头
        # 输入维度：graph_embedding (hidden_dim) + node_mean (hidden_dim) + node_std (hidden_dim) = hidden_dim * 3
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出0-1之间的完整度
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: 图数据
        Returns:
            完整度预测值 (1,)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 处理边特征并聚合到节点
        if edge_attr.size(0) > 0 and edge_attr.size(0) == edge_index.size(1):
            edge_features = self.edge_mlp(edge_attr)  # (num_edges, hidden_dim)
            
            # 将边特征聚合到节点：对每个节点，聚合所有与其相连的边的特征
            node_edge_features = torch.zeros(x.size(0), edge_features.size(1), device=x.device)
            
            # 计算每个节点的度数
            node_degrees = torch.zeros(x.size(0), device=x.device)
            for i in range(edge_index.size(1)):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                node_edge_features[src] += edge_features[i]
                node_edge_features[dst] += edge_features[i]
                node_degrees[src] += 1
                node_degrees[dst] += 1
            
            # 归一化（避免除零）
            node_degrees = node_degrees.clamp(min=1)
            node_edge_features = node_edge_features / node_degrees.unsqueeze(1)
            
            # 将边特征投影到节点特征维度并拼接
            if node_edge_features.size(1) >= x.size(1):
                node_edge_features = node_edge_features[:, :x.size(1)]
            else:
                padding = torch.zeros(x.size(0), x.size(1) - node_edge_features.size(1), device=x.device)
                node_edge_features = torch.cat([node_edge_features, padding], dim=1)
            
            # 将边特征信息融合到节点特征
            x = x + node_edge_features * 0.5  # 加权融合
        
        # GCN层处理节点特征
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.conv_out(x, edge_index)
        
        # 图级别池化
        graph_embedding = global_mean_pool(x, batch=None)  # 可能是 (hidden_dim,) 或 (1, hidden_dim)
        # 展平为1D tensor
        graph_embedding = graph_embedding.flatten()
        
        # 使用节点特征统计信息增强
        node_mean = x.mean(dim=0)  # (hidden_dim,)
        node_std = x.std(dim=0)    # (hidden_dim,)
        
        # 确保所有tensor都是1D
        node_mean = node_mean.flatten()
        node_std = node_std.flatten()
        
        # 结合图嵌入和节点统计
        combined_features = torch.cat([graph_embedding, node_mean, node_std])  # (hidden_dim * 3,)
        
        # 预测完整度
        completeness = self.predictor(combined_features)
        
        return completeness


def train_model(model: GraphCompletenessPredictor, 
                generator: GraphDataGenerator,
                occluder: GraphOccluder,
                num_epochs: int = 100,
                batch_size: int = 32,
                learning_rate: float = 0.001,
                num_samples: int = 1000,
                complete_sample_ratio: float = 0.1):
    """训练模型
    
    Args:
        model: 模型
        generator: 数据生成器
        occluder: 遮挡器
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        num_samples: 每个epoch的样本数
        complete_sample_ratio: 完整图（完整度为1）样本的比例
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    print(f"使用设备: {device}")
    print(f"开始训练，共 {num_epochs} 个epoch...")
    print(f"完整图样本比例: {complete_sample_ratio*100:.1f}%")
    
    # 计算batch数量
    num_batches_per_epoch = (num_samples + batch_size - 1) // batch_size
    
    # 使用tqdm显示epoch进度
    epoch_pbar = tqdm(range(num_epochs), desc="训练进度", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 使用tqdm显示batch进度
        batch_range = range(0, num_samples, batch_size)
        batch_pbar = tqdm(batch_range, desc=f"Epoch {epoch+1}/{num_epochs}", 
                         leave=False, unit="batch")
        
        for batch_idx in batch_pbar:
            batch_graphs = []
            batch_targets = []
            
            # 生成一个batch的数据
            batch_size_actual = min(batch_size, num_samples - batch_idx)
            num_complete_in_batch = int(batch_size_actual * complete_sample_ratio)
            
            for i in range(batch_size_actual):
                # 生成完整图
                full_graph = generator.generate_graph()
                
                # 根据比例决定是否使用完整图（完整度为1）
                if i < num_complete_in_batch:
                    # 使用完整图，不应用遮挡，完整度为1.0
                    batch_graphs.append(full_graph)
                    batch_targets.append(1.0)
                else:
                    # 应用遮挡
                    occluded_graph, completeness = occluder.occlude_graph(full_graph)
                    batch_graphs.append(occluded_graph)
                    batch_targets.append(completeness)
            
            # 将图移动到设备
            batch_graphs = [g.to(device) for g in batch_graphs]
            batch_targets = torch.tensor(batch_targets, dtype=torch.float32, device=device).unsqueeze(1)
            
            # 前向传播
            predictions = []
            for graph in batch_graphs:
                pred = model(graph)
                predictions.append(pred)
            predictions = torch.cat(predictions, dim=0)
            
            # 计算损失
            loss = criterion(predictions, batch_targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新batch进度条
            batch_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # 更新epoch进度条
        epoch_pbar.set_postfix({'avg_loss': f'{avg_loss:.6f}'})
    
    print("训练完成！")
    return model


def evaluate_model(model: GraphCompletenessPredictor,
                   generator: GraphDataGenerator,
                   occluder: GraphOccluder,
                   num_samples: int = 100,
                   num_complete_samples: int = 20):
    """评估模型
    
    Args:
        model: 模型
        generator: 数据生成器
        occluder: 遮挡器
        num_samples: 遮挡样本数量
        num_complete_samples: 完整度1的样本数量
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    total_error = 0.0
    errors = []
    
    with torch.no_grad():
        # 使用tqdm显示评估进度
        eval_pbar = tqdm(range(num_samples), desc="评估中", unit="样本")
        for _ in eval_pbar:
            # 生成完整图
            full_graph = generator.generate_graph()
            # 应用遮挡
            occluded_graph, true_completeness = occluder.occlude_graph(full_graph)
            occluded_graph = occluded_graph.to(device)
            
            # 预测
            pred_completeness = model(occluded_graph).item()
            
            error = abs(pred_completeness - true_completeness)
            errors.append(error)
            total_error += error
            
            # 更新进度条
            eval_pbar.set_postfix({'当前误差': f'{error:.6f}', '平均误差': f'{total_error/len(errors):.6f}'})
    
    avg_error = total_error / num_samples
    mse = np.mean([e**2 for e in errors])
    print(f"\n评估结果 (遮挡样本数: {num_samples}):")
    print(f"平均绝对误差: {avg_error:.6f}")
    print(f"均方误差: {mse:.6f}")
    print(f"最大误差: {max(errors):.6f}")
    print(f"最小误差: {min(errors):.6f}")
    
    # 测试完整图（完整度为1）
    print(f"\n评估完整图（完整度为1，样本数: {num_complete_samples}）:")
    complete_errors = []
    complete_pbar = tqdm(range(num_complete_samples), desc="评估完整图", unit="样本")
    for _ in complete_pbar:
        full_graph = generator.generate_graph().to(device)
        # 不应用遮挡，完整度应该是1.0
        pred = model(full_graph).item()
        error = abs(pred - 1.0)
        complete_errors.append(error)
        complete_pbar.set_postfix({'预测完整度': f'{pred:.4f}', '误差': f'{error:.4f}'})
    
    complete_avg_error = np.mean(complete_errors)
    complete_mse = np.mean([e**2 for e in complete_errors])
    print(f"完整图平均绝对误差: {complete_avg_error:.6f}")
    print(f"完整图均方误差: {complete_mse:.6f}")
    print(f"完整图最大误差: {max(complete_errors):.6f}")
    print(f"完整图最小误差: {min(complete_errors):.6f}")


def main():
    print("=" * 60)
    print("图缺失预测任务")
    print("=" * 60)
    
    # 创建基础图生成器（生成5个固定的基础图）
    print("\n生成基础图...")
    base_generator = BaseGraphGenerator(
        num_classes=5,
        num_shape_classes=3,
        seed=42
    )
    
    # 创建数据生成器（基于基础图添加噪声）
    generator = GraphDataGenerator(
        base_generator=base_generator,
        coord_noise_std=0.1,  # 坐标高斯噪声标准差
        edge_noise_std=0.1,   # 边特征高斯噪声标准差
        seed=42
    )
    
    occluder = GraphOccluder(
        node_occlusion_rate_range=(0.0, 0.6),
        edge_occlusion_rate_range=(0.0, 0.6)
    )
    
    # 创建模型
    model = GraphCompletenessPredictor(
        node_feature_dim=6,
        edge_feature_dim=2,
        hidden_dim=64,
        num_layers=3
    )
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    print("\n开始训练...")
    model = train_model(
        model=model,
        generator=generator,
        occluder=occluder,
        num_epochs=50,
        batch_size=16,
        learning_rate=0.001,
        num_samples=500,
        complete_sample_ratio=0.2  # 增加完整图样本比例到20%
    )
    
    # 评估模型
    print("\n开始评估...")
    evaluate_model(
        model=model,
        generator=generator,
        occluder=occluder,
        num_samples=100
    )
    
    print("\n任务完成！")


if __name__ == "__main__":
    main()
