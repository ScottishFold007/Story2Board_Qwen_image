 # Story2Board：一种免训练的故事板生成方法，在保持角色身份一致性的同时，兼顾电影式构图多样性
 <img width="934" height="607" alt="image" src="https://github.com/user-attachments/assets/38145af2-7d56-4aaa-bd2b-c7f5a46716ce" />

## 原项目详见
**Project page:** https://daviddinkevich.github.io/Story2Board  
**Paper (arXiv):** https://arxiv.org/abs/2508.09983    
**Code:** [this repo](https://github.com/DavidDinkevich/Story2Board)


Story2Board 是一个**无需训练**（training-free）的框架，它通过引入一套轻量级的**一致性机制**来实现叙事连贯性，特别是在保持**主体身份**（subject identity）的同时，允许**角色姿势、大小和位置**等在不同画板之间进行**动态变化**。

Story2Board 的一致性框架由两个互补的组成部分构成：

### 1. 潜面板锚定 (Latent Panel Anchoring, LPA)

潜面板锚定（LPA）旨在**利用模型的自注意力机制来促进画板之间的视觉一致性**。

**实现机制：**

*   **提示分解与组合**：首先，Story2Board 使用现成的语言模型（如 GPT-4o）将自然语言叙事分解为**一个共享的参考面板提示**（描述所有重复出现的角色或对象）和**一系列场景特定的画板提示**（每个画板一个）。
*   **构建复合提示**：对于故事中的每个画板，会构建一个复合提示，格式为：“**A storyboard of [参考提示] (top) and [场景提示] (bottom)**”，这促使模型在堆叠的子面板中描绘一个一致的角色，但具有不同的场景背景。
*   **生成两部分潜在网格**：每个复合提示都会对模型进行条件限制，使其生成一个两部分的潜在网格：**上半部分（$R$）演变成参考子面板，下半部分（$p_i$）演变成目标场景子面板**。
*   **共享锚定**：研究人员构建了 $n$ 个这样的潜在网格批次，确保所有目标子面板都以**相同的参考提示**为条件。
*   **去噪过程中的同步**：在联合去噪过程中，**LPA 在每个变换器块之后，将每个潜在表示（$R'_i$）的上半部分替换为批次中第一个元素的版本（$R' = R'_1$）**。这确保了每个场景都是相对于一个**共享的、同步的角色描绘**（即锚点）进行演变的。
*   **利用注意力结构**：LPA 利用了基于变换器（transformer-based）的扩散模型所展现的结构化注意力行为：对应于同一对象的词元（如角色的头发、衣物）倾向于在键空间中形成紧密的“集群”。通过在每个潜在网格中放置一个角色的参考描绘，LPA 允许注意力层对齐和融合顶部和底部子面板之间的视觉特征，从而传播纹理和风格。

### 2. 互惠注意力值混合 (Reciprocal Attention Value Mixing, RAVM)

互惠注意力值混合（RAVM）是 LPA 的补充机制，它通过**软特征混合**（soft feature blending）来增强语义对齐词元之间的跨面板对应关系，以解决 LPA 在涉及复杂布局或大姿势变化时的不足。

**实现机制：**

*   **操作目标**：RAVM **仅作用于值向量**（Value Vectors）。在注意力模型中，值向量编码了**纹理、颜色和外观**等精细的视觉细节，而键（Keys）和查询（Queries）则影响空间布局和注意力权重。通过仅修改值向量，RAVM 可以在**不影响场景空间布局**的前提下，保持角色身份。
*   **识别对齐词元**：RAVM 通过识别那些在堆叠面板之间**相互强烈关注**（strongly attend to each other）的词元对，使跨面板对应关系显性化。
*   **互惠注意力分数**：研究人员将双面板潜在表示解释为一个**有向二分注意力图**，并通过计算**双向注意力（top-to-bottom 和 bottom-to-top）的最小值**来定义词元 $u$（顶部面板）和词元 $v$（底部面板）之间的互惠注意力分数 $RA(u, v)$。
*   **选择性混合**：RAVM 对**具有最高相互连通性**（highest mutual connectivity）的词元对选择性地混合值向量。它使用 **Otsu 阈值法**对互惠注意力矩阵的指数移动平均（EMA）进行处理，以提取高置信度的对应关系。
*   **软值更新**：对于每个选定的底部词元 $v$，它找到具有最高互惠分数的顶部词元 $u^*$。然后应用一个**软值更新**公式：$V'_v = \lambda V_v + (1 - \lambda) V_{u^*}$，其中 $\lambda$ 是混合权重。这使得纹理和风格在对应区域之间软性传播。

### 整体效果与优势

Story2Board 通过结合 LPA 和 RAVM，在不引入架构更改或微调的情况下，实现了强大的连贯性：

*   **强化身份与多样性平衡**：该方法增强了**角色身份**和**面板间连贯性**，同时保留了基础模型的**完整的组合灵活性和视觉丰富性**。它实现了**角色一致性、提示对齐和场景多样性**之间的最佳平衡。
*   **超越以往方法**：与许多侧重于保持角色身份但牺牲构图多样性的现有方法不同，Story2Board 旨在支持丰富的、富有表现力的场景构造，包括**变化的视点、深度和背景叙事**。
*   **训练免费**：Story2Board 是一个**完全训练免费**的流程，可以直接应用于预训练的基于变换器（transformer-based）的扩散模型，如 Flux 和 Stable Diffusion 3。

# Qwen-Image Pipelines（含 Story2Board 扩展）

本目录包含 Qwen-Image 系列推理管线与工具，包括基础生图、编辑、Inpaint、ControlNet 等。并提供 Story2Board（训练免改造，LPA + RAVM）在 Qwen-Image 上的实现与使用说明（实现文件位于仓库根目录 `story2board_qwen_image/`）。

## 环境依赖

- Python 3.10+
- PyTorch 2.0+（需支持 `scaled_dot_product_attention`）
- diffusers（与本仓库版本对应）
- transformers
- 其它依赖见项目根目录/上级目录的 requirements（如果有）

## 目录结构（节选）

- `pipeline_qwenimage.py`：基础 T2I 管线
- `pipeline_qwenimage_*`：编辑、Inpaint、ControlNet、Img2Img 等扩展
- `pipeline_output.py`：输出类型定义
- Story2Board 扩展（实现放在根目录 `story2board_qwen_image/`）：
  - `story2board_transformer.py`：带 LPA 的 Transformer 子类
  - `story2board_attention_processor.py`：RAVM 注意力处理器
  - `attention_store.py`：跨步/层互惠注意力的聚合存储
  - `pipeline_qwenimage_story2board.py`：Story2Board Qwen-Image 管线

## 基础用法（普通 Qwen-Image 生图）

```python
import torch
from diffusers import QwenImagePipeline

pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16).to("cuda")
prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, num_inference_steps=50).images[0]
image.save("qwenimage.png")
```

## Story2Board（LPA + RAVM）在 Qwen-Image 上的用法

Story2Board 的实现位于根目录 `story2board_qwen_image/`，请从该包导入：

```python
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from story2board_qwen_image import (
    Story2BoardQwenImagePipeline,
    Story2BoardQwenTransformer2DModel,
)
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from diffusers.models import AutoencoderKLQwenImage

# 按你的方式加载/组装组件（示例中略去权重加载细节）
vae = AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae", torch_dtype=torch.bfloat16)
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16)
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
transformer = Story2BoardQwenTransformer2DModel.from_pretrained("Qwen/Qwen-Image", subfolder="transformer", torch_dtype=torch.bfloat16)
scheduler = FlowMatchEulerDiscreteScheduler()

pipe = Story2BoardQwenImagePipeline(
    scheduler=scheduler,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    transformer=transformer,
).to("cuda")

# 构造"两联图"提示：top=参考（共享）、bottom=场景
prompts = [
    "A storyboard of [reference prompt] (top) and [scene-1] (bottom)",
    "A storyboard of [reference prompt] (top) and [scene-2] (bottom)",
]

result = pipe(
    prompts,
    num_inference_steps=28,
    same_noise=True,
    ravm_mixing_coef=0.5,
    first_mixing_block=8,
    last_mixing_block=30,
    first_mixing_denoising_step=4,
    last_mixing_denoising_step=24,
)
images = result.images

# 每张输出是上下两联图，裁切下半部分即为目标分镜
for i, im in enumerate(images):
    w, h = im.size
    bottom = im.crop((0, h//2, w, h))
    bottom.save(f"storyboard_panel_{i}.png")
```

### 关键参数

- `same_noise`: 批内共享初始噪声（稳定身份）
- `ravm_mixing_coef`: Value 混合系数 α（建议 0.4~0.6）
- `first_mixing_block / last_mixing_block`: RAVM 生效的层范围
- `first_mixing_denoising_step / last_mixing_denoising_step`: RAVM 生效的去噪步范围

### 实现要点（与论文/算法文档一致）

- LPA：在每个 Transformer block 后，将 batch[0] 的"上半图像 token"覆写到其余样本的"上半 token"位置，稳定参考主体。
- RAVM：在注意力处计算互惠注意力（top↔bottom 的最小值），对 bottom 的高置信区域仅混合 Value，实现外观一致性而不影响布局（Q/K）。

## 备注

- 如果你已有成熟的加载封装，可直接替换 Transformer 与 Pipeline 为 Story2Board 版本，并传入上述参数即可。
- 参考实现细节见根目录 `story2board_flux/` 与 `Story2Board_Algorithm.md`。

