# AnimeLens 开发日志

## 1. 项目概述

AnimeLens 是一个基于 FastAPI 的动漫及视觉内容识别应用，支持上传 JPG、JPEG 或 PNG 图片，并尝试识别图片中的动漫、国漫、条漫、插画、同人图和海报等内容。

本阶段开发的核心目标是将原有的单一识别流程升级为更可靠的多服务级联流程，同时完善 Vision LLM 结果、官方封面获取、前端异常回退机制，并新增一个独立的 Persona 5 风格前端页面用于视觉实验。

本次开发没有覆盖默认首页 `static/index.html`，Persona 5 页面作为独立预览入口保留在 `static/persona5.html`。

---

## 2. 初始需求：升级为三级级联识别

项目最初需要将识别策略重构为三级级联降级：

第一层使用 Trace.moe 查询动漫视频帧，适合处理动画截图，并可以返回作品、集数、视频片段和时间位置等信息。

第二层使用多模态 Vision LLM，主要解决 Trace.moe 不擅长的场景，包括中国动画、条漫、静态插画和缺少视频帧索引的图片。

第三层使用 SauceNAO，作为插画、同人图、海报以及前两层无法可靠识别时的兜底服务。

最终确定的请求顺序为：

```text
Trace.moe → Vision LLM → SauceNAO
```

级联的基本原则是：单个上游服务失败、超时、响应异常或无法可靠识别时，不应该直接终止整个请求，而是继续进入下一层服务。

---

## 3. 后端架构改造

### 3.1 `config.py`

配置文件进行了集中化整理，统一维护外部服务地址、环境变量、API Key、Vision 模型、识别阈值、文件限制和缓存配置。

主要配置包括：

```python
TRACE_MOE_URL = "https://api.trace.moe/search"
SAUCENAO_URL = "https://saucenao.com/search.php"
ANILIST_GRAPHQL_URL = "https://graphql.anilist.co"

SAUCENAO_API_KEY = os.getenv("SAUCENAO_API_KEY", "").strip()
VISION_API_KEY = os.getenv("VISION_API_KEY", "").strip()
VISION_BASE_URL = os.getenv(
    "VISION_BASE_URL", "https://api.openai.com"
).strip().rstrip("/")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini").strip()

TRACE_CONFIDENCE_THRESHOLD = 0.85
MAX_FILE_SIZE = 10 * 1024 * 1024
```

Vision LLM 使用 OpenAI-compatible 接口，最终请求地址由以下方式拼接：

```text
{VISION_BASE_URL}/v1/chat/completions
```

### 3.2 `schemas.py`

稳定的 API 数据结构进行了扩展。

`SearchResult.engine` 现在明确支持以下三个值：

```python
Literal["tracemoe", "ai_vision", "saucenao"]
```

结果中保留并规范化了以下字段：

```text
title
similarity
video_url
image_url
thumbnail_url
episode
from
to
character
description
engine
```

其中 `similarity` 被限制在 `0.0` 到 `1.0` 之间，`character` 和 `description` 用于承载 Vision LLM 返回的角色和画面描述信息。

### 3.3 `services.py`

`services.py` 负责所有上游服务访问、响应解析、字段清洗和结果标准化。

保留并完善了 Trace.moe 处理逻辑，包括标题清洗、相似度归一化、AniList 标题补充和结果标准化。

新增 Vision LLM 识别流程，图片先编码为 Base64 Data URL，再发送到 OpenAI-compatible Vision API。模型被要求返回严格 JSON，字段包括：

```json
{
  "title": "作品名称",
  "character": "角色名称",
  "description": "画面或角色描述",
  "confidence": 0.86
}
```

Vision 返回有效作品名时，会生成 `engine="ai_vision"` 的 `SearchResult`。Vision 结果不提供视频地址，因为静态识别结果通常没有可验证的视频片段。

---

## 4. Vision LLM 结果图片的处理

开发过程中发现 Vision LLM 可以正确识别动漫名称，但如果直接让模型生成图片 URL，很容易产生不存在的链接和 404 错误。

因此确定不让模型生成图片地址，而是增加官方数据源查询流程：

```python
async def _fetch_anilist_cover(
    client: httpx.AsyncClient,
    anime_title: str,
) -> str | None:
```

该函数通过 AniList GraphQL 使用标题搜索动漫，并读取：

```graphql
coverImage {
  large
}
```

查询成功后，官方封面地址同时写入：

```python
image_url=cover_url
thumbnail_url=cover_url
```

如果 AniList 请求失败、响应格式异常、GraphQL 返回错误或没有找到封面，则返回 `None`，由前端进行图片回退处理。

这一方案避免了模型幻觉链接，同时可以让 Vision LLM 结果具备与其他识别引擎一致的封面展示能力。

---

## 5. Vision LLM 置信度策略

最初 Vision 结果的相似度固定为 `1.0`，无法反映模型实际的识别把握程度。之后将置信度交由模型根据图像证据动态判断，并制定了明确的判断区间：

| 置信度区间 | 使用条件 |
| --- | --- |
| 0.92 - 0.98 | 具有极强的角色标志性特征，或存在明确台词、水印等直接证据 |
| 0.80 - 0.91 | 画风或角色非常熟悉，大概率属于该作品，但缺少决定性特征 |
| 0.65 - 0.79 | 依据大众画风或常见人设进行合理推测 |
| 低于 0.60 | 无法可靠识别，标题应填写为 `UNKNOWN` |

同时采用了兼顾安全性和合理范围的异常值处理策略：

```python
raw_confidence = result_data.get("confidence", 0.82)
try:
    confidence = float(raw_confidence)
except (TypeError, ValueError):
    confidence = 0.82

if not math.isfinite(confidence):
    confidence = 0.82

confidence = max(0.0, min(confidence, 1.0))

if confidence < 0.60 or confidence > 0.98:
    confidence = 0.82
```

这样可以避免模型返回负数、超过 1、无穷大、非数字或过度自信的异常值。

---

## 6. `main.py` 三级级联编排

`main.py` 继续负责 FastAPI 应用、静态文件服务、上传校验和请求编排。

`/api/search` 的最终流程如下：

```text
1. 校验 MIME 类型和文件扩展名
2. 读取 MAX_FILE_SIZE + 1 字节，检测是否超过 10MB
3. 请求 Trace.moe
4. 如果 Trace.moe 最高相似度达到 0.85，直接返回 Trace.moe 结果
5. 如果 Trace.moe 不够可靠，调用 Vision LLM
6. 如果 Vision LLM 成功识别，返回 Vision 结果
7. 如果 Vision LLM 失败、返回 UNKNOWN 或配置缺失，调用 SauceNAO
8. 如果 SauceNAO 失败但 Trace.moe 有低置信度结果，返回低置信度 Trace.moe 结果
9. 如果所有引擎都没有可用结果，返回友好的 HTTP 错误
```

FastAPI lifespan 创建共享的 `httpx.AsyncClient`，应用关闭时负责释放客户端。各个上游服务的超时、HTTP 错误、异常响应和配置缺失都会转换为可继续降级的错误，而不是直接导致级联中断。

---

## 7. 默认首页的前端改造

`static/index.html` 保持原有的浅色蓝粉视觉风格，但补充了与后端结果结构对应的展示能力。

主要改造包括：

- 使用 `URL.createObjectURL(file)` 保存当前上传图片
- 优先显示 Trace.moe 视频
- 没有视频时显示官方封面或 SauceNAO 图片
- 官方图片缺失或加载失败时回退到当前上传图片
- 显示识别引擎
- 显示置信度百分比
- 显示动漫标题和多语言副标题
- 显示具体集数
- 显示画面起止时间
- 显示 Vision LLM 的角色和描述
- 对后端返回的文本使用 `escapeHtml`，避免直接插入 HTML
- 重置时释放 object URL，避免长期占用浏览器资源

当前主页面的结果卡片根据不同引擎和置信度使用不同颜色，能够区分 Trace.moe、Vision AI 和 SauceNAO 的来源。

---

## 8. 独立 Persona 5 风格页面

在不覆盖默认首页的前提下，新增了：

```text
static/persona5.html
```

该页面使用 Persona 5 风格进行视觉实验，设计方向包括：

- 纯黑背景
- 红、黑、白高对比配色
- 解构主义排版
- 剪报式中文标题
- 粗犷黑色边框
- Neo-brutalism 硬阴影
- 斜向红色背景色块
- Calling Card 风格上传区域
- 右下角 Joker 吉祥物图片

页面保留了原有上传功能所依赖的 DOM ID：

```text
drop-zone
file-input
loading
results-section
results-grid
error-message
reset-button
```

同时保留了点击上传、拖放上传、剪贴板粘贴、加载状态、错误状态和 `/api/search` 请求。

---

## 9. Persona 上传区域的迭代过程

Persona 页面上传区域经历了多次视觉迭代。

最初采用较大的 Calling Card 布局，上传区域宽度为 `max-w-5xl`，内部使用较大的上下内边距。

随后尝试了大面积的放射状碎片菜单，将多个按钮和 `clip-path` 标签铺满上传区域。该方案虽然符合 Persona 5 游戏菜单的视觉方向，但实际效果过于拥挤、信息层级不清晰，因此根据反馈撤销了这一轮修改，恢复到更简洁的上传卡片。

之后采用了更克制的方案：

- 将上传区域缩小到接近默认首页的 `max-w-3xl`
- 收紧横向和纵向内边距
- 只保留少量左右两侧标签
- 让标签像针刺一样从卡片边缘向中心插入
- 使用 `clip-path: polygon(...)` 制作尖锐形状
- 使用红色底层和黑色错位顶层
- 使用不同旋转角度和短促 hover 位移
- 移动端隐藏侧边标签，避免遮挡主要上传信息
- 将装饰标签设置为 `pointer-events-none`，避免干扰上传区域点击

这一轮最终保留了 Persona 风格，同时避免再次形成全屏、混乱的碎片菜单。

---

## 10. Persona 识别结果区域升级

Persona 页面最初的结果区域只有简单的标题和描述，缺少默认首页中的完整识别信息。

本阶段补充了以下功能，并将 UI 重新适配为 Persona 5 风格：

### 媒体展示

当结果包含 `video_url` 时，使用可控制的视频播放器显示视频片段。视频加载失败时，自动显示当前上传图片作为回退。

当结果包含 `image_url` 或 `thumbnail_url` 时，显示官方封面或其他来源图片。如果图片加载失败，优先回退到当前上传图片。

### 置信度

结果卡片显示类似以下内容：

```text
86.0% CONFIDENCE
```

置信度同时支持后端返回 `0~1` 或 `0~100` 的形式，并会在前端归一化到百分比展示。

### 识别引擎

结果卡片使用 Persona 风格标签显示引擎来源：

```text
TRACE.MOE // FRAME
VISION AI // ANALYSIS
SAUCENAO // FALLBACK
```

### 集数和画面时间

结果卡片增加两个信息块：

```text
Episode      第 3 集
Frame time   12:24 — 12:28
```

同时兼容后端可能使用的 `from_` 和 `to_` 字段别名。

### Vision 细节

对于 Vision LLM 结果，增加红黑硬阴影样式的识别细节区域，显示：

- 角色名称
- 模型返回的画面描述

所有动态结果字段都经过 HTML 转义后再写入卡片，避免用户控制的结果内容直接注入页面。

---

## 11. 遇到的问题与解决方式

### 11.1 `CLAUDE.md` 初次编辑匹配失败

第一次使用精确文本替换更新 `CLAUDE.md` 时，旧文本没有完全匹配，导致替换失败。

之后改用完整覆盖方式写入文件，成功完成项目文档更新。

### 11.2 Python 环境依赖冲突

运行 `pip check` 时发现当前环境存在外部依赖冲突：

```text
pipx 1.0.0 has requirement argcomplete>=1.9.4, but you have argcomplete 1.8.1.
```

该问题来自当前 Python 环境，而不是 AnimeLens 项目代码，因此没有修改项目依赖文件进行强行规避。

### 11.3 本地环境缺少 Pydantic

执行 Python 导入冒烟测试时发现环境缺少 `pydantic`：

```text
ModuleNotFoundError: No module named 'pydantic'
```

项目源代码的语法检查仍然通过，但完整运行时测试需要先按照 `requirements.txt` 安装项目依赖。

### 11.4 Endpoint 参数字段使用错误

曾经尝试使用错误的 endpoint 描述字段，之后确认 FastAPI 应使用 `summary`，并将接口摘要更新为三级级联引擎描述。

### 11.5 `_normalise_name` 缩进错误

`services.py` 曾出现 `_normalise_name` 函数缩进位置错误，使函数体落在 Vision 相关代码附近。之后将其恢复为独立的顶层函数，并重新完成语法检查。

### 11.6 JavaScript 脚本提取方式错误

第一次检查 Persona 页面 JavaScript 时，临时脚本错误地包含了 `</script>` 标签，导致 Node 报错：

```text
SyntaxError: Unexpected token '<'
```

之后改为只提取 `<script>...</script>` 标签内部的内容，Node 语法检查通过。

### 11.7 放射状碎片菜单过于混乱

大范围绝对定位的碎片菜单在视觉上过于拥挤，影响上传区域的可读性和操作聚焦。

解决方式是撤销整轮菜单改造，改为缩小上传卡片、减少标签数量，并将碎片限制在上传区域左右两侧。

### 11.8 结果区初始信息不完整

Persona 页面初始结果卡片只显示标题和描述，缺少视频、图片、置信度、集数、时间和引擎等字段。

之后参考默认首页的结果处理逻辑，重新实现了 Persona 风格的完整结果卡片。

---

## 12. 验证与检查记录

本阶段已完成以下检查：

- `static/persona5.html` 保留所有必需 DOM ID
- `/api/search` 请求逻辑仍然存在
- 点击上传事件仍然存在
- 拖放上传事件仍然存在
- 剪贴板粘贴事件仍然存在
- 原图 object URL 创建和释放逻辑已加入
- 结果区域只有一个有效的 `renderResults` 实现
- 结果字段已进行 HTML 转义
- `video_url`、图片 URL、置信度、集数、时间、角色和描述均已接入
- Persona 页面 JavaScript 已通过 Node 语法检查
- 默认首页未被 Persona 页面覆盖
- `CLAUDE.md` 已记录当前项目架构、开发命令、运行方式和测试方式

需要注意的是，之前的完整运行验证受到本地环境缺少 Pydantic 和依赖冲突的影响。若要进行端到端测试，应先安装项目依赖并配置相应的上游 API Key。

---

## 13. 当前项目状态

当前项目已经完成本阶段的主要开发目标：

```text
后端三级级联识别       已完成
Vision LLM 接入         已完成
AniList 官方封面查询    已完成
动态置信度处理          已完成
默认首页结果展示        已完成
Persona 独立页面        已完成
Persona 上传区域改造    已完成
Persona 结果区域升级    已完成
CLAUDE.md 项目文档      已完成
```

当前默认首页仍然是：

```text
static/index.html
```

独立 Persona 5 风格页面为：

```text
static/persona5.html
```

后端接口仍为：

```text
POST /api/search
```

---

## 14. 后续可选工作

虽然本阶段开发已经告一段落，后续仍可以考虑补充自动化测试，尤其是针对 Trace.moe、Vision LLM 和 SauceNAO 的模拟响应测试；增加 AniList 缓存命中和失败场景测试；为前端结果卡片补充浏览器端交互测试；在真实环境中验证不同 Vision API 提供商的 JSON 输出兼容性；以及为 Persona 页面增加独立访问入口或导航链接。

在正式部署前，还应根据实际运行环境安装完整依赖、配置 API Key、检查各个上游服务的速率限制，并确认 CORS 策略是否符合部署要求。

---

## 日志结束

本阶段的开发重点从后端识别链路重构开始，逐步扩展到结果数据补全、官方图片来源校正、置信度治理、前端回退处理和独立视觉页面设计，最终形成了一个具备三级识别能力、可展示完整识别证据，并且保留两套不同 UI 风格的 AnimeLens 原型。
