# Multimodal Industrial Document Agent Architecture

## 1. 项目概述

### 项目名称

GEA Multimodal Product Intelligence Agent

### 项目目标

构建一个基于视觉文档理解的工业产品智能助手。

系统面向复杂工业 PDF 文档：

* 产品手册
* 技术规格书
* 安装说明书
* CAD 图纸
* 三视图
* 参数表
* 结构示意图

通过 Vision Retrieval + Vision Language Model 实现：

* 产品信息检索
* 参数查询
* 图纸理解
* 尺寸解析
* 部件定位
* 技术文档问答

---

# 2. 核心设计理念

传统 PDF RAG：

```
PDF
 ↓
OCR
 ↓
文本解析
 ↓
Chunk
 ↓
Embedding
 ↓
Retriever
 ↓
LLM
```

存在问题：

* 表格结构丢失
* 图片信息丢失
* 页面布局丢失
* 三视图无法理解

本系统采用：

```
Document Image Retrieval
+
Vision Language Understanding
```

即：

```
PDF Page Image

        ↓

ColQwen Vision Retriever

        ↓

Relevant Pages

        ↓

Qwen-VL Understanding

        ↓

Answer Generation
```

该设计来源于 ColPali 提出的 Vision Space Retrieval 思想：

> directly embedding the images of document pages

通过直接对文档页面视觉特征进行编码，避免复杂 OCR、布局分析和 chunk pipeline。


---

# 3. 总体架构

```
                         User

                          |
                          v

                 Application Layer

                          |

                 Agent Runtime

                          |

        --------------------------------

        |                              |

        v                              v


 Vision Retrieval Tool          Document Analysis Tool


        |                              |


 ColQwen Retriever              Qwen-VL Analyzer


        |                              |


 Multi-vector Index              Image Understanding



        |

        v


 Evidence Manager


        |

        v


 Answer Generator


```

---

# 4. 系统模块设计

## 4.1 Document Processing Pipeline

职责：

负责工业 PDF 转换为视觉知识库。

输入：

```
PDF
```

输出：

```
Document Pages
```

流程：

```
PDF

 ↓

Page Renderer

 ↓

Page Images

 ↓

Visual Embedding

 ↓

Vector Storage

```

---

## Page Object

核心数据结构：

```python
DocumentPage
```

```json
{
    "document_id": "gea_manual",

    "page_number":35,

    "image_path":
    "pages/page_035.png",

    "embedding":
    [
        "multi vector"
    ],

    "metadata":
    {
        "product":"GEA-X200",
        "source":"manual"
    }
}
```

---

# 5. Vision Retrieval Layer

## 5.1 ColQwen Retriever

职责：

从大量工业文档页面中找到相关页面。

输入：

```
User Query
```

输出：

```
Top-K Pages
```

流程：

```
Query

 ↓

ColQwen Encoder

 ↓

Query Tokens

 ↓

Late Interaction

 ↓

Ranking

 ↓

Top-K pages

```

---

## Retrieval Strategy

采用：

Multi-vector Retrieval

不是：

```
page -> single vector
```

而是：

```
page

 ↓

patch/token vectors

[
v1,
v2,
v3
...

]

```

匹配：

```
Query Tokens

        ×

Page Tokens

```

采用 Late Interaction。

ColPali论文中指出：

多向量表示结合 Late Interaction 可以更好利用页面中的视觉信息。


---

# 6. Vision Understanding Layer

## Qwen-VL Analyzer

职责：

理解检索到的页面。

输入：

```
Page Image

+

User Question
```

输出：

```
Structured Answer
```

能力：

* OCR
* 表格理解
* 图纸理解
* 尺寸读取
* 部件识别
* 空间关系分析

---

例如：

输入：

```
GEA-X200安装尺寸是多少？
```

模型看到：

```
page_035.png
```

输出：

```json
{
 "answer":
 "600×400×500mm",

 "evidence":
 {
   "page":35,

   "region":
   "dimension drawing"
 }
}
```

---

# 7. Evidence Layer

工业场景必须保留证据。

设计：

```
Evidence Object
```

```json
{
 "document":

 "GEA_manual.pdf",

 "page":

 35,


 "image":

 "page_035.png",


 "region":

 [
 100,
 200,
 600,
 700
 ]
}
```

用于：

* 用户验证
* 报告生成
* 工程审查

---

# 8. Agent Layer

未来扩展：

Agent Runtime

Tools:

```
search_product_document()

analyze_drawing()

extract_parameter()

compare_products()

generate_report()

```

---

# 9. API Design

## Query API

POST:

```
/api/v1/query
```

Request:

```json
{
 "question":

 "GEA-X200流量是多少?"
}
```

Response:

```json
{
 "answer":

 "最大流量100m3/h",

 "sources":[

 {
 "page":35
 }

 ]

}
```

---

# 10. Storage Design

## Vector Storage

保存：

```
Visual Embeddings

+
Metadata
```

支持：

* FAISS
* Milvus
* LanceDB

---

## Document Storage

保存：

```
PDF

Page Images

Metadata

Evidence

```

---

# 11. 技术栈

## Backend

```
Python

FastAPI

Pydantic

```

## AI

```
ColQwen

Qwen-VL

PyTorch

Transformers

```

## Storage

```
Vector Database

Object Storage

```

## Development

```
uv

Docker

Git

```

---

# 12. Roadmap

## Milestone 1

Vision Retrieval Core

完成：

* PDF page extraction
* ColQwen index
* Query retrieval

---

## Milestone 2

Vision QA

完成：

* Qwen-VL integration
* Answer generation

---

## Milestone 3

Evidence System

完成：

* page citation
* image region reference

---

## Milestone 4

Agent System

完成：

* Tool calling
* Workflow planning
* Product assistant

---

## Milestone 5

Industrial Deployment

完成：

* API service
* Authentication
* Monitoring

