#!/usr/bin/env python3
"""
THESIS RESOURCE DOWNLOADER
Downloads all 104 resources organized by research phase.

Run: python download_all_resources.py

Requirements:
    pip install requests beautifulsoup4 youtube-transcript-api arxiv
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Optional imports - install as needed
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    print("Warning: arxiv not installed. Run: pip install arxiv")

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False
    print("Warning: youtube-transcript-api not installed")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("Warning: beautifulsoup4 not installed")


@dataclass
class Resource:
    """Represents a thesis resource with metadata."""
    id: int
    title: str
    authors: str
    year: int
    phase: str
    subphase: str
    resource_type: str  # paper, blog, video, tool
    url: str
    arxiv_id: Optional[str] = None
    youtube_id: Optional[str] = None
    importance: int = 3  # 1-5 stars
    is_critical: bool = False
    bibtex_key: Optional[str] = None
    local_path: Optional[str] = None


# ============================================================
# RESOURCE DEFINITIONS - All 104 Resources
# ============================================================

RESOURCES: List[Resource] = [
    # ========== COMPETITORS (4) ==========
    Resource(
        id=101, title="Language Models Can Explain Neurons in Language Models",
        authors="Bills et al.", year=2023,
        phase="phase0_competitors", subphase="openai",
        resource_type="paper",
        url="https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html",
        importance=5, is_critical=True,
        bibtex_key="bills2023language"
    ),
    Resource(
        id=102, title="Explaining Black Box Text Modules in Natural Language",
        authors="Singh et al.", year=2023,
        phase="phase0_competitors", subphase="sasc",
        resource_type="paper",
        url="https://arxiv.org/abs/2305.09863",
        arxiv_id="2305.09863",
        importance=5, is_critical=True,
        bibtex_key="singh2023explaining"
    ),
    Resource(
        id=103, title="Scaling Monosemanticity",
        authors="Templeton et al.", year=2024,
        phase="phase0_competitors", subphase="anthropic",
        resource_type="blog",
        url="https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html",
        importance=5, is_critical=True,
        bibtex_key="templeton2024scaling"
    ),
    Resource(
        id=104, title="MIB: A Mechanistic Interpretability Benchmark",
        authors="Mueller et al.", year=2025,
        phase="phase0_competitors", subphase="mib",
        resource_type="paper",
        url="https://arxiv.org/abs/2504.13151",
        arxiv_id="2504.13151",
        importance=5, is_critical=True,
        bibtex_key="mueller2025mib"
    ),

    # ========== PHASE 1: FOUNDATIONS (15) ==========
    Resource(
        id=1, title="Neural Networks: Zero to Hero",
        authors="Andrej Karpathy", year=2023,
        phase="phase1_foundations", subphase="pytorch",
        resource_type="video",
        url="https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ",
        importance=5
    ),
    Resource(
        id=2, title="PyTorch Official Tutorials",
        authors="PyTorch Team", year=2024,
        phase="phase1_foundations", subphase="pytorch",
        resource_type="docs",
        url="https://pytorch.org/tutorials/",
        importance=4
    ),
    Resource(
        id=3, title="Practical Deep Learning",
        authors="Fast.ai", year=2024,
        phase="phase1_foundations", subphase="pytorch",
        resource_type="video",
        url="https://course.fast.ai/",
        importance=4
    ),
    Resource(
        id=4, title="Python for Data Analysis",
        authors="Wes McKinney", year=2022,
        phase="phase1_foundations", subphase="pytorch",
        resource_type="book",
        url="https://wesmckinney.com/book/",
        importance=3
    ),
    Resource(
        id=5, title="Attention Is All You Need",
        authors="Vaswani et al.", year=2017,
        phase="phase1_foundations", subphase="transformers",
        resource_type="paper",
        url="https://arxiv.org/abs/1706.03762",
        arxiv_id="1706.03762",
        importance=5, is_critical=True,
        bibtex_key="vaswani2017attention"
    ),
    Resource(
        id=6, title="The Illustrated Transformer",
        authors="Jay Alammar", year=2018,
        phase="phase1_foundations", subphase="transformers",
        resource_type="blog",
        url="https://jalammar.github.io/illustrated-transformer/",
        importance=5
    ),
    Resource(
        id=7, title="Attention in Transformers",
        authors="3Blue1Brown", year=2024,
        phase="phase1_foundations", subphase="transformers",
        resource_type="video",
        url="https://www.youtube.com/watch?v=eMlx5fFNoYc",
        youtube_id="eMlx5fFNoYc",
        importance=5
    ),
    Resource(
        id=8, title="Language Models are Unsupervised Multitask Learners",
        authors="Radford et al.", year=2019,
        phase="phase1_foundations", subphase="transformers",
        resource_type="paper",
        url="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
        importance=4,
        bibtex_key="radford2019language"
    ),
    Resource(
        id=9, title="The Illustrated GPT-2",
        authors="Jay Alammar", year=2019,
        phase="phase1_foundations", subphase="transformers",
        resource_type="blog",
        url="https://jalammar.github.io/illustrated-gpt2/",
        importance=4
    ),
    Resource(
        id=10, title="The Transformer Family v2",
        authors="Lilian Weng", year=2023,
        phase="phase1_foundations", subphase="transformers",
        resource_type="blog",
        url="https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/",
        importance=4
    ),
    Resource(
        id=11, title="Neural Machine Translation by Jointly Learning to Align and Translate",
        authors="Bahdanau et al.", year=2014,
        phase="phase1_foundations", subphase="attention",
        resource_type="paper",
        url="https://arxiv.org/abs/1409.0473",
        arxiv_id="1409.0473",
        importance=3,
        bibtex_key="bahdanau2014neural"
    ),
    Resource(
        id=12, title="Visualizing Neural Machine Translation",
        authors="Jay Alammar", year=2018,
        phase="phase1_foundations", subphase="attention",
        resource_type="blog",
        url="https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/",
        importance=4
    ),
    Resource(
        id=13, title="A Survey of Transformers",
        authors="Lin et al.", year=2022,
        phase="phase1_foundations", subphase="attention",
        resource_type="paper",
        url="https://arxiv.org/abs/2106.04554",
        arxiv_id="2106.04554",
        importance=3,
        bibtex_key="lin2022survey"
    ),
    Resource(
        id=14, title="Einops Tutorial",
        authors="Alex Rogozhnikov", year=2022,
        phase="phase1_foundations", subphase="attention",
        resource_type="docs",
        url="https://einops.rocks/1-einops-basics/",
        importance=4
    ),
    Resource(
        id=15, title="What Do Vision Transformers Learn?",
        authors="Raghu et al.", year=2021,
        phase="phase1_foundations", subphase="attention",
        resource_type="paper",
        url="https://arxiv.org/abs/2108.08810",
        arxiv_id="2108.08810",
        importance=3,
        bibtex_key="raghu2021vision"
    ),

    # ========== PHASE 2: MECHANISTIC INTERPRETABILITY (20) ==========
    Resource(
        id=16, title="A Mathematical Framework for Transformer Circuits",
        authors="Elhage et al.", year=2021,
        phase="phase2_mech_interp", subphase="foundational",
        resource_type="blog",
        url="https://transformer-circuits.pub/2021/framework/index.html",
        importance=5, is_critical=True,
        bibtex_key="elhage2021mathematical"
    ),
    Resource(
        id=17, title="In-context Learning and Induction Heads",
        authors="Olsson et al.", year=2022,
        phase="phase2_mech_interp", subphase="foundational",
        resource_type="blog",
        url="https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html",
        importance=5, is_critical=True,
        bibtex_key="olsson2022context"
    ),
    Resource(
        id=18, title="Interpretability in the Wild: IOI Circuit",
        authors="Wang et al.", year=2023,
        phase="phase2_mech_interp", subphase="foundational",
        resource_type="paper",
        url="https://arxiv.org/abs/2211.00593",
        arxiv_id="2211.00593",
        importance=5, is_critical=True,
        bibtex_key="wang2023interpretability"
    ),
    Resource(
        id=19, title="Toy Models of Superposition",
        authors="Elhage et al.", year=2022,
        phase="phase2_mech_interp", subphase="foundational",
        resource_type="blog",
        url="https://transformer-circuits.pub/2022/toy_model/index.html",
        importance=4,
        bibtex_key="elhage2022toy"
    ),
    Resource(
        id=20, title="Softmax Linear Units",
        authors="Elhage et al.", year=2022,
        phase="phase2_mech_interp", subphase="foundational",
        resource_type="blog",
        url="https://transformer-circuits.pub/2022/solu/index.html",
        importance=3
    ),
    Resource(
        id=21, title="Scaling Monosemanticity",
        authors="Templeton et al.", year=2024,
        phase="phase2_mech_interp", subphase="foundational",
        resource_type="blog",
        url="https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html",
        importance=5, is_critical=True,
        bibtex_key="templeton2024scaling"
    ),
    Resource(
        id=22, title="Comprehensive Mechanistic Interpretability Playlist",
        authors="Neel Nanda", year=2023,
        phase="phase2_mech_interp", subphase="foundational",
        resource_type="video",
        url="https://www.youtube.com/playlist?list=PL7m7hLIqA0hoIUPhC26ASCVs_VrqcDpAz",
        importance=5
    ),
    Resource(
        id=23, title="Locating and Editing Factual Associations in GPT (ROME)",
        authors="Meng et al.", year=2022,
        phase="phase2_mech_interp", subphase="circuits",
        resource_type="paper",
        url="https://arxiv.org/abs/2202.05262",
        arxiv_id="2202.05262",
        importance=4,
        bibtex_key="meng2022locating"
    ),
    Resource(
        id=24, title="Causal Scrubbing",
        authors="Redwood Research", year=2022,
        phase="phase2_mech_interp", subphase="circuits",
        resource_type="blog",
        url="https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing",
        importance=4
    ),
    Resource(
        id=25, title="Attribution Patching",
        authors="Neel Nanda", year=2023,
        phase="phase2_mech_interp", subphase="circuits",
        resource_type="blog",
        url="https://www.neelnanda.io/mechanistic-interpretability/attribution-patching",
        importance=4
    ),
    Resource(
        id=26, title="Progress Measures for Grokking",
        authors="Nanda et al.", year=2023,
        phase="phase2_mech_interp", subphase="circuits",
        resource_type="paper",
        url="https://arxiv.org/abs/2301.05217",
        arxiv_id="2301.05217",
        importance=3,
        bibtex_key="nanda2023progress"
    ),
    Resource(
        id=27, title="Towards Automated Circuit Discovery (ACDC)",
        authors="Conmy et al.", year=2023,
        phase="phase2_mech_interp", subphase="circuits",
        resource_type="paper",
        url="https://arxiv.org/abs/2304.14997",
        arxiv_id="2304.14997",
        importance=4,
        bibtex_key="conmy2023automated"
    ),
    Resource(
        id=28, title="Copy Suppression",
        authors="Anthropic", year=2024,
        phase="phase2_mech_interp", subphase="circuits",
        resource_type="blog",
        url="https://transformer-circuits.pub/2024/copy-suppression/index.html",
        importance=3
    ),
    Resource(
        id=29, title="Activation Patching Tutorial",
        authors="Neel Nanda", year=2023,
        phase="phase2_mech_interp", subphase="causal",
        resource_type="blog",
        url="https://www.neelnanda.io/mechanistic-interpretability/activation-patching",
        importance=5
    ),
    Resource(
        id=30, title="Path Patching Tutorial",
        authors="Neel Nanda", year=2023,
        phase="phase2_mech_interp", subphase="causal",
        resource_type="blog",
        url="https://www.neelnanda.io/mechanistic-interpretability/path-patching",
        importance=4
    ),
    Resource(
        id=31, title="The Logit Lens",
        authors="nostalgebraist", year=2020,
        phase="phase2_mech_interp", subphase="causal",
        resource_type="blog",
        url="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens",
        importance=4
    ),
    Resource(
        id=32, title="Tuned Lens",
        authors="Belrose et al.", year=2023,
        phase="phase2_mech_interp", subphase="causal",
        resource_type="paper",
        url="https://arxiv.org/abs/2303.08112",
        arxiv_id="2303.08112",
        importance=4,
        bibtex_key="belrose2023tuned"
    ),
    Resource(
        id=33, title="ARENA 3.0 - Chapter 1",
        authors="Callum McDougall", year=2024,
        phase="phase2_mech_interp", subphase="causal",
        resource_type="docs",
        url="https://arena3.org/",
        importance=5, is_critical=True
    ),
    Resource(
        id=34, title="200 Concrete Open Problems in Mechanistic Interpretability",
        authors="Neel Nanda", year=2022,
        phase="phase2_mech_interp", subphase="causal",
        resource_type="blog",
        url="https://www.alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability",
        importance=4
    ),
    Resource(
        id=35, title="Transformer Circuits Thread",
        authors="Anthropic", year=2024,
        phase="phase2_mech_interp", subphase="causal",
        resource_type="blog",
        url="https://transformer-circuits.pub/",
        importance=5
    ),

    # ========== PHASE 3: BASELINE METHODS (13) ==========
    Resource(
        id=36, title="SHAP: A Unified Approach to Interpreting Model Predictions",
        authors="Lundberg & Lee", year=2017,
        phase="phase3_baselines", subphase="attribution",
        resource_type="paper",
        url="https://arxiv.org/abs/1705.07874",
        arxiv_id="1705.07874",
        importance=5, is_critical=True,
        bibtex_key="lundberg2017unified"
    ),
    Resource(
        id=37, title="LIME: Why Should I Trust You?",
        authors="Ribeiro et al.", year=2016,
        phase="phase3_baselines", subphase="attribution",
        resource_type="paper",
        url="https://arxiv.org/abs/1602.04938",
        arxiv_id="1602.04938",
        importance=4,
        bibtex_key="ribeiro2016should"
    ),
    Resource(
        id=38, title="Axiomatic Attribution for Deep Networks (Integrated Gradients)",
        authors="Sundararajan et al.", year=2017,
        phase="phase3_baselines", subphase="attribution",
        resource_type="paper",
        url="https://arxiv.org/abs/1703.01365",
        arxiv_id="1703.01365",
        importance=4,
        bibtex_key="sundararajan2017axiomatic"
    ),
    Resource(
        id=39, title="Attention is not Explanation",
        authors="Jain & Wallace", year=2019,
        phase="phase3_baselines", subphase="attention_debate",
        resource_type="paper",
        url="https://arxiv.org/abs/1902.10186",
        arxiv_id="1902.10186",
        importance=4,
        bibtex_key="jain2019attention"
    ),
    Resource(
        id=40, title="Attention is not not Explanation",
        authors="Wiegreffe & Pinter", year=2019,
        phase="phase3_baselines", subphase="attention_debate",
        resource_type="paper",
        url="https://arxiv.org/abs/1908.04626",
        arxiv_id="1908.04626",
        importance=4,
        bibtex_key="wiegreffe2019attention"
    ),
    Resource(
        id=41, title="A Survey on Explainable AI",
        authors="Saeed & Omlin", year=2023,
        phase="phase3_baselines", subphase="attention_debate",
        resource_type="paper",
        url="https://arxiv.org/abs/2309.01029",
        arxiv_id="2309.01029",
        importance=3,
        bibtex_key="saeed2023survey"
    ),
    Resource(
        id=42, title="What Does BERT Look At?",
        authors="Clark et al.", year=2019,
        phase="phase3_baselines", subphase="probing",
        resource_type="paper",
        url="https://arxiv.org/abs/1906.04341",
        arxiv_id="1906.04341",
        importance=3,
        bibtex_key="clark2019bert"
    ),
    Resource(
        id=43, title="BertViz",
        authors="Jesse Vig", year=2019,
        phase="phase3_baselines", subphase="probing",
        resource_type="tool",
        url="https://github.com/jessevig/bertviz",
        importance=4
    ),
    Resource(
        id=44, title="Visualizing Attention in Transformers",
        authors="Jesse Vig", year=2019,
        phase="phase3_baselines", subphase="probing",
        resource_type="paper",
        url="https://arxiv.org/abs/1904.02679",
        arxiv_id="1904.02679",
        importance=3,
        bibtex_key="vig2019visualizing"
    ),
    Resource(
        id=45, title="A Primer in BERTology",
        authors="Rogers et al.", year=2020,
        phase="phase3_baselines", subphase="probing",
        resource_type="paper",
        url="https://arxiv.org/abs/2002.12327",
        arxiv_id="2002.12327",
        importance=4,
        bibtex_key="rogers2020primer"
    ),
    Resource(
        id=46, title="Probing Classifiers",
        authors="Belinkov", year=2022,
        phase="phase3_baselines", subphase="probing",
        resource_type="paper",
        url="https://arxiv.org/abs/2102.12452",
        arxiv_id="2102.12452",
        importance=3,
        bibtex_key="belinkov2022probing"
    ),
    Resource(
        id=47, title="Do Vision Transformers See Like CNNs?",
        authors="Raghu et al.", year=2021,
        phase="phase3_baselines", subphase="probing",
        resource_type="paper",
        url="https://arxiv.org/abs/2108.08810",
        arxiv_id="2108.08810",
        importance=3
    ),
    Resource(
        id=48, title="Emergent Abilities of Large Language Models",
        authors="Wei et al.", year=2022,
        phase="phase3_baselines", subphase="probing",
        resource_type="paper",
        url="https://arxiv.org/abs/2206.07682",
        arxiv_id="2206.07682",
        importance=3,
        bibtex_key="wei2022emergent"
    ),

    # ========== PHASE 4: EXPLANATION GENERATION (17) ==========
    Resource(
        id=49, title="Language Models Can Explain Neurons in Language Models",
        authors="Bills et al.", year=2023,
        phase="phase4_explanation_gen", subphase="automated",
        resource_type="paper",
        url="https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html",
        importance=5, is_critical=True,
        bibtex_key="bills2023language"
    ),
    Resource(
        id=50, title="Automatically Interpreting Millions of Features in LLMs",
        authors="Anthropic", year=2024,
        phase="phase4_explanation_gen", subphase="automated",
        resource_type="blog",
        url="https://transformer-circuits.pub/2024/autointerp/index.html",
        importance=5
    ),
    Resource(
        id=51, title="OpenAI Automated Interpretability GitHub",
        authors="OpenAI", year=2023,
        phase="phase4_explanation_gen", subphase="automated",
        resource_type="tool",
        url="https://github.com/openai/automated-interpretability",
        importance=5
    ),
    Resource(
        id=52, title="Representation Engineering",
        authors="Zou et al.", year=2023,
        phase="phase4_explanation_gen", subphase="automated",
        resource_type="paper",
        url="https://arxiv.org/abs/2310.01405",
        arxiv_id="2310.01405",
        importance=4,
        bibtex_key="zou2023representation"
    ),
    Resource(
        id=53, title="Sparse Autoencoders Find Interpretable Features",
        authors="Cunningham et al.", year=2023,
        phase="phase4_explanation_gen", subphase="automated",
        resource_type="paper",
        url="https://arxiv.org/abs/2309.08600",
        arxiv_id="2309.08600",
        importance=4,
        bibtex_key="cunningham2023sparse"
    ),
    Resource(
        id=54, title="Chain-of-Thought Prompting",
        authors="Wei et al.", year=2022,
        phase="phase4_explanation_gen", subphase="llm_based",
        resource_type="paper",
        url="https://arxiv.org/abs/2201.11903",
        arxiv_id="2201.11903",
        importance=4,
        bibtex_key="wei2022chain"
    ),
    Resource(
        id=55, title="Large Language Models are Zero-Shot Reasoners",
        authors="Kojima et al.", year=2022,
        phase="phase4_explanation_gen", subphase="llm_based",
        resource_type="paper",
        url="https://arxiv.org/abs/2205.11916",
        arxiv_id="2205.11916",
        importance=3,
        bibtex_key="kojima2022large"
    ),
    Resource(
        id=56, title="Prompting Guide",
        authors="DAIR.AI", year=2024,
        phase="phase4_explanation_gen", subphase="llm_based",
        resource_type="docs",
        url="https://www.promptingguide.ai/",
        importance=3
    ),
    Resource(
        id=57, title="Self-Consistency Improves Chain of Thought Reasoning",
        authors="Wang et al.", year=2022,
        phase="phase4_explanation_gen", subphase="llm_based",
        resource_type="paper",
        url="https://arxiv.org/abs/2203.11171",
        arxiv_id="2203.11171",
        importance=3,
        bibtex_key="wang2022self"
    ),
    Resource(
        id=58, title="Constitutional AI",
        authors="Bai et al.", year=2022,
        phase="phase4_explanation_gen", subphase="llm_based",
        resource_type="paper",
        url="https://arxiv.org/abs/2212.08073",
        arxiv_id="2212.08073",
        importance=3,
        bibtex_key="bai2022constitutional"
    ),
    Resource(
        id=59, title="Self-Explaining Neural Networks",
        authors="Alvarez-Melis & Jaakkola", year=2018,
        phase="phase4_explanation_gen", subphase="self_explaining",
        resource_type="paper",
        url="https://arxiv.org/abs/1806.07538",
        arxiv_id="1806.07538",
        importance=4,
        bibtex_key="alvarez2018towards"
    ),
    Resource(
        id=60, title="Rationalizing Neural Predictions",
        authors="Lei et al.", year=2016,
        phase="phase4_explanation_gen", subphase="self_explaining",
        resource_type="paper",
        url="https://arxiv.org/abs/1606.04155",
        arxiv_id="1606.04155",
        importance=4,
        bibtex_key="lei2016rationalizing"
    ),
    Resource(
        id=61, title="Explain, then Predict",
        authors="Camburu et al.", year=2020,
        phase="phase4_explanation_gen", subphase="self_explaining",
        resource_type="paper",
        url="https://arxiv.org/abs/2012.01441",
        arxiv_id="2012.01441",
        importance=3,
        bibtex_key="camburu2020explain"
    ),
    Resource(
        id=62, title="Faithful Explanations Using LLM-generated Counterfactuals",
        authors="Chen et al.", year=2023,
        phase="phase4_explanation_gen", subphase="self_explaining",
        resource_type="paper",
        url="https://arxiv.org/abs/2310.00603",
        arxiv_id="2310.00603",
        importance=4,
        bibtex_key="chen2023faithful"
    ),
    Resource(
        id=63, title="Measuring Faithfulness in Chain-of-Thought Reasoning",
        authors="Lanham et al.", year=2023,
        phase="phase4_explanation_gen", subphase="self_explaining",
        resource_type="paper",
        url="https://arxiv.org/abs/2307.13702",
        arxiv_id="2307.13702",
        importance=4,
        bibtex_key="lanham2023measuring"
    ),
    Resource(
        id=64, title="LLM-Generated Explanations Survey",
        authors="Zhao et al.", year=2023,
        phase="phase4_explanation_gen", subphase="self_explaining",
        resource_type="paper",
        url="https://arxiv.org/abs/2311.02922",
        arxiv_id="2311.02922",
        importance=3,
        bibtex_key="zhao2023survey"
    ),
    Resource(
        id=65, title="Explaining Explanations",
        authors="Mittelstadt et al.", year=2019,
        phase="phase4_explanation_gen", subphase="self_explaining",
        resource_type="paper",
        url="https://arxiv.org/abs/1806.00069",
        arxiv_id="1806.00069",
        importance=3,
        bibtex_key="mittelstadt2019explaining"
    ),

    # ========== PHASE 5: EVALUATION (13) ==========
    Resource(
        id=66, title="ERASER: A Benchmark of Rationale Extraction",
        authors="DeYoung et al.", year=2020,
        phase="phase5_evaluation", subphase="faithfulness",
        resource_type="paper",
        url="https://arxiv.org/abs/1911.03429",
        arxiv_id="1911.03429",
        importance=5, is_critical=True,
        bibtex_key="deyoung2020eraser"
    ),
    Resource(
        id=67, title="Evaluating Explanations: How Much Do They Help?",
        authors="Hase & Bansal", year=2020,
        phase="phase5_evaluation", subphase="faithfulness",
        resource_type="paper",
        url="https://arxiv.org/abs/2012.01441",
        arxiv_id="2012.01441",
        importance=4
    ),
    Resource(
        id=68, title="Measuring Association Between Labels and Free-Text Rationales",
        authors="Wiegreffe et al.", year=2021,
        phase="phase5_evaluation", subphase="faithfulness",
        resource_type="paper",
        url="https://arxiv.org/abs/2110.08454",
        arxiv_id="2110.08454",
        importance=4,
        bibtex_key="wiegreffe2021measuring"
    ),
    Resource(
        id=69, title="Towards Faithfully Interpretable NLP Systems",
        authors="Jacovi & Goldberg", year=2020,
        phase="phase5_evaluation", subphase="faithfulness",
        resource_type="paper",
        url="https://arxiv.org/abs/2004.03685",
        arxiv_id="2004.03685",
        importance=4,
        bibtex_key="jacovi2020towards"
    ),
    Resource(
        id=70, title="How Can I Explain This to You?",
        authors="Ehsan et al.", year=2020,
        phase="phase5_evaluation", subphase="faithfulness",
        resource_type="paper",
        url="https://arxiv.org/abs/2002.01711",
        arxiv_id="2002.01711",
        importance=3
    ),
    Resource(
        id=71, title="Evaluating Feature Attribution Methods",
        authors="Kindermans et al.", year=2019,
        phase="phase5_evaluation", subphase="faithfulness",
        resource_type="paper",
        url="https://arxiv.org/abs/1711.00867",
        arxiv_id="1711.00867",
        importance=3,
        bibtex_key="kindermans2019reliability"
    ),
    Resource(
        id=72, title="Human Evaluation of Explanations Best Practices",
        authors="van der Lee et al.", year=2021,
        phase="phase5_evaluation", subphase="human",
        resource_type="paper",
        url="https://arxiv.org/abs/2107.13626",
        arxiv_id="2107.13626",
        importance=3
    ),
    Resource(
        id=73, title="Proxy Tasks and Subjective Measures",
        authors="Doshi-Velez & Kim", year=2017,
        phase="phase5_evaluation", subphase="human",
        resource_type="paper",
        url="https://arxiv.org/abs/2005.01831",
        arxiv_id="2005.01831",
        importance=3
    ),
    Resource(
        id=74, title="Forward Simulatability Evaluation",
        authors="Various", year=2020,
        phase="phase5_evaluation", subphase="human",
        resource_type="docs",
        url="https://github.com/jayded/eraserbenchmark",
        importance=4
    ),
    Resource(
        id=75, title="e-SNLI: Natural Language Inference with Explanations",
        authors="Camburu et al.", year=2018,
        phase="phase5_evaluation", subphase="datasets",
        resource_type="paper",
        url="https://arxiv.org/abs/1812.01193",
        arxiv_id="1812.01193",
        importance=5, is_critical=True,
        bibtex_key="camburu2018snli"
    ),
    Resource(
        id=76, title="IOI Dataset",
        authors="Wang et al.", year=2023,
        phase="phase5_evaluation", subphase="datasets",
        resource_type="tool",
        url="https://github.com/redwoodresearch/Easy-Transformer",
        importance=5
    ),
    Resource(
        id=77, title="BoolQ Dataset",
        authors="Clark et al.", year=2019,
        phase="phase5_evaluation", subphase="datasets",
        resource_type="tool",
        url="https://huggingface.co/datasets/boolq",
        importance=3
    ),
    Resource(
        id=78, title="CoS-E: Commonsense Explanations",
        authors="Rajani et al.", year=2019,
        phase="phase5_evaluation", subphase="datasets",
        resource_type="paper",
        url="https://arxiv.org/abs/1906.02361",
        arxiv_id="1906.02361",
        importance=3,
        bibtex_key="rajani2019explain"
    ),

    # ========== PHASE 6: TOOLS (12) ==========
    Resource(
        id=79, title="TransformerLens",
        authors="Neel Nanda", year=2024,
        phase="phase6_tools", subphase="core",
        resource_type="tool",
        url="https://github.com/TransformerLensOrg/TransformerLens",
        importance=5, is_critical=True
    ),
    Resource(
        id=80, title="Captum",
        authors="PyTorch", year=2024,
        phase="phase6_tools", subphase="core",
        resource_type="tool",
        url="https://github.com/pytorch/captum",
        importance=4
    ),
    Resource(
        id=81, title="SHAP Library",
        authors="Lundberg", year=2024,
        phase="phase6_tools", subphase="core",
        resource_type="tool",
        url="https://github.com/shap/shap",
        importance=4
    ),
    Resource(
        id=82, title="Hugging Face Transformers",
        authors="Hugging Face", year=2024,
        phase="phase6_tools", subphase="core",
        resource_type="tool",
        url="https://github.com/huggingface/transformers",
        importance=5
    ),
    Resource(
        id=83, title="CircuitsVis",
        authors="TransformerLens", year=2024,
        phase="phase6_tools", subphase="visualization",
        resource_type="tool",
        url="https://github.com/TransformerLensOrg/CircuitsVis",
        importance=4
    ),
    Resource(
        id=84, title="SAELens",
        authors="Joseph Bloom", year=2024,
        phase="phase6_tools", subphase="core",
        resource_type="tool",
        url="https://github.com/jbloomAus/SAELens",
        importance=3
    ),
    Resource(
        id=85, title="BertViz",
        authors="Jesse Vig", year=2024,
        phase="phase6_tools", subphase="visualization",
        resource_type="tool",
        url="https://github.com/jessevig/bertviz",
        importance=4
    ),
    Resource(
        id=86, title="Plotly",
        authors="Plotly", year=2024,
        phase="phase6_tools", subphase="visualization",
        resource_type="tool",
        url="https://plotly.com/python/",
        importance=3
    ),
    Resource(
        id=87, title="Weights & Biases",
        authors="W&B", year=2024,
        phase="phase6_tools", subphase="tracking",
        resource_type="tool",
        url="https://wandb.ai/",
        importance=4
    ),
    Resource(
        id=88, title="nbstripout",
        authors="kynan", year=2024,
        phase="phase6_tools", subphase="tracking",
        resource_type="tool",
        url="https://github.com/kynan/nbstripout",
        importance=3
    ),
    Resource(
        id=89, title="PyTest",
        authors="PyTest", year=2024,
        phase="phase6_tools", subphase="tracking",
        resource_type="tool",
        url="https://docs.pytest.org/",
        importance=3
    ),
    Resource(
        id=90, title="Black + isort + mypy",
        authors="Various", year=2024,
        phase="phase6_tools", subphase="tracking",
        resource_type="tool",
        url="https://black.readthedocs.io/",
        importance=3
    ),

    # ========== PHASE 7: WRITING (10) ==========
    Resource(
        id=91, title="How to Write a Great Research Paper",
        authors="Simon Peyton Jones", year=2016,
        phase="phase7_writing", subphase="academic",
        resource_type="video",
        url="https://www.youtube.com/watch?v=WP-FkUaOcOM",
        youtube_id="WP-FkUaOcOM",
        importance=5
    ),
    Resource(
        id=92, title="Writing in the Sciences",
        authors="Stanford", year=2024,
        phase="phase7_writing", subphase="academic",
        resource_type="docs",
        url="https://www.coursera.org/learn/sciwrite",
        importance=4
    ),
    Resource(
        id=93, title="The Science of Scientific Writing",
        authors="Gopen & Swan", year=1990,
        phase="phase7_writing", subphase="academic",
        resource_type="blog",
        url="https://www.americanscientist.org/blog/the-long-view/the-science-of-scientific-writing",
        importance=4
    ),
    Resource(
        id=94, title="Overleaf LaTeX Templates",
        authors="Overleaf", year=2024,
        phase="phase7_writing", subphase="latex",
        resource_type="tool",
        url="https://www.overleaf.com/latex/templates",
        importance=4
    ),
    Resource(
        id=95, title="Target Venues Reference",
        authors="Various", year=2024,
        phase="phase7_writing", subphase="venues",
        resource_type="docs",
        url="https://aclrollingreview.org/",
        importance=4
    ),
    Resource(
        id=96, title="How to Write Good Reviews",
        authors="EMNLP", year=2020,
        phase="phase7_writing", subphase="venues",
        resource_type="blog",
        url="https://2020.emnlp.org/blog/2020-05-17-write-good-reviews",
        importance=3
    ),
    Resource(
        id=97, title="Semantic Scholar",
        authors="AI2", year=2024,
        phase="phase7_writing", subphase="venues",
        resource_type="tool",
        url="https://www.semanticscholar.org/",
        importance=4
    ),
    Resource(
        id=98, title="Overleaf",
        authors="Overleaf", year=2024,
        phase="phase7_writing", subphase="latex",
        resource_type="tool",
        url="https://www.overleaf.com/",
        importance=5
    ),
    Resource(
        id=99, title="Zotero + Better BibTeX",
        authors="Zotero", year=2024,
        phase="phase7_writing", subphase="latex",
        resource_type="tool",
        url="https://www.zotero.org/",
        importance=4
    ),
    Resource(
        id=100, title="Draw.io / Excalidraw",
        authors="Various", year=2024,
        phase="phase7_writing", subphase="latex",
        resource_type="tool",
        url="https://app.diagrams.net/",
        importance=4
    ),
]


# ============================================================
# DOWNLOAD FUNCTIONS
# ============================================================

def create_directories(base_path: str) -> Dict[str, Path]:
    """Create phase-organized directory structure."""
    phases = [
        "phase0_competitors",
        "phase1_foundations",
        "phase2_mech_interp",
        "phase3_baselines",
        "phase4_explanation_gen",
        "phase5_evaluation",
        "phase6_tools",
        "phase7_writing",
    ]
    
    dirs = {}
    base = Path(base_path)
    
    for phase in phases:
        phase_dir = base / "pdfs" / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        dirs[phase] = phase_dir
    
    # Additional directories
    (base / "web_articles").mkdir(exist_ok=True)
    (base / "youtube_transcripts").mkdir(exist_ok=True)
    (base / "vector_store").mkdir(exist_ok=True)
    
    return dirs


def download_arxiv_paper(arxiv_id: str, output_path: Path) -> bool:
    """Download paper from arXiv."""
    if not ARXIV_AVAILABLE:
        print(f"  ⚠ arxiv library not installed, skipping {arxiv_id}")
        return False
    
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        paper.download_pdf(dirpath=str(output_path.parent), filename=output_path.name)
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {arxiv_id}: {e}")
        return False


def download_direct_pdf(url: str, output_path: Path) -> bool:
    """Download PDF directly from URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {url}: {e}")
        return False


def scrape_blog_post(url: str, output_path: Path) -> bool:
    """Scrape blog post content and save as text."""
    if not BS4_AVAILABLE:
        print(f"  ⚠ beautifulsoup4 not installed, skipping {url}")
        return False
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove scripts and styles
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        output_path.write_text(text, encoding='utf-8')
        return True
    except Exception as e:
        print(f"  ✗ Failed to scrape {url}: {e}")
        return False


def download_youtube_transcript(video_id: str, output_path: Path) -> bool:
    """Download YouTube transcript."""
    if not YOUTUBE_AVAILABLE:
        print(f"  ⚠ youtube-transcript-api not installed, skipping {video_id}")
        return False
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = '\n'.join([entry['text'] for entry in transcript])
        output_path.write_text(text, encoding='utf-8')
        return True
    except Exception as e:
        print(f"  ✗ Failed to get transcript for {video_id}: {e}")
        return False


def download_all_resources(base_path: str = "data", 
                           skip_existing: bool = True,
                           only_critical: bool = False) -> Dict:
    """Download all resources organized by phase."""
    
    print("=" * 60)
    print("THESIS RESOURCE DOWNLOADER")
    print("=" * 60)
    
    dirs = create_directories(base_path)
    
    stats = {
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "by_type": {"paper": 0, "blog": 0, "video": 0, "tool": 0, "docs": 0}
    }
    
    resources_to_process = RESOURCES
    if only_critical:
        resources_to_process = [r for r in RESOURCES if r.is_critical]
        print(f"\n📌 Processing only CRITICAL resources ({len(resources_to_process)})")
    
    for resource in resources_to_process:
        print(f"\n[{resource.id}] {resource.title[:50]}...")
        
        # Determine output path
        if resource.resource_type == "paper":
            safe_name = f"{resource.id:03d}_{resource.bibtex_key or 'paper'}_{resource.year}.pdf"
            output_path = dirs.get(resource.phase, Path(base_path) / "pdfs") / safe_name
        elif resource.resource_type == "blog":
            safe_name = f"{resource.id:03d}_{resource.authors.split()[0].lower()}_{resource.year}.txt"
            output_path = Path(base_path) / "web_articles" / safe_name
        elif resource.resource_type == "video" and resource.youtube_id:
            safe_name = f"{resource.id:03d}_{resource.youtube_id}.txt"
            output_path = Path(base_path) / "youtube_transcripts" / safe_name
        else:
            print(f"  ⏭ Skipping {resource.resource_type} (not downloadable)")
            stats["skipped"] += 1
            continue
        
        # Skip if exists
        if skip_existing and output_path.exists():
            print(f"  ✓ Already exists")
            stats["skipped"] += 1
            resource.local_path = str(output_path)
            continue
        
        # Download based on type
        success = False
        
        if resource.arxiv_id:
            print(f"  📥 Downloading from arXiv: {resource.arxiv_id}")
            success = download_arxiv_paper(resource.arxiv_id, output_path)
        elif resource.resource_type == "paper" and resource.url.endswith(".pdf"):
            print(f"  📥 Downloading PDF directly")
            success = download_direct_pdf(resource.url, output_path)
        elif resource.resource_type == "blog":
            print(f"  📥 Scraping blog post")
            success = scrape_blog_post(resource.url, output_path)
        elif resource.youtube_id:
            print(f"  📥 Getting YouTube transcript")
            success = download_youtube_transcript(resource.youtube_id, output_path)
        
        if success:
            print(f"  ✓ Saved to {output_path}")
            stats["downloaded"] += 1
            stats["by_type"][resource.resource_type] += 1
            resource.local_path = str(output_path)
        else:
            stats["failed"] += 1
        
        # Be nice to servers
        time.sleep(1)
    
    # Save manifest
    manifest_path = Path(base_path) / "resources_manifest.json"
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "total_resources": len(RESOURCES),
        "stats": stats,
        "resources": [asdict(r) for r in RESOURCES]
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n📋 Manifest saved to {manifest_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✓ Downloaded: {stats['downloaded']}")
    print(f"⏭ Skipped:    {stats['skipped']}")
    print(f"✗ Failed:     {stats['failed']}")
    print(f"\nBy type: {stats['by_type']}")
    
    return stats


def get_critical_resources() -> List[Resource]:
    """Get list of critical resources for thesis."""
    return [r for r in RESOURCES if r.is_critical]


def get_resources_by_phase(phase: str) -> List[Resource]:
    """Get resources for a specific phase."""
    return [r for r in RESOURCES if r.phase == phase]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download thesis resources")
    parser.add_argument("--base-path", default="data", help="Base directory for downloads")
    parser.add_argument("--critical-only", action="store_true", help="Download only critical resources")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    
    args = parser.parse_args()
    
    download_all_resources(
        base_path=args.base_path,
        skip_existing=not args.force,
        only_critical=args.critical_only
    )
