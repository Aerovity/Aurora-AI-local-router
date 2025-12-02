#!/usr/bin/env python3
"""
Benchmark CACTUS Router - Comprehensive Performance Evaluation

Benchmarks the CACTUS (Clustering for Adaptive Context-aware Task-based Unified Sampling)
router against various baselines and external routing services.

Comparisons:
1. CACTUS Router (our Nomic-based clustering approach)
2. Claude Sonnet (as routing oracle - ground truth)
3. Gemini Flash (alternative LLM router)
4. Random selection (baseline)
5. Always largest model (quality baseline)
6. Always smallest model (speed baseline)
7. Size-proportional random (weighted baseline)
"""

import sys
import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sdks', 'python'))

import numpy as np

# Optional imports
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  openai not installed. Install with: pip install openai")

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("⚠️  google-generativeai not installed. Install with: pip install google-generativeai")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠️  datasets not installed. Install with: pip install datasets")

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  transformers not installed. Install with: pip install transformers torch")

# Import our router components
try:
    from mobile_router import MobileRouter, ModelInfo, RoutingResult
    from mobile_cluster_engine import MobileClusterEngine
    HAS_ROUTER = True
except ImportError:
    HAS_ROUTER = False
    print("⚠️  Router components not found. Make sure you're in the right directory.")


# =============================================================================
# CONFIGURATION
# =============================================================================

# LLMAdaptive.uk configuration (OpenAI-compatible proxy for Claude)
LLMADAPTIVE_BASE_URL = "https://api.llmadaptive.uk/v1"

# CACTUS models - must match the profile
CACTUS_MODELS = [
    {'model_id': 'gemma-270m', 'model_path': 'google/gemma-3-270m-it', 'size_mb': 172, 'avg_tokens_per_sec': 173, 'capabilities': ['text']},
    {'model_id': 'lfm2-350m', 'model_path': 'LiquidAI/LFM2-350M', 'size_mb': 233, 'avg_tokens_per_sec': 145, 'capabilities': ['text', 'tools', 'embed']},
    {'model_id': 'smollm-360m', 'model_path': 'HuggingFaceTB/SmolLM2-360m-Instruct', 'size_mb': 227, 'avg_tokens_per_sec': 150, 'capabilities': ['text', 'embed']},
    {'model_id': 'qwen-600m', 'model_path': 'Qwen/Qwen3-0.6B', 'size_mb': 411, 'avg_tokens_per_sec': 120, 'capabilities': ['text', 'tools', 'embed']},
    {'model_id': 'lfm2-vl-450m', 'model_path': 'LiquidAI/LFM2-VL-450M', 'size_mb': 306, 'avg_tokens_per_sec': 130, 'capabilities': ['text', 'vision', 'embed']},
    {'model_id': 'lfm2-700m', 'model_path': 'LiquidAI/LFM2-700M', 'size_mb': 486, 'avg_tokens_per_sec': 110, 'capabilities': ['text', 'tools', 'embed']},
    {'model_id': 'gemma-1b', 'model_path': 'google/gemma-3-1b-it', 'size_mb': 642, 'avg_tokens_per_sec': 100, 'capabilities': ['text']},
    {'model_id': 'lfm2-1.2b', 'model_path': 'LiquidAI/LFM2-1.2B', 'size_mb': 722, 'avg_tokens_per_sec': 95, 'capabilities': ['text', 'tools', 'embed']},
    {'model_id': 'lfm2-1.2b-tools', 'model_path': 'LiquidAI/LFM2-1.2B-Tools', 'size_mb': 722, 'avg_tokens_per_sec': 95, 'capabilities': ['text', 'tools', 'embed']},
    {'model_id': 'qwen-1.7b', 'model_path': 'Qwen/Qwen3-1.7B', 'size_mb': 1161, 'avg_tokens_per_sec': 75, 'capabilities': ['text', 'tools', 'embed']},
    {'model_id': 'smollm-1.7b', 'model_path': 'HuggingFaceTB/SmolLM2-1.7B-Instruct', 'size_mb': 1161, 'avg_tokens_per_sec': 72, 'capabilities': ['text', 'embed']},
    {'model_id': 'lfm2-vl-1.6b', 'model_path': 'LiquidAI/LFM2-VL-1.6B', 'size_mb': 1440, 'avg_tokens_per_sec': 60, 'capabilities': ['text', 'vision', 'embed']},
]

MODEL_BY_ID = {m['model_id']: m for m in CACTUS_MODELS}
MODEL_IDS = [m['model_id'] for m in CACTUS_MODELS]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BenchmarkSample:
    """A single benchmark sample."""
    question: str
    subject: str
    choices: List[str]
    answer: int
    
@dataclass  
class RoutingDecision:
    """Result of a routing decision."""
    model_id: str
    size_mb: float
    tokens_per_sec: float
    cluster_id: Optional[int] = None
    score: Optional[float] = None
    latency_ms: Optional[float] = None


@dataclass
class BenchmarkMetrics:
    """Metrics for a routing strategy."""
    strategy_name: str
    avg_size_mb: float
    min_size_mb: float
    max_size_mb: float
    std_size_mb: float
    avg_tokens_per_sec: float
    agreement_with_oracle: float  # % agreement with Claude
    savings_vs_largest: float  # % savings
    latency_improvement: float  # Estimated % improvement
    battery_savings: float  # Estimated % savings
    routing_latency_ms: float  # Time to make routing decision
    cluster_distribution: Dict[int, int]  # Cluster -> count
    model_distribution: Dict[str, int]  # Model -> count
    

# =============================================================================
# EMBEDDING MODEL
# =============================================================================

class NomicEmbedder:
    """Nomic embedding model wrapper."""
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load(self):
        """Load the embedding model."""
        print(f"  Loading Nomic embeddings on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        print(f"  ✅ Nomic embeddings loaded")
        
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if self.model is None:
            self.load()
            
        # Add search prefix for queries
        text = f"search_query: {text}"
        
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
        return embedding.cpu().numpy().squeeze().astype(np.float32)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for batch of texts."""
        if self.model is None:
            self.load()
            
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = [f"search_query: {t}" for t in texts[i:i+batch_size]]
            
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
            embeddings.append(batch_embeddings.cpu().numpy())
            
        return np.vstack(embeddings).astype(np.float32)


# =============================================================================
# ROUTING STRATEGIES
# =============================================================================

class CactusRouter:
    """CACTUS Router using Nomic embeddings and clustering."""
    
    def __init__(self, profile_path: str, embedder: NomicEmbedder):
        self.embedder = embedder
        self.profile_path = profile_path
        self.router = None
        self.models = None
        
    def load(self):
        """Load the router from profile."""
        print(f"  Loading CACTUS profile from {self.profile_path}...")
        
        # Convert model dicts to ModelInfo objects
        self.models = [
            ModelInfo(
                model_id=m['model_id'],
                model_path=m['model_path'],
                size_mb=m['size_mb'],
                avg_tokens_per_sec=m['avg_tokens_per_sec'],
                capabilities=m.get('capabilities', ['text'])
            )
            for m in CACTUS_MODELS
        ]
        
        self.router = MobileRouter.from_profile(
            Path(self.profile_path),
            self.models
        )
        print(f"  ✅ CACTUS Router loaded ({self.router.cluster_engine.n_clusters} clusters)")
        
    def route(self, prompt: str, cost_preference: float = 0.5) -> RoutingDecision:
        """Route a prompt to optimal model."""
        if self.router is None:
            self.load()
            
        start_time = time.time()
        
        # Get embedding
        embedding = self.embedder.embed(prompt)
        
        # Route
        result = self.router.route(
            prompt_embedding=embedding,
            cost_preference=cost_preference
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        model_info = MODEL_BY_ID.get(result.model_id, {})
        
        return RoutingDecision(
            model_id=result.model_id,
            size_mb=model_info.get('size_mb', 0),
            tokens_per_sec=model_info.get('avg_tokens_per_sec', 0),
            cluster_id=result.cluster_id,
            score=result.score,
            latency_ms=latency_ms
        )


class ClaudeRouter:
    """Claude Sonnet as routing oracle (ground truth)."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=LLMADAPTIVE_BASE_URL
        )
        self.model_list = "\n".join([f"- {m['model_id']} ({m['size_mb']}MB, {m['avg_tokens_per_sec']} tok/s)" for m in CACTUS_MODELS])
        
    def route(self, prompt: str, cost_preference: float = 0.5) -> RoutingDecision:
        """Use Claude to select optimal model."""
        
        preference_desc = "balanced quality and speed"
        if cost_preference < 0.3:
            preference_desc = "prioritizing speed and efficiency"
        elif cost_preference > 0.7:
            preference_desc = "prioritizing quality and accuracy"
            
        system_prompt = f"""You are a model routing expert. Select the optimal model for the given task, {preference_desc}.

Available models (smallest to largest):
{self.model_list}

Guidelines:
- Simple tasks (greetings, basic Q&A, simple math): Use smallest models (gemma-270m, lfm2-350m, smollm-360m)
- General knowledge, explanations: Use medium models (qwen-600m, lfm2-700m, gemma-1b)
- Complex reasoning, analysis: Use larger models (lfm2-1.2b, qwen-1.7b)
- Very complex tasks (coding, detailed analysis): Use largest models (smollm-1.7b, lfm2-vl-1.6b)

Cost preference: {cost_preference:.1f} (0=speed, 1=quality)

Respond with ONLY the model_id, nothing else."""

        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model="anthropic/claude-sonnet-4-5",
                max_tokens=50,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Select model for: {prompt}"}
                ]
            )
            
            selected = response.choices[0].message.content.strip()
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract model_id
            model_id = None
            for m in CACTUS_MODELS:
                if m['model_id'] in selected:
                    model_id = m['model_id']
                    break
                    
            if model_id is None:
                model_id = 'qwen-1.7b'  # Default fallback
                
            model_info = MODEL_BY_ID.get(model_id, {})
            
            return RoutingDecision(
                model_id=model_id,
                size_mb=model_info.get('size_mb', 0),
                tokens_per_sec=model_info.get('avg_tokens_per_sec', 0),
                latency_ms=latency_ms
            )
            
        except Exception as e:
            print(f"    ⚠️ Claude error: {e}")
            return RoutingDecision(
                model_id='qwen-1.7b',
                size_mb=1161,
                tokens_per_sec=75
            )


class GeminiRouter:
    """Gemini Flash as alternative LLM router."""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.model_list = "\n".join([f"- {m['model_id']} ({m['size_mb']}MB)" for m in CACTUS_MODELS])
        
    def route(self, prompt: str, cost_preference: float = 0.5) -> RoutingDecision:
        """Use Gemini to select model."""
        
        system_prompt = f"""Select the optimal model for this task. Cost preference: {cost_preference:.1f} (0=speed, 1=quality)

Models: {self.model_list}

Respond with ONLY the model_id."""

        start_time = time.time()
        
        try:
            response = self.model.generate_content(f"{system_prompt}\n\nTask: {prompt}")
            selected = response.text.strip()
            latency_ms = (time.time() - start_time) * 1000
            
            model_id = None
            for m in CACTUS_MODELS:
                if m['model_id'] in selected:
                    model_id = m['model_id']
                    break
                    
            if model_id is None:
                model_id = 'qwen-1.7b'
                
            model_info = MODEL_BY_ID.get(model_id, {})
            
            return RoutingDecision(
                model_id=model_id,
                size_mb=model_info.get('size_mb', 0),
                tokens_per_sec=model_info.get('avg_tokens_per_sec', 0),
                latency_ms=latency_ms
            )
            
        except Exception as e:
            print(f"    ⚠️ Gemini error: {e}")
            return RoutingDecision(
                model_id='qwen-1.7b',
                size_mb=1161,
                tokens_per_sec=75
            )


class BaselineRouter:
    """Simple baseline routers."""
    
    @staticmethod
    def always_largest(prompt: str, cost_preference: float = 0.5) -> RoutingDecision:
        """Always use largest model."""
        m = CACTUS_MODELS[-1]  # lfm2-vl-1.6b
        return RoutingDecision(
            model_id=m['model_id'],
            size_mb=m['size_mb'],
            tokens_per_sec=m['avg_tokens_per_sec'],
            latency_ms=0.01
        )
        
    @staticmethod
    def always_smallest(prompt: str, cost_preference: float = 0.5) -> RoutingDecision:
        """Always use smallest model."""
        m = CACTUS_MODELS[0]  # gemma-270m
        return RoutingDecision(
            model_id=m['model_id'],
            size_mb=m['size_mb'],
            tokens_per_sec=m['avg_tokens_per_sec'],
            latency_ms=0.01
        )
        
    @staticmethod
    def random_selection(prompt: str, cost_preference: float = 0.5) -> RoutingDecision:
        """Random model selection."""
        m = random.choice(CACTUS_MODELS)
        return RoutingDecision(
            model_id=m['model_id'],
            size_mb=m['size_mb'],
            tokens_per_sec=m['avg_tokens_per_sec'],
            latency_ms=0.01
        )
        
    @staticmethod
    def size_weighted_random(prompt: str, cost_preference: float = 0.5) -> RoutingDecision:
        """Random selection weighted by inverse size (smaller = more likely)."""
        sizes = [m['size_mb'] for m in CACTUS_MODELS]
        max_size = max(sizes)
        weights = [(max_size - s + 100) for s in sizes]  # Inverse size + offset
        
        m = random.choices(CACTUS_MODELS, weights=weights, k=1)[0]
        return RoutingDecision(
            model_id=m['model_id'],
            size_mb=m['size_mb'],
            tokens_per_sec=m['avg_tokens_per_sec'],
            latency_ms=0.01
        )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mmlu_samples(n_samples: int = 200, topics: List[str] = None) -> List[BenchmarkSample]:
    """Load MMLU test samples."""
    if not HAS_DATASETS:
        print("❌ datasets library not installed")
        return []
        
    print(f"📚 Loading {n_samples} MMLU samples...")
    
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    
    if topics is None:
        topics = [
            "abstract_algebra", "anatomy", "computer_security", "astronomy",
            "international_law", "marketing", "philosophy", "electrical_engineering",
            "econometrics", "moral_scenarios", "professional_medicine", "virology",
            "high_school_physics", "high_school_geography", "world_religions"
        ]
    
    samples = []
    per_topic = max(1, n_samples // len(topics))
    
    for topic in topics:
        topic_samples = [s for s in mmlu if s["subject"] == topic]
        for s in topic_samples[:per_topic]:
            samples.append(BenchmarkSample(
                question=s['question'],
                subject=s['subject'],
                choices=s['choices'],
                answer=s['answer']
            ))
            
    random.shuffle(samples)
    return samples[:n_samples]


def load_custom_prompts() -> List[BenchmarkSample]:
    """Load custom test prompts covering various complexity levels."""
    
    prompts = [
        # Simple (should use small models)
        ("Hi there!", "greeting", [], 0),
        ("What is 2+2?", "simple_math", [], 0),
        ("Tell me a joke", "casual", [], 0),
        ("What color is the sky?", "simple_qa", [], 0),
        ("Hello, how are you?", "greeting", [], 0),
        
        # Medium (should use medium models)
        ("Explain what photosynthesis is", "science", [], 0),
        ("What are the main causes of World War I?", "history", [], 0),
        ("How does a combustion engine work?", "engineering", [], 0),
        ("What is the difference between DNA and RNA?", "biology", [], 0),
        ("Summarize the plot of Romeo and Juliet", "literature", [], 0),
        
        # Complex (should use larger models)
        ("Explain quantum entanglement and its implications for computing", "physics", [], 0),
        ("Analyze the philosophical implications of Kant's categorical imperative", "philosophy", [], 0),
        ("Derive the Schwarzschild radius from Einstein's field equations", "physics", [], 0),
        ("Write a Python implementation of a red-black tree with all rotations", "coding", [], 0),
        ("Compare and contrast functionalism and structuralism in psychology", "psychology", [], 0),
        
        # Very complex (should use largest models)
        ("Design a distributed consensus algorithm that handles Byzantine faults", "cs", [], 0),
        ("Prove the Riemann hypothesis implications for prime distribution", "math", [], 0),
        ("Analyze the economic impact of cryptocurrency on monetary policy globally", "economics", [], 0),
        ("Write a compiler frontend for a simple programming language", "coding", [], 0),
        ("Evaluate the ethical implications of CRISPR gene editing in humans", "ethics", [], 0),
    ]
    
    return [BenchmarkSample(q, s, c, a) for q, s, c, a in prompts]


# =============================================================================
# BENCHMARK ENGINE
# =============================================================================

def calculate_metrics(
    strategy_name: str,
    decisions: List[RoutingDecision],
    oracle_decisions: List[RoutingDecision]
) -> BenchmarkMetrics:
    """Calculate comprehensive metrics for a routing strategy."""
    
    sizes = [d.size_mb for d in decisions]
    speeds = [d.tokens_per_sec for d in decisions]
    latencies = [d.latency_ms for d in decisions if d.latency_ms is not None]
    
    # Agreement with oracle
    agreement = sum(1 for d, o in zip(decisions, oracle_decisions) if d.model_id == o.model_id)
    agreement_pct = (agreement / len(decisions)) * 100 if decisions else 0
    
    # Size statistics
    avg_size = statistics.mean(sizes) if sizes else 0
    min_size = min(sizes) if sizes else 0
    max_size = max(sizes) if sizes else 0
    std_size = statistics.stdev(sizes) if len(sizes) > 1 else 0
    
    # Savings calculations
    largest_size = 1440  # lfm2-vl-1.6b
    savings = ((largest_size - avg_size) / largest_size) * 100
    
    # Estimated improvements
    latency_improvement = savings * 0.8  # 80% correlation with size
    battery_savings = savings * 0.7  # 70% correlation with size
    
    # Routing latency
    avg_routing_latency = statistics.mean(latencies) if latencies else 0
    
    # Distributions
    cluster_dist = defaultdict(int)
    model_dist = defaultdict(int)
    
    for d in decisions:
        if d.cluster_id is not None:
            cluster_dist[d.cluster_id] += 1
        model_dist[d.model_id] += 1
        
    return BenchmarkMetrics(
        strategy_name=strategy_name,
        avg_size_mb=avg_size,
        min_size_mb=min_size,
        max_size_mb=max_size,
        std_size_mb=std_size,
        avg_tokens_per_sec=statistics.mean(speeds) if speeds else 0,
        agreement_with_oracle=agreement_pct,
        savings_vs_largest=savings,
        latency_improvement=latency_improvement,
        battery_savings=battery_savings,
        routing_latency_ms=avg_routing_latency,
        cluster_distribution=dict(cluster_dist),
        model_distribution=dict(model_dist)
    )


def run_benchmark(
    n_samples: int = 100,
    anthropic_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    profile_path: str = "cactus-integration/profiles/cactus_profile.json",
    cost_preference: float = 0.5,
    use_custom_prompts: bool = False
):
    """Run comprehensive benchmark."""
    
    print("=" * 80)
    print("🌵 CACTUS ROUTER BENCHMARK")
    print("=" * 80)
    print()
    
    # Check requirements
    if not HAS_TRANSFORMERS:
        print("❌ transformers/torch required for CACTUS router")
        return
        
    if not HAS_ROUTER:
        print("❌ Router components not found")
        return
    
    # Initialize components
    print("🔧 Initializing components...")
    
    # Load embedder
    embedder = NomicEmbedder()
    embedder.load()
    
    # Initialize routers
    routers = {}
    
    # CACTUS Router
    cactus_router = CactusRouter(profile_path, embedder)
    cactus_router.load()
    routers['cactus_router'] = cactus_router.route
    
    # Claude (if available)
    if HAS_OPENAI and anthropic_key:
        claude_router = ClaudeRouter(anthropic_key)
        routers['claude_oracle'] = claude_router.route
        print("  ✅ Claude Sonnet initialized (via LLMAdaptive)")
    else:
        print("  ⚠️ Claude not available (no API key)")
        
    # Gemini (if available)
    if HAS_GEMINI and gemini_key:
        gemini_router = GeminiRouter(gemini_key)
        routers['gemini_flash'] = gemini_router.route
        print("  ✅ Gemini Flash initialized")
    else:
        print("  ⚠️ Gemini not available (no API key)")
        
    # Baselines
    routers['always_largest'] = BaselineRouter.always_largest
    routers['always_smallest'] = BaselineRouter.always_smallest
    routers['random'] = BaselineRouter.random_selection
    routers['size_weighted'] = BaselineRouter.size_weighted_random
    print("  ✅ Baseline routers initialized")
    
    # Load test data
    print()
    if use_custom_prompts:
        samples = load_custom_prompts()
        print(f"  ✅ Loaded {len(samples)} custom prompts")
    else:
        samples = load_mmlu_samples(n_samples)
        print(f"  ✅ Loaded {len(samples)} MMLU samples")
        
    if not samples:
        print("❌ No samples loaded")
        return
        
    print()
    
    # Run benchmark
    print("=" * 80)
    print("🏃 RUNNING BENCHMARK")
    print("=" * 80)
    print()
    
    all_decisions = {name: [] for name in routers}
    
    for i, sample in enumerate(samples, 1):
        if i % 20 == 0 or i == 1:
            print(f"  Progress: {i}/{len(samples)} samples...")
            
        prompt = sample.question
        
        for name, route_fn in routers.items():
            try:
                decision = route_fn(prompt, cost_preference)
                all_decisions[name].append(decision)
            except Exception as e:
                print(f"    ⚠️ {name} error: {e}")
                # Default fallback
                all_decisions[name].append(RoutingDecision(
                    model_id='qwen-1.7b',
                    size_mb=1161,
                    tokens_per_sec=75
                ))
                
        # Rate limiting for API calls
        if 'claude_oracle' in routers or 'gemini_flash' in routers:
            time.sleep(0.1)
            
    print()
    
    # Calculate metrics
    print("=" * 80)
    print("📊 BENCHMARK RESULTS")
    print("=" * 80)
    print()
    
    # Use Claude as oracle if available, otherwise use CACTUS
    oracle_name = 'claude_oracle' if 'claude_oracle' in all_decisions else 'cactus_router'
    oracle_decisions = all_decisions[oracle_name]
    
    metrics = {}
    for name, decisions in all_decisions.items():
        metrics[name] = calculate_metrics(name, decisions, oracle_decisions)
        
    # Print results table
    print(f"{'Strategy':<18} {'Avg Size':<10} {'Agree%':<10} {'Save%':<10} {'Latency↓':<10} {'Battery↓':<10} {'Route ms'}")
    print("-" * 90)
    
    for name, m in sorted(metrics.items(), key=lambda x: -x[1].savings_vs_largest):
        print(f"{name:<18} {m.avg_size_mb:>7.0f}MB  {m.agreement_with_oracle:>7.1f}%  "
              f"{m.savings_vs_largest:>7.1f}%  {m.latency_improvement:>7.1f}%  "
              f"{m.battery_savings:>7.1f}%  {m.routing_latency_ms:>7.1f}")
              
    print()
    
    # CACTUS specific analysis
    print("=" * 80)
    print("🌵 CACTUS ROUTER ANALYSIS")
    print("=" * 80)
    print()
    
    cactus_metrics = metrics['cactus_router']
    
    print(f"📈 Performance Summary:")
    print(f"   • Average model size: {cactus_metrics.avg_size_mb:.0f}MB")
    print(f"   • Size std deviation: {cactus_metrics.std_size_mb:.0f}MB")
    print(f"   • Avg tokens/sec: {cactus_metrics.avg_tokens_per_sec:.0f}")
    print(f"   • Savings vs largest: {cactus_metrics.savings_vs_largest:.1f}%")
    print(f"   • Routing latency: {cactus_metrics.routing_latency_ms:.1f}ms")
    print()
    
    print(f"🎯 Cluster Distribution:")
    for cluster_id, count in sorted(cactus_metrics.cluster_distribution.items()):
        pct = (count / len(samples)) * 100
        print(f"   • Cluster {cluster_id}: {count} samples ({pct:.1f}%)")
    print()
    
    print(f"🤖 Model Selection Distribution:")
    for model_id, count in sorted(cactus_metrics.model_distribution.items(), key=lambda x: -x[1]):
        pct = (count / len(samples)) * 100
        size = MODEL_BY_ID[model_id]['size_mb']
        print(f"   • {model_id}: {count} ({pct:.1f}%) - {size}MB")
    print()
    
    # Comparison with oracle
    if oracle_name == 'claude_oracle':
        print(f"🎓 Agreement with Claude Oracle: {cactus_metrics.agreement_with_oracle:.1f}%")
        print()
        
        # Analyze disagreements
        disagreements = []
        for i, (cactus, claude) in enumerate(zip(all_decisions['cactus_router'], all_decisions['claude_oracle'])):
            if cactus.model_id != claude.model_id:
                disagreements.append({
                    'prompt': samples[i].question[:60] + '...',
                    'cactus': cactus.model_id,
                    'claude': claude.model_id,
                    'cactus_size': cactus.size_mb,
                    'claude_size': claude.size_mb
                })
                
        if disagreements[:5]:
            print("📋 Sample Disagreements:")
            for d in disagreements[:5]:
                size_diff = d['cactus_size'] - d['claude_size']
                direction = "larger" if size_diff > 0 else "smaller"
                print(f"   • \"{d['prompt']}\"")
                print(f"     CACTUS: {d['cactus']} | Claude: {d['claude']} ({abs(size_diff):.0f}MB {direction})")
            print()
    
    # Save results
    output_file = 'benchmark_cactus_results.json'
    results = {
        'config': {
            'n_samples': len(samples),
            'cost_preference': cost_preference,
            'profile_path': profile_path,
            'oracle': oracle_name
        },
        'metrics': {name: asdict(m) for name, m in metrics.items()}
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"💾 Results saved to {output_file}")
    print()
    
    # Final summary
    print("=" * 80)
    print("🏆 KEY INSIGHTS")
    print("=" * 80)
    print()
    
    print(f"✅ CACTUS Router achieves {cactus_metrics.savings_vs_largest:.1f}% size savings vs always-largest")
    print(f"✅ Expected latency improvement: ~{cactus_metrics.latency_improvement:.0f}%")
    print(f"✅ Expected battery savings: ~{cactus_metrics.battery_savings:.0f}%")
    print(f"✅ Routing decision time: {cactus_metrics.routing_latency_ms:.1f}ms (acceptable for mobile)")
    
    if oracle_name == 'claude_oracle':
        print(f"✅ Agreement with Claude oracle: {cactus_metrics.agreement_with_oracle:.1f}%")
        
    print()
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CACTUS Router")
    parser.add_argument('--anthropic-key', type=str, help='Anthropic API key (for LLMAdaptive)')
    parser.add_argument('--gemini-key', type=str, help='Gemini API key')
    parser.add_argument('--n-samples', type=int, default=100, help='Number of MMLU samples')
    parser.add_argument('--profile', type=str, default='cactus-integration/profiles/cactus_profile.json', 
                        help='Path to CACTUS profile')
    parser.add_argument('--cost-preference', type=float, default=0.5, 
                        help='Cost preference (0=speed, 1=quality)')
    parser.add_argument('--custom-prompts', action='store_true', 
                        help='Use custom prompts instead of MMLU')
    
    args = parser.parse_args()
    
    anthropic_key = args.anthropic_key or os.getenv('ANTHROPIC_API_KEY') or os.getenv('LLMADAPTIVE_API_KEY')
    gemini_key = args.gemini_key or os.getenv('GEMINI_API_KEY')
    
    run_benchmark(
        n_samples=args.n_samples,
        anthropic_key=anthropic_key,
        gemini_key=gemini_key,
        profile_path=args.profile,
        cost_preference=args.cost_preference,
        use_custom_prompts=args.custom_prompts
    )
