from transformers import pipeline
import os
import json
from typing import List, Dict, Any

_CONTEXTS_CACHE: Dict[str, List[Dict[str, Any]]] = {}

def load_trained_model():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_dir, "modelo_treinado")

    if os.path.exists(model_path):
        qa_pipeline = pipeline(
            "question-answering",
            model=model_path,
            tokenizer=model_path,
            local_files_only=True
        )
    else:
        qa_pipeline = pipeline(
            "question-answering",
            model="pierreguillou/bert-large-cased-squad-v1.1-portuguese"
        )

    return qa_pipeline

def _extract_contexts_from_obj(obj: Any, fname: str) -> List[Dict[str, str]]:
    contexts = []
    if isinstance(obj, str):
        if len(obj.strip()) > 20:
            contexts.append({"context": obj.strip(), "source": fname})
        return contexts

    if isinstance(obj, dict):
        # prefer common keys
        for key in ("context", "answer", "text", "passage", "body"):
            v = obj.get(key)
            if isinstance(v, str) and len(v.strip()) > 20:
                contexts.append({"context": " ".join(v.split()), "source": f"{fname}:{key}", "meta": obj.get("question") or obj.get("qid")})
                return contexts
        # fallback: pick longest string field
        longest = None
        for k, v in obj.items():
            if isinstance(v, str):
                if longest is None or len(v) > len(longest[1]):
                    longest = (k, v)
        if longest and len(longest[1].strip()) > 20:
            contexts.append({"context": " ".join(longest[1].split()), "source": f"{fname}:{longest[0]}"})
        return contexts

    if isinstance(obj, list):
        for item in obj:
            contexts.extend(_extract_contexts_from_obj(item, fname))
        return contexts

    return contexts

def _load_contexts_from_model_dir(model_dir: str, max_files: int = 50, max_contexts: int = 500) -> List[Dict[str, str]]:
    if model_dir in _CONTEXTS_CACHE:
        return _CONTEXTS_CACHE[model_dir]

    contexts: List[Dict[str, str]] = []
    if not os.path.isdir(model_dir):
        _CONTEXTS_CACHE[model_dir] = contexts
        return contexts

    files = [f for f in os.listdir(model_dir) if f.lower().endswith((".json", ".jsonl"))]
    files = files[:max_files]

    for fname in files:
        path = os.path.join(model_dir, fname)
        try:
            if fname.lower().endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        contexts.extend(_extract_contexts_from_obj(obj, fname))
            else:
                with open(path, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                    # if top-level dict with 'data' or 'records', drill into it
                    if isinstance(obj, dict) and ("data" in obj and isinstance(obj["data"], list)):
                        items = obj["data"]
                    elif isinstance(obj, dict) and ("records" in obj and isinstance(obj["records"], list)):
                        items = obj["records"]
                    elif isinstance(obj, list):
                        items = obj
                    else:
                        items = [obj]
                    for it in items:
                        contexts.extend(_extract_contexts_from_obj(it, fname))
        except Exception:
            # ignore problematic files
            continue

        # deduplicate and limit
        unique = []
        seen = set()
        for c in contexts:
            key = (c["context"][:200])
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)
            if len(unique) >= max_contexts:
                break
        contexts = unique
        if len(contexts) >= max_contexts:
            break

    _CONTEXTS_CACHE[model_dir] = contexts
    return contexts

def get_treatment(qa_pipeline, symptom: str, top_k: int = 1, max_contexts_to_search: int = 200) -> List[Dict[str, Any]]:
    if not symptom or not isinstance(symptom, str):
        return []

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_dir, "modelo_treinado")

    # load candidate contexts from json files inside modelo_treinado
    candidates = _load_contexts_from_model_dir(model_dir, max_files=100, max_contexts=max_contexts_to_search)
    results: List[Dict[str, Any]] = []
    question = f"Qual é o tratamento indicado para {symptom}?"

    # if no candidates found, fall back to single inference using symptom as context
    if not candidates:
        try:
            out = qa_pipeline(question=question, context=symptom)
            if isinstance(out, list):
                out = out[0]
            return [{
                "answer": out.get("answer", ""),
                "score": float(out.get("score", 0.0)),
                "source": "fallback_symptom_context",
                "source_snippet": symptom[:400]
            }]
        except Exception:
            return []

    for cand in candidates:
        ctx = cand.get("context", "")
        if not ctx:
            continue
        try:
            out = qa_pipeline(question=question, context=ctx)
        except Exception:
            continue
        if isinstance(out, list):
            out = out[0]
        results.append({
            "answer": out.get("answer", ""),
            "score": float(out.get("score", 0.0)),
            "source": cand.get("source"),
            "source_snippet": ctx[:400]
        })

    if not results:
        return []

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max(1, top_k)]