from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import duckdb
import numpy as np
import json
import torch
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness
)
import pandas as pd

class LlamaRAG:
    def __init__(self, db_path="document_vectors.db", faiss_path="docs.faiss"):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
        
        # Initialize retrieval database - EXACTLY as in your working code
        self.conn = duckdb.connect(db_path)
        self.index = faiss.read_index(faiss_path)
        
        # Initialize Llama model - using your exact working configuration
        self.llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using multilingual-e5-large"""
        input_text = f"query: {text}" if not text.startswith("query: ") else text
        return self.embedding_model.encode(input_text, convert_to_tensor=False)

    # EXACTLY your retrieval method from working code
    def retrieve(self, query_text: str, top_k: int = 3) -> dict:
        """Robust retrieval with proper error handling"""
        try:
            query_embedding = self._get_embedding(query_text)
            distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), top_k * 2)

            faiss_ids = [int(i) for i in indices[0]]
            faiss_docs = self.conn.execute(f"""
                SELECT d.id, d.text, d.metadata, d.source_type, d.source_id 
                FROM documents d
                WHERE d.id IN ({','.join(map(str, faiss_ids))})
            """).fetchall()

            # Extract sample_ids from FAISS metadata to try to fetch summary
            sample_ids = set()
            for doc in faiss_docs:
                try:
                    metadata = json.loads(doc[2]) if doc[2] else {}
                    sample_id = doc[4] if doc[3] == "sample_summary" else metadata.get("sample_id")
                    if sample_id:
                        sample_ids.add(str(sample_id))
                except:
                    continue

            summary_docs = []
            if sample_ids:
                summary_docs = self.conn.execute(f"""
                    SELECT id, text, metadata, source_type, source_id FROM documents
                    WHERE source_type = 'sample_summary' AND source_id IN ({','.join([f"'{sid}'" for sid in sample_ids])})
                """).fetchall()

            # Deduplicate summary + FAISS docs
            seen_ids = set()
            documents = []
            for doc in summary_docs + faiss_docs:
                if doc[0] in seen_ids:
                    continue
                seen_ids.add(doc[0])
                try:
                    metadata = json.loads(doc[2]) if doc[2] else {}
                    doc_data = {
                        'id': doc[0],
                        'text': doc[1],
                        'metadata': metadata,
                        'source_type': doc[3],
                        'source_id': doc[4],
                        'score': 1.0 if doc in summary_docs else float(distances[0][faiss_ids.index(doc[0])])
                    }
                    documents.append(doc_data)
                except Exception as e:
                    print(f"Error processing doc {doc[0]}: {str(e)}")

            documents = sorted(documents, key=lambda x: -x['score'])[:top_k]

            # Retrieve sample features
            sample_ids = [d["metadata"].get("sample_id") or d["source_id"] for d in documents if d["source_type"] == "sample_summary" or d["metadata"].get("sample_id")]
            sample_features = {}

            if sample_ids:
                features = self.conn.execute(f"""
                    SELECT sample_id, feature_id, abundance, kingdom, phylum, class, genus
                    FROM features
                    WHERE sample_id IN ({','.join([f"'{sid}'" for sid in sample_ids])})
                    ORDER BY abundance DESC
                    LIMIT 50
                """).fetchall()

                for feature in features:
                    sid = feature[0]
                    if sid not in sample_features:
                        sample_features[sid] = []
                    sample_features[sid].append({
                        'feature_id': feature[1],
                        'abundance': feature[2],
                        'taxonomy': {
                            'kingdom': feature[3],
                            'phylum': feature[4],
                            'class': feature[5],
                            'genus': feature[6]
                        }
                    })

            return {
                'documents': documents,
                'sample_features': sample_features
            }

        except Exception as e:
            print(f"Retrieval failed: {str(e)}")
            return {'documents': [], 'sample_features': {}}

    # Your exact generate method from working code
    def generate(self, query: str, context: str) -> str:
        try:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a scientific assistant. Answer the user's question using ONLY the context below.
    If the answer is not in the context, say "I don't know".<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context:
    {context}

    Question: {query}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>"""

            inputs = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4000
            ).to(self.llm_model.device)

            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                eos_token_id=self.llm_tokenizer.eos_token_id,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )

            output_text = self.llm_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            return output_text.strip().split("\n")[0] or "I don't know"

        except Exception as e:
            print(f"Generation failed: {str(e)}")
            return "I encountered an error processing this request."



    def query(self, query_text: str, top_k: int = 3) -> dict:
        """Full RAG pipeline - using your exact retrieval and generation"""
        retrieval = self.retrieve(query_text, top_k)
        context = "\n".join([
            f"Document {i+1} ({doc['source_type']}): {doc['text']}"
            for i, doc in enumerate(retrieval['documents'])
        ])

        #################################################################################
        
        if retrieval['sample_features']:
            context += "\n\nSample Features:\n" + "\n".join([
                f"Sample {sample_id} has {len(features)} features"
                for sample_id, features in retrieval['sample_features'].items()
            ])

        return {
            'answer': self.generate(query_text, context),
            'documents': retrieval['documents'],
            'sample_features': retrieval['sample_features']
        }
    
    def evaluate_with_ragas(self, query_text: str, ground_truth: str = None):
        """Complete RAGAS evaluation using your exact retrieval/generation"""
        response = self.query(query_text)

        if not response["documents"]:
            raise ValueError("No documents retrieved — cannot evaluate RAGAS metrics.")

        data = {
            "question": [query_text],
            "answer": [response["answer"]],
            "contexts": [[doc["text"] for doc in response["documents"]]],
        }
        print(data["contexts"])

        if ground_truth:
            data["ground_truth"] = [ground_truth]

        if not response["answer"]:
            response["answer"] = "I don't know"

        metrics = [faithfulness, answer_relevancy]
        if ground_truth:
            metrics.extend([context_recall, answer_correctness])
        else:
            metrics.append(context_precision)

        try:
            result = evaluate(Dataset.from_dict(data), metrics=metrics)

            results = {
                "question": query_text,
                "answer": response["answer"],
            }
            for metric in metrics:
                score = result[metric.name][0]
                results[metric.name] = float(score) if score is not None else None

            print("\nEvaluation Results:")
            for key, value in results.items():
                print(f"{key:>20}: {value:.4f}" if isinstance(value, float) else f"{key:>20}: {value}")

            return results

        except Exception as e:
            print(f"RAGAS evaluation failed: {str(e)}")
            return {
                "question": query_text,
                "answer": response["answer"],
                "error": str(e)
            }

# Initialize RAG system
rag = LlamaRAG()

# Your EXACT test questions from original code
test_questions = [("What is the total number of features detected in sample 1?", "3080"),
    ("Which genus had the highest abundance in sample 2?", "aeromonas/genus"),
    ("What phylum is dominant in sample 3", "proteobacteria"),
    ("Is feature 27 present in sample 4?", "yes"),
    ("What is the NCBI taxonomy of feature 2?", "NCBITaxon_2559773"),
    
    ("Which feature contributes the most to the total reads in sample 5?", "Feature1 (aeromonas)"),
    ("Is the genus with the highest abundance also the dominant phylum?", "yes, aeromonas belongs to proteobacteria"),
    ("Which feature among the top 5 in sample 6 has the lowest abundance?", "Feature22 (dicarboxylicus)"),
    ("What is the percentage contribution of Feature1 to the total reads in sample 7?", "1.8"),
    ("Does sample 8 not have any abundant features with genus 'microbacter'?", "no"),

    ("If the experiment focuses on rare genera, which top feature from Sample 3 qualifies as rare?", "Feature24 (aquitalea)"),
    ("If we only consider genera with known genus names, how many top features remain in sample 3?", "4"),
    ("If proteobacteria is excluded, which phylum has the highest abundance?", "firmicutes"),
    ("If feature 27s abundance is 0 in one row and 23 in another, should it be considered active?", "yes"),
    ("If Feature1 were removed, which genus would become most abundant in sample 6?", "Feature10 (herbaspirillum)"),

    ("What can you tell me about the dominant phylum in Sample 7?", "yes"),
    ("I am curious, which feature was most abundant in Sample 8", "proteobacteria"),
    ("Can you name a genus from Sample 5 that appears more than once in the top features?", "Feature1"),
    ("Do you know how many reads Sample 8 has in total?", "aeromonas"),
    ("Could you tell me if Feature 27 is relevant in Sample 2 at all?", "(1,381,219)")
]

# Full evaluation using your exact questions and retrieval
results = [] 
for i, (question, truth) in enumerate(test_questions):
    print(f"\nEvaluating {i + 1}/{len(test_questions)}: {question}")
    try:
        result = rag.evaluate_with_ragas(question, truth)
        results.append(result)
    except Exception as e:
        print(f"Failed on question {i + 1}: {e}")
        results.append({
            "question": question,
            "answer": "ERROR",
            "faithfulness": None,
            "answer_relevancy": None,
            "context_recall": None,
            "answer_correctness": None,
            "context_precision": None,
            "error": str(e)
        })

# Convert to DataFrame and show
df_results = pd.DataFrame(results)
print("\n=== Full Evaluation Summary ===")
print(df_results[["question", "answer", "faithfulness", "answer_relevancy", "context_recall", "answer_correctness"]])
df_results.to_csv("evaluation_results_llama.csv", index=False)

