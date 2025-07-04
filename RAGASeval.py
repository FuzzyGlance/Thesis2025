import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness
)


class RAGASEvaluator:
    def __init__(self, rag_system):
        self.rag = rag_system
    
    def prepare_dataset(self, questions: list[str], ground_truths: list[str]) -> pd.DataFrame:
        """
        Prepare evaluation dataset by running queries through the RAG system
        """
        results = []
        for question, ground_truth in zip(questions, ground_truths):
            response = self.rag.query(question)
            results.append({
                'question': question,
                'answer': response['answer'],
                'contexts': [doc['text'] for doc in response['documents']],
                'ground_truth': ground_truth
            })
        return pd.DataFrame(results)
    
    def evaluate(self, questions: list[str], ground_truths: list[str]) -> dict:
        """
        Evaluate the RAG system using RAGAS metrics
        """
        dataset = self.prepare_dataset(questions, ground_truths)
        
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
                answer_correctness
            ],
        )
        
        return result

# Example usage with your existing RAG system
if __name__ == "__main__":
    # Initialize your existing RAG system
    from RAGgpt import OpenAIRAG
    rag = OpenAIRAG()
    
    # Prepare evaluation questions and ground truths
    evaluation_questions = [
        "What is the total number of features detected in sample 1?",
        "Which genus had the highest abundance in sample 2?",
        "What phylum is dominant in sample 3",
        "Is feature 27 present in sample 4?",
        "What is the NCBI taxonomy of feature 2?",

        "Which feature contributes the most to the total reads in sample 5?",
        "Is the genus with the highest abundance also the dominant phylum?",
        "Which feature among the top 5 in sample 6 has the lowest abundance?",
        "What is the percentage contribution of Feature1 to the total reads in sample 7?",
        "Does sample 8 not have any abundant features with genus 'microbacter'?",
        
        "If the experiment focuses on rare genera, which top feature from Sample 3 qualifies as rare?",
        "If we only consider genera with known genus names, how many top features remain in sample 3?",
        "If proteobacteria is excluded, which phylum has the highest abundance?",
        "If feature 27s abundance is 0 in one row and 23 in another, should it be considered active?",
        "If Feature1 were removed, which genus would become most abundant in sample 6?",

        "What can you tell me about the dominant phylum in Sample 7?",
        "I am curious, which feature was most abundant in Sample 8",
        "Can you name a genus from Sample 5 that appears more than once in the top features?",
        "Do you know how many reads Sample 8 has in total?",
        "Could you tell me if Feature 27 is relevant in Sample 2 at all?"
    ]
    
    ground_truths = [
        "3080",
        "aeromonas",
        "proteobacteria",
        "yes",
        "NCBITaxon_2559773",

        "Feature1 (aeromonas)",
        "yes, aeromonas belongs to proteobacteria",
        "Feature22 (dicarboxylicus)",
        "1.8",
        "no",

        "Feature24 (aquitalea)",
        "4",
        "firmicutes",
        "yes",
        "Feature10 (herbaspirillum)",

        "yes",
        "proteobacteria",
        "Feature1",
        "aeromonas",
        "1,381,219",
        "no it is not"
    ]
    
    # Run evaluation
    evaluator = RAGASEvaluator(rag)
    evaluation_results = evaluator.evaluate(evaluation_questions, ground_truths)
    
    # Print results
    print("RAGAS Evaluation Results:")
    print(evaluation_results)