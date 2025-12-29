import json
import pandas as pd
from rag_engine import RAGService
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
rag = RAGService()
judge_llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    temperature=0
)
judge_prompt = ChatPromptTemplate.from_template("""
You are an impartial evaluator for a RAG system.
Compare the AI's Actual Answer with the Ground Truth.
<Question>
{question}
</Question>
<Ground Truth>
{ground_truth}
</Ground Truth>
<Actual Answer>
{actual_answer}
</Actual Answer>
<Task>
Rate the Actual Answer on a scale of 1 to 5:
1: Completely wrong or hallucinated.
3: Partially correct but missing details.
5: Perfectly accurate and faithful to the ground truth.
Output ONLY the number (1, 2, 3, 4, or 5).
</Task>
""")
judge_chain = judge_prompt | judge_llm | StrOutputParser()
def evaluate():
    print("Starting Evaluation Suite...\n")
    with open("data/test_set.json", "r") as f:
        test_data = json.load(f)    
    results = []    
    for item in test_data:
        print(f"Testing: {item['question']}...", end="")
        try:
            actual_answer = rag.ask(item['question'])
            score = judge_chain.invoke({
                "question": item['question'],
                "ground_truth": item['ground_truth'],
                "actual_answer": actual_answer
            })
            score = int(score.strip())           
            print(f" Score: {score}/5")           
            results.append({
                "Question": item['question'],
                "Expected": item['ground_truth'],
                "Actual": actual_answer,
                "Score": score,
                "Type": item['type']
            })           
        except Exception as e:
            print(f" Error: {e}")
    df = pd.DataFrame(results)
    avg_score = df["Score"].mean()
    
    print("\n" + "="*30)
    print(f"FINAL EVALUATION REPORT")
    print("="*30)
    print(f"Average Accuracy Score: {avg_score:.1f} / 5.0")
    print(f"Total Questions: {len(df)}")
    print("="*30)
    df.to_csv("evaluation_results.csv", index=False)
    print("Results saved to 'evaluation_results.csv'")
if __name__ == "__main__":
    evaluate()