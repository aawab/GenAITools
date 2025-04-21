import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_unique_ctx_examples(squad, n=500):
    context2idx = {}
    for i, entry in enumerate(squad['validation']):
        if not entry['context'] in context2idx:
            context2idx[entry['context']] = []
        context2idx[entry['context']].append(i)
    
    queries, contexts, answers = [], [], []
    for k, v in context2idx.items():
        idx = v[0]
        queries.append(squad['validation'][idx]['question'])
        contexts.append(squad['validation'][idx]['context'])
        answers.append(squad['validation'][idx]['answers'])
        if len(queries) == n:
            break
    
    return queries, contexts, answers

def retrieve(contexts, embeddings, query):
    # Encode the query using Sentence Transformer
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute cosine similarity between query and all context embeddings
    similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    
    # Get index of highest similarity
    best_idx = torch.argmax(similarities).item()
    
    # Get the retrieved context
    ret_context = contexts[best_idx]
    
    return best_idx, ret_context

def generate_response(model, tokenizer, query, ret_context):
    # Create instruction template
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. "
         "Provide one Answer ONLY to the following query based on the context provided below. "
         "Do not generate or answer any other questions. "
         "Do not make up or infer any information that is not directly stated in the context. "
         "Provide a concise answer."
         f"\nContext: {ret_context}"},
        {"role": "user", "content": query}
    ]
    
    # Format messages according to model's expected format
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with typical parameters for instruction models
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return only the new tokens
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# Main execution
if __name__ == "__main__":
    # Load SQuAD dataset
    print("Loading SQuAD dataset...")
    squad = load_dataset("squad")
    
    # Load Sentence Transformer
    print("Loading Sentence Transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Get unique context examples
    print("Extracting unique context examples...")
    queries, contexts, answers = get_unique_ctx_examples(squad)
    
    # Part 3.1: Context Retrieval
    print("\n--- Part 3.1: Context Retrieval ---")
    
    # Encode all contexts
    print("Encoding contexts...")
    context_embeddings = model.encode(contexts, convert_to_tensor=True)
    
    # Count correct retrievals
    print("Retrieving contexts for each query...")
    correct_retrievals = 0
    predicted_contexts = []
    
    for i, query in enumerate(queries):
        idx, ret_context = retrieve(contexts, context_embeddings, query)
        predicted_contexts.append((i, idx, ret_context))
        
        if idx == i:  # If retrieved context is the original context
            correct_retrievals += 1
    
    # Print retrieval accuracy
    retrieval_accuracy = correct_retrievals / len(queries)
    print(f"Retrieval Accuracy: {correct_retrievals}/{len(queries)} = {retrieval_accuracy:.4f}")
    
    # Part 3.2: Response Generation
    print("\n--- Part 3.2: Response Generation ---")
    
    # Load language model
    print("Loading language model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Phi-3-Mini model with bfloat16 precision
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    try:
        lm_tokenizer = AutoTokenizer.from_pretrained(model_id)
        lm_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading Phi-3 model: {e}")
        print("Attempting to load Llama model instead...")
        try:
            model_id = "meta-llama/Llama-3.2-3B-Instruct"
            lm_tokenizer = AutoTokenizer.from_pretrained(model_id)
            lm_model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        except Exception as e2:
            print(f"Error loading Llama model: {e2}")
            print("Using smaller available model as fallback...")
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            lm_tokenizer = AutoTokenizer.from_pretrained(model_id)
            lm_model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
    
    # Get 5 correct and 5 incorrect retrievals
    correct_idxs = []
    incorrect_idxs = []
    
    for i, original_idx, _ in predicted_contexts:
        if original_idx == i and len(correct_idxs) < 5:
            correct_idxs.append(i)
        elif original_idx != i and len(incorrect_idxs) < 5:
            incorrect_idxs.append(i)
            
        if len(correct_idxs) >= 5 and len(incorrect_idxs) >= 5:
            break
    
    # Generate responses for selected examples
    examples = correct_idxs + incorrect_idxs
    results = []
    
    print("Generating responses for selected examples...")
    for i in examples:
        query = queries[i]
        original_ctx = contexts[i]
        _, retrieved_ctx = retrieve(contexts, context_embeddings, query)
        response = generate_response(lm_model, lm_tokenizer, query, retrieved_ctx)
        
        is_correct = "CORRECT" if original_ctx == retrieved_ctx else "INCORRECT"
        actual_answer = answers[i]['text'][0]
        
        results.append({
            "index": i,
            "query": query,
            "retrieval_status": is_correct,
            "actual_answer": actual_answer,
            "generated_response": response
        })
    
    # Print results to console and save to output file
    output_text = "--- Part 3.1: Context Retrieval ---\n"
    output_text += f"Retrieval Accuracy: {correct_retrievals}/{len(queries)} = {retrieval_accuracy:.4f}\n\n"
    output_text += "--- Part 3.2: Generate Answers ---\n"
    
    for result in results:
        output_text += f"\nExample {result['index']} ({result['retrieval_status']} retrieval):\n"
        output_text += f"Question: {result['query']}\n"
        output_text += f"Actual Answer: {result['actual_answer']}\n"
        output_text += f"Generated Response: {result['generated_response']}\n"
        output_text += "-" * 80 + "\n"
    
    # Save output to file
    with open("a3_p3_lastname_id_OUTPUT.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
    
    print("\nResults saved to a3_p3_lastname_id_OUTPUT.txt")