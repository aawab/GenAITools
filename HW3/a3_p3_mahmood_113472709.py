import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

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
    queryEmbed = model.encode(query, convert_to_tensor=True)
    
    cosineSim = util.pytorch_cos_sim(queryEmbed, embeddings)[0]
    
    highestSim = torch.argmax(cosineSim).item()
    
    context = contexts[highestSim]
    
    return highestSim, context

def generate_response(model, tokenizer, query, ret_context):
    # Instruction template

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. "
         "Provide one Answer ONLY to the following query based on the context provided below. "
         "Do not generate or answer any other questions. "
         "Do not make up or infer any information that is not directly stated in the context. "
         "Provide a concise answer."
         f"\nContext: {ret_context}"},
        {"role": "user", "content": query}
    ]
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,) 

    genArgs = { "max_new_tokens": 500, "return_full_text": False, "temperature": 0.0, "do_sample": False,} 

    output = pipe(messages, **genArgs) 
        
    return output[0]['generated_text'].strip()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    squad = load_dataset("squad")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    queries, contexts, answers = get_unique_ctx_examples(squad)
    
    # Checkpoint 3.1
    print("Checkpoint 3.1")
    
    embeddings = model.encode(contexts, convert_to_tensor=True)
    
    # Count correctly retrieved contexts
    correctRetrieved = 0
    predContexts = []
    
    for i, query in enumerate(queries):
        retIndex, retCon = retrieve(contexts, embeddings, query)
        predContexts.append((i, retIndex, retCon))
        
        if retIndex == i:
            correctRetrieved += 1
    
    print(f"Retrieval Accuracy: {correctRetrieved}/{len(queries)}")
    
    # Checkpoint 3.2
    print("\nCheckpoint 3.2")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    phi3 = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")
    
    correctIDs = []
    incorrectIDs = []
    
    for i, (original, retID, _) in enumerate(predContexts):
        if retID == original and len(correctIDs) < 5:
            correctIDs.append(original)
        elif retID != original and len(incorrectIDs) < 5:
            incorrectIDs.append(original)
            
        if len(correctIDs) >= 5 and len(incorrectIDs) >= 5:
            break
    
    exampleIDs = correctIDs + incorrectIDs
    
    for i in exampleIDs:
        query = queries[i]
        originalContext = contexts[i]
        retID, retContext = retrieve(contexts, embeddings, query)
        
        correct = "CORRECT" if retID == i else "INCORRECT"
        realAnswer = answers[i]['text'][0]
        
        response = generate_response(phi3, tokenizer, query, retContext)
        
        print(f"{correct} Example #{i}")
        print(f"Query: {query}")
        print(f"Retrieved Context: {retContext}")
        print(f"LM-Generated Response: {response}")
        print(f"Actual Answer: {realAnswer}\n")
        print("----------------------------------------")