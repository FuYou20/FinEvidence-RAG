
from src.retriever import index_document,retrieve
from src.embedder import qwen_embed_batch,embed_chunks

pdf_path  = "data/pdf/传音控股--奔跑的新兴市场，传音的长期主义--2022-04-22.pdf"


index,metadata = index_document(doc_id="report_001",pdf_path=pdf_path,embed_batch=qwen_embed_batch, model_name="qwen3.7-text-embedding",chunk_size = 512,overlap = 64,batch_size = 20)

query = "传音控股在非洲市场具有哪些竞争优势？"

res = retrieve(query=query,index=index,embed_batch=qwen_embed_batch,metadata=metadata,top_k=5)

print(res)
    