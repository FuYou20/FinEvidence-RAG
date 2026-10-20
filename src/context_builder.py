

def build_context(retrieval_results):
    # 返回rag的得到的上下文格式
    if not retrieval_results:
        return ""

    sorted_results = sorted(
        retrieval_results,
        key=lambda result: result["rank"],
    )
    context_blocks = []
    for idx,result in enumerate(sorted_results):
        doc_id = result["doc_id"]
        page_number = result["page_number"]
        chunk_id = result["chunk_id"]
        text = result["text"]
        content_type = result.get("content_type", "text")
        type_label = "表格" if content_type == "table" else "正文"
        result_str = f"[S{idx+1}]\n文档:{doc_id}\n页码:{page_number}\n证据类型:{type_label}\n块编号:{chunk_id}\n文本:{text}"
        context_blocks.append(result_str)
    result = "\n\n".join(context_blocks)
    return result
        
