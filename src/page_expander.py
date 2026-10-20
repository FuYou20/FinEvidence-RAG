


def expand_neighbor_pages(
    retrieved_pages,
    metadata,
    *,
    top_k=5,
    seed_top_n=3,
    window=1,
    decay=0.8,
):
    """
    扩展相邻页面
    
    """
    
    page_number = retrieved_pages["page_number"]
    doc_id = retrieved_pages["doc_id"]
    seed_score = retrieved_pages["score"]
    start_page_num = page_number - window
    end_page_num  = page_number + window
    
    page_catalog = {
    (item["doc_id"], item["page_number"]): item
    for item in metadata
}
    neighbor_pages = []
    for page_num in range(start_page_num,end_page_num+1):
        if page_num == page_number:
            continue
        neighbor_key = (doc_id, page_num)
        if neighbor_key in page_catalog:
            
            neighbor_pages.append({"page_num":neighbor_key[1],"neighbor_score":seed_score * decay * abs(page_number - page_num)})
            
    top_k_neighbor_pages = neighbor_pages.sort(key=lambda x :x["neighbor_score"],reverse=True)[:top_k]
    
    
            

    
    
    