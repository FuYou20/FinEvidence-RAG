
from pathlib import Path
import hashlib
import faiss
import json
import copy
def compute_file_sha256(path):
 
    if  isinstance(path,str):
        path = Path(path)
    elif isinstance(path,Path):
        pass
    
    else:
        raise TypeError("path 必须是 str 或者 Path 类型之一")
    
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    hexdigest = h.hexdigest()
    return hexdigest


def read_jsonl(path:str|Path):
    if  isinstance(path,str):
        path = Path(path)
    elif isinstance(path,Path):
            pass
        
    else:
        raise TypeError("path 必须是 str 或者 Path 类型之一")
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line) # 惰性加载，调用时才加载1行数据。避免一次加载整个文件导致内存占用过大
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_no} 行解析失败: {e}")
    
def save_index_bundle(cache_dir,index,metadata,build_config,):
    cache_dir = Path(cache_dir)
    if not isinstance(metadata,list):
        raise TypeError("metadata 必须是 list")
    if not isinstance(build_config,dict):
        raise TypeError("build_config 必须是 dict")
    if  len(metadata) != index.ntotal:
        raise ValueError("metadata 和 index 数量不匹配")
    
    cache_dir.mkdir(parents=True,exist_ok=True)
    faiss.write_index(index, str(cache_dir/"index.faiss"))
    
    with cache_dir.joinpath("metadata.jsonl").open("w", encoding="utf-8") as f:
        for record in metadata:
            f.write(json.dumps(record,ensure_ascii=False) + "\n")
    
    build_config_copy=copy.deepcopy(build_config)
    
    build_config_copy.update({
    "index_size": int(index.ntotal),
    "index_dimension": int(index.d),
})
    
    with cache_dir.joinpath("build_config.json").open("w", encoding="utf-8") as f:
            f.write(json.dumps(build_config_copy,ensure_ascii=False,indent=2))
    
        
def load_index_bundle(cache_dir, expected_config=None):
    cache_dir = Path(cache_dir)
    file_paths = [index_path := cache_dir.joinpath("index.faiss"),metadata_path := cache_dir.joinpath("metadata.jsonl"),build_config_path := cache_dir.joinpath("build_config.json")]
    
    missing_files = [f for f in file_paths if not Path.is_file(f)]
    missing_names = ", ".join(str(path) for path in missing_files)
    if missing_files:
        raise FileNotFoundError(f"缺失文件:{missing_names}")
    
    index = faiss.read_index(str(index_path))
    metadata = []
    
    for obj in read_jsonl(metadata_path):
        metadata.append(obj)
    with build_config_path.open(encoding="utf-8") as f:
        build_config = json.load(f)

    if not isinstance(build_config, dict):
        raise ValueError(
            f"build_config 必须是 JSON 对象, path={build_config_path}"
        )

    required_fields = {"index_size", "index_dimension"}
    missing_fields = required_fields - build_config.keys()
    if missing_fields:
        raise ValueError(
            f"build_config 缺少必要字段: {sorted(missing_fields)}, "
            f"path={build_config_path}"
        )

    if index.ntotal != len(metadata):
        raise ValueError("index.ntotal != len(metadata)")
    if index.d != build_config["index_dimension"]:
        raise ValueError("index.d != build_config[index_dimension]")
    if index.ntotal != build_config["index_size"]:
        raise ValueError("index.ntotal != build_config[index_size]")
    if expected_config is not None:
        if not isinstance(expected_config,dict):
            raise TypeError("expected_config 必须是字典类型")
        for key,value in expected_config.items():
            if key not in build_config.keys():
                raise ValueError(f"build_config 缺少配置项{key},path={build_config_path}")
            if build_config[key] != value:
               raise ValueError(f"build_config 配置项{key}值{build_config[key]}，期望值：{value},path={str(build_config_path)}") 
    return index, metadata, build_config



        
def load_or_build_index(
    cache_dir,
    build_index_fn,
    *,
    build_config,
):
    """
    加载兼容的索引缓存；缓存不可用时重新建库并保存。

    Returns:
        tuple:
            index: FAISS 索引
            metadata: 与索引位置一一对应的 metadata 列表
            cache_status: 缓存状态。命中兼容缓存时为 ``"hit"``，
                缓存缺失、不兼容或损坏并重新构建时为 ``"rebuilt"``。
    """
    # 这里是幂等吗？也就是无论是path 还是 str对象都会得到一个path 对象
    cache_dir = Path(cache_dir)
        
    
    
    if not callable(build_index_fn):
        raise TypeError(f"期待 build_index_fn 是 callable 类型,实际类型:{type(build_index_fn)}")
    if not isinstance(build_config,dict):
        raise TypeError(f"期待 build_config 是 dict 类型,实际类型:{type(build_config)}")
    if build_config == {}:
        raise ValueError("build_config 不可为空dict")
    is_hit = False
    try:
        index,metadata,_ =  load_index_bundle(cache_dir,expected_config=build_config)
        is_hit = True
    except (FileNotFoundError,ValueError) as e:
        index, metadata = build_index_fn()
        save_index_bundle(index=index,metadata=metadata,cache_dir=cache_dir,build_config=build_config)
        
    cache_status = "hit" if is_hit else "rebuilt"
    return index,metadata, cache_status







def compute_corpus_fingerprint(
    corpus_records,
    project_root,
):
    
    
    
    if not isinstance(corpus_records,list):
        raise TypeError("corpus_records必须是list")
    if not corpus_records:
            raise ValueError("corpus_records 不能为空")
    project_root = Path(project_root).resolve()
    if not project_root.is_dir():
        raise ValueError("project_root 不是一个目录地址")
    
    sorted_corpus_records =  sorted(corpus_records,key=lambda x:(x["doc_id"],x["relative_path"]))
    
    file_records = []
    for record in sorted_corpus_records:
        relative_path = Path(record["relative_path"])
        # 得到绝对路径
        abs_path = (project_root/relative_path).resolve()
        if not abs_path.is_relative_to(project_root):
            raise ValueError(f"PDF 路径越过项目目录: {abs_path}")

        if not abs_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {abs_path}")

        if not abs_path.is_file():
            raise ValueError(f"PDF 路径不是文件: {abs_path}")
        
        
        file_sha256 = compute_file_sha256(abs_path)
        file_record = {
            "doc_id": record["doc_id"],
            "relative_path": str(relative_path),
            "file_sha256": file_sha256,
        }
        file_records.append(file_record)
        
        
    file_records = json.dumps(file_records,
               ensure_ascii=False,
               sort_keys=True,
               separators=(",", ":"),)
   
    h = hashlib.sha256(file_records.encode("utf-8")).hexdigest()
    return h
