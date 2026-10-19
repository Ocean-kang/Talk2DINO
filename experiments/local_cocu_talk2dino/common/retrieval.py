"""Cross-image, cross-head top-neighbor search for B/E; FAISS HNSW keeps COCO manageable."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def local_neighbors(heads, ids, topk=50, per_head_search=25, ef=128, batch=128, index_type='hnsw'):
    """Return {ID:[neighborIDs]} and per neighbor matched target/neighbor head info.

    FAISS exact flat index is useful for validation/ablation; HNSW approximate index
    uses less query time. Both store each (image,head) separately and search ALL heads.
    """
    if len(ids)<topk+1: raise ValueError('Need more images than topk (self excluded)')
    try: import faiss
    except ImportError as e:
        raise RuntimeError('Scheme B/E requires faiss-cpu (pip install faiss-cpu)') from e
    normed=[]
    nheads=None
    for iid in ids:
        x=torch.as_tensor(heads[iid]).float()
        if x.ndim!=2: raise ValueError(f'Bad [heads,dim] image {iid}: {x.shape}')
        if nheads is None: nheads=x.shape[0]
        if x.shape[0]!=nheads: raise ValueError('All images must have same head count')
        normed.append(F.normalize(x,dim=-1))
    arr=torch.stack(normed).numpy()
    flat=arr.reshape(-1,arr.shape[-1]).astype('float32')
    if index_type=='flat': index=faiss.IndexFlatIP(flat.shape[-1])
    elif index_type=='hnsw':
        index=faiss.IndexHNSWFlat(flat.shape[-1],32,faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction=120
        index.hnsw.efSearch=ef
    else: raise ValueError('index_type must be flat or hnsw')
    index.add(flat)
    neighbors,details={},{}
    take=max(per_head_search,topk+2)
    for start in range(0,len(ids),batch):
        selected=arr[start:start+batch]
        scores,matches=index.search(selected.reshape(-1,arr.shape[-1]).astype('float32'),take)
        scores=scores.reshape(len(selected),nheads,take)
        matches=matches.reshape(len(selected),nheads,take)
        for local_idx,source in enumerate(ids[start:start+batch]):
            lookup={}
            for h in range(nheads):
                for s,idx in zip(scores[local_idx,h],matches[local_idx,h]):
                    if idx<0: continue
                    iid=ids[idx//nheads]
                    if iid==source: continue
                    match={'image_id':int(iid),'query_head':h,'neighbor_head':int(idx%nheads),'score':float(s)}
                    if iid not in lookup or s>lookup[iid]['score']: lookup[iid]=match
            ranked=sorted(lookup.values(),key=lambda r:(-r['score'],r['image_id']))[:topk]
            neighbors[str(source)]=[x['image_id'] for x in ranked]
            details[str(source)]=ranked
        print(f'[Head Retrieval] {start+len(selected)}/{len(ids)}',flush=True)
    return neighbors, details
