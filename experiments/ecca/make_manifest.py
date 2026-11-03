"""Build diagnostic manifests from existing COCO or VOC data; GT never enters B."""
import argparse
from pathlib import Path
from .common import read, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='kind',required=True)
    c=sub.add_parser('coco');c.add_argument('--captions',required=True);c.add_argument('--instances',required=True)
    c.add_argument('--ids',help='Optional text file, one image ID per line (e.g. previous 200 samples)')
    v=sub.add_parser('voc');v.add_argument('--voc-root',required=True);v.add_argument('--split-file',required=True)
    for q in (c,v):
        q.add_argument('--output',required=True);q.add_argument('--limit',type=int,default=200)
    a=p.parse_args();out=Path(a.output).resolve();images=[]
    if a.kind=='coco':
        from .coco_masks import semantic_mask, BACKEND
        from PIL import Image
        instances=read(a.instances);data=read(a.captions);caps={}
        image_index={int(rec['id']):rec for rec in instances['images']}
        category_index={int(rec['id']):rec for rec in instances['categories']}
        if len(image_index)!=len(instances['images']):raise ValueError('Duplicate COCO image IDs')
        if len(category_index)!=len(instances['categories']):raise ValueError('Duplicate COCO category IDs')
        for ann in data['annotations']:
            caps.setdefault(int(ann['image_id']),[]).append(ann['caption'])
        chosen=sorted(set(caps)&set(image_index))
        if a.ids:
            requested=[int(x) for x in Path(a.ids).read_text().split()]
            if set(requested)-set(chosen):raise ValueError('Requested IDs missing from captions/instances')
            chosen=requested
        if a.limit:chosen=chosen[:a.limit]
        if not chosen:raise ValueError('No common image IDs in captions/instances')
        cats=sorted(category_index)
        classes=[{'id':i+1,'name':category_index[k]['name']} for i,k in enumerate(cats)]
        lookup={k:i+1 for i,k in enumerate(cats)}
        selected=set(chosen);annotations={iid:[] for iid in chosen}
        for ann in instances['annotations']:
            iid=int(ann['image_id'])
            if iid in selected:annotations[iid].append(ann)
        masks=out.parent/(out.stem+'_masks');masks.mkdir(parents=True,exist_ok=True)
        for iid in chosen:
            rec=image_index[iid]
            gt=semantic_mask(annotations[iid],rec['height'],rec['width'],lookup)
            target=masks/(str(iid)+'.png');Image.fromarray(gt).save(target)
            images.append({'image_id':iid,'file_name':rec['file_name'],'captions':caps[iid],
                           'mask':str(target.relative_to(out.parent))})
            print('manifest mask',iid,flush=True)
        ignore=[0,255]  # COCO instances do not annotate stuff/background.
    else:
        names='aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()
        # Common class spellings match COCO captions/gallery when possible.
        names=['airplane' if x=='aeroplane' else 'dining table' if x=='diningtable' else 'motorcycle' if x=='motorbike'
               else 'potted plant' if x=='pottedplant' else 'tv' if x=='tvmonitor' else x for x in names]
        classes=[{'id':i+1,'name':n} for i,n in enumerate(names)]
        split=Path(a.split_file).read_text().split()
        if a.limit:split=split[:a.limit]
        for i,key in enumerate(split):
            images.append({'image_id':i,'file_name':str((Path(a.voc_root)/'JPEGImages'/(key+'.jpg')).resolve()),
                           'captions':[], 'concepts':names,
                           'mask':str((Path(a.voc_root)/'SegmentationClass'/(key+'.png')).resolve())})
        ignore=[0,255]
    write(out,{'classes':classes,'ignore_values':ignore,'images':images,
               'mask_backend':BACKEND if a.kind=='coco' else 'existing_voc_png',
               'note':'GT only for diagnostics. No B preparation/training reader accepts this manifest.'})
    print('Manifest written:',out,'images:',len(images),flush=True)


if __name__=='__main__':main()
