import os, sys, glob, collections, xml.etree.ElementTree as ET
R=sys.argv[1]; O=sys.argv[2]
A=f'{R}/Annotations/Oriented Bounding Boxes'
tv=sorted(os.path.splitext(f)[0] for f in os.listdir(f'{R}/JPEGImages-trainval'))
te=sorted(os.path.splitext(f)[0] for f in os.listdir(f'{R}/JPEGImages-test'))
names=set()
parsed={}
for i in tv+te:
    t=ET.parse(f'{A}/{i}.xml').getroot(); W=float(t.find('size/width').text); H=float(t.find('size/height').text); objs=[]
    for o in t.findall('object'):
        n=o.find('name').text.strip(); b=o.find('robndbox'); names.add(n)
        p=[float(b.find(k).text) for k in ('x_left_top','y_left_top','x_right_top','y_right_top','x_right_bottom','y_right_bottom','x_left_bottom','y_left_bottom')]
        objs.append((n,p))
    parsed[i]=(W,H,objs)
names=sorted(names); cid={n:k for k,n in enumerate(names)}
cnt=collections.Counter()
for split,lst,src in (('train',tv,'JPEGImages-trainval'),('val',te,'JPEGImages-test')):
    os.makedirs(f'{O}/images/{split}',exist_ok=True); os.makedirs(f'{O}/labels/{split}',exist_ok=True)
    for i in lst:
        W,H,objs=parsed[i]; os.symlink(os.path.abspath(f'{R}/{src}/{i}.jpg'),f'{O}/images/{split}/{i}.jpg'); out=[]
        for n,p in objs:
            cnt[(split,n)]+=1
            out.append(f"{cid[n]} "+" ".join(f"{min(max(v/(W if k%2==0 else H),0),1):.6f}" for k,v in enumerate(p)))
        open(f'{O}/labels/{split}/{i}.txt','w').write("\n".join(out)+("\n" if out else ""))
open(f'{O}/data.yaml','w').write(f"path: {os.path.abspath(O)}\ntrain: images/train\nval: images/val\nnames:\n"+"".join(f"  {k}: {n}\n" for k,n in enumerate(names)))
print(len(tv),len(te),len(names),sum(v for (s,_),v in cnt.items() if s=='train'),sum(v for (s,_),v in cnt.items() if s=='val'))
