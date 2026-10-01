import os, sys, zipfile, json, re
d=os.path.join(os.path.dirname(os.path.abspath(__file__)),'modelo'); ok=True
for suf,nf in (('mmu3',5),('indx8',7)):
    f=os.path.join(d,f'esfera_prusa_{suf}','esfera_colormix.3mf')
    if not os.path.exists(f): print('FALTA',f); ok=False; continue
    z=zipfile.ZipFile(f); j=json.loads(z.read('Metadata/Prusa_Slicer_full_spectrum.json'))
    ids=[v['id'] for v in j['virtual_extruders']]
    good=len(j['physical_extruders'])==nf and len(ids)==105 and ids[0]==nf+1
    print(suf,'OK' if good else 'FALLO',len(j['physical_extruders']),ids[0],ids[-1]); ok&=good
# el GLB debe dar el mismo pintado que el OBJ equivalente
fo=os.path.join(d,'esfera_prusa_mmu3','esfera_colormix.3mf'); fg=os.path.join(d,'esferaglb_prusa_mmu3','esferaglb_colormix.3mf')
if not os.path.exists(fg): print('FALTA',fg); ok=False
else:
    seg=lambda f: re.findall(r'mmu_segmentation="([^"]*)"', zipfile.ZipFile(f).read('3D/3dmodel.model').decode())
    a,b=seg(fo),seg(fg); igual=sum(x==y for x,y in zip(a,b))/max(len(a),1)
    good=len(a)==len(b) and igual>0.999
    print('glb','OK' if good else 'FALLO', f'{igual*100:.2f} % igual al OBJ'); ok&=good
sys.exit(0 if ok else 1)
