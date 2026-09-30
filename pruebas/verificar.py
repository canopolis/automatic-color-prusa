import os, sys, zipfile, json, re
d=os.path.join(os.path.dirname(os.path.abspath(__file__)),'modelo'); ok=True
for suf,nf in (('mmu3',5),('indx8',7)):
    f=os.path.join(d,f'esfera_prusa_{suf}','esfera_colormix.3mf')
    if not os.path.exists(f): print('FALTA',f); ok=False; continue
    z=zipfile.ZipFile(f); j=json.loads(z.read('Metadata/Prusa_Slicer_full_spectrum.json'))
    ids=[v['id'] for v in j['virtual_extruders']]
    good=len(j['physical_extruders'])==nf and len(ids)==105 and ids[0]==nf+1
    print(suf,'OK' if good else 'FALLO',len(j['physical_extruders']),ids[0],ids[-1]); ok&=good
sys.exit(0 if ok else 1)
