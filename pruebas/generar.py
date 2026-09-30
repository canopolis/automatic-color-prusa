import os, numpy as np
from PIL import Image
d=os.path.join(os.path.dirname(os.path.abspath(__file__)),'modelo'); os.makedirs(d,exist_ok=True)
N=150; u,v=np.meshgrid(np.linspace(0,1,N),np.linspace(0,1,N)); th,ph=u*2*np.pi,v*np.pi
V=np.stack([np.sin(ph)*np.cos(th),np.cos(ph),np.sin(ph)*np.sin(th)],-1).reshape(-1,3)
UV=np.stack([u,1-v],-1).reshape(-1,2); i=np.arange(N-1); I,J=np.meshgrid(i,i); a=(J*N+I).ravel()
F=np.concatenate([np.stack([a,a+N,a+1],1),np.stack([a+1,a+N,a+N+1],1)])
Image.fromarray((np.stack([u,v,0.5+0.5*np.sin(8*u)],-1)*255).astype(np.uint8)).resize((512,512)).save(os.path.join(d,'t.png'))
open(os.path.join(d,'t.mtl'),'w').write('newmtl m\nmap_Kd t.png\n')
with open(os.path.join(d,'esfera.obj'),'w') as f:
    f.write('mtllib t.mtl\n'); f.writelines(f'v {x} {y} {z}\n' for x,y,z in V); f.writelines(f'vt {s} {t}\n' for s,t in UV)
    f.write('usemtl m\n'); f.writelines(f'f {p+1}/{p+1} {q+1}/{q+1} {r+1}/{r+1}\n' for p,q,r in F)
# 3MF de prueba con 7 cabezales (simula el guardado desde PrusaSlicer con 'full color print indx8')
import zipfile, re
base=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg=open(os.path.join(base,'plantilla_Slic3r_PE.config'),encoding='utf-8').read()
cfg=re.sub(r'^; nozzle_diameter = .*$','; nozzle_diameter = '+','.join(['0.4']*7),cfg,flags=re.M)
cfg=re.sub(r'^; extruder_colour = .*$','; extruder_colour = #009BC3;#DB5182;#FFD100;#F5F5F5;#1A1A1A;#C9A64B;#B4B4B4',cfg,flags=re.M)
with zipfile.ZipFile(os.path.join(d,'indx8_prueba.3mf'),'w') as z: z.writestr('Metadata/Slic3r_PE.config',cfg)
