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
# misma esfera exportada como GLB (con un nodo desplazado), para probar la lectura de .glb
import json, struct, io
def glb(path, V, UV, F, png_bytes):
    uvg = UV.copy(); uvg[:, 1] = 1 - uvg[:, 1]
    pos = V.astype('<f4').tobytes(); tex = uvg.astype('<f4').tobytes(); ind = F.astype('<u4').ravel().tobytes()
    def pad(b): return b + b'\x00' * ((4 - len(b) % 4) % 4)
    partes = [pad(pos), pad(tex), pad(ind), pad(png_bytes)]
    offs = np.cumsum([0] + [len(p) for p in partes[:-1]]).tolist(); binc = b''.join(partes)
    js = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
          'nodes': [{'mesh': 0, 'translation': [5.0, 0.0, 0.0]}],
          'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'TEXCOORD_0': 1}, 'indices': 2, 'material': 0}]}],
          'materials': [{'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}}}],
          'textures': [{'source': 0}], 'images': [{'bufferView': 3, 'mimeType': 'image/png'}],
          'buffers': [{'byteLength': len(binc)}],
          'bufferViews': [{'buffer': 0, 'byteOffset': offs[0], 'byteLength': len(pos)},
                          {'buffer': 0, 'byteOffset': offs[1], 'byteLength': len(tex)},
                          {'buffer': 0, 'byteOffset': offs[2], 'byteLength': len(ind)},
                          {'buffer': 0, 'byteOffset': offs[3], 'byteLength': len(png_bytes)}],
          'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': len(V), 'type': 'VEC3', 'min': V.min(0).tolist(), 'max': V.max(0).tolist()},
                        {'bufferView': 1, 'componentType': 5126, 'count': len(V), 'type': 'VEC2'},
                        {'bufferView': 2, 'componentType': 5125, 'count': F.size, 'type': 'SCALAR'}]}
    jb = json.dumps(js).encode(); jb += b' ' * ((4 - len(jb) % 4) % 4)
    total = 12 + 8 + len(jb) + 8 + len(binc)
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, total)); f.write(struct.pack('<I4s', len(jb), b'JSON')); f.write(jb)
        f.write(struct.pack('<I4s', len(binc), b'BIN\x00')); f.write(binc)
b = io.BytesIO(); Image.open(os.path.join(d, 't.png')).save(b, 'PNG')
glb(os.path.join(d, 'esferaglb.glb'), V, UV, F, b.getvalue())
