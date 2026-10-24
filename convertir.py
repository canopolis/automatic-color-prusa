# -*- coding: utf-8 -*-
"""
FullSpectrumPrusa - Convierte un OBJ full spectrum (OBJ+MTL+textura) en un 3MF de PrusaSlicer 2.9.6
con ColorMix: 5 filamentos reales (C M Y W K) + hasta 105 virtuales, pintado listo para Core One MMU3.
Uso: arrastrar el .obj sobre el .exe   |   python convertir.py modelo.obj [altura_mm]
"""
import sys, os, json, zipfile, itertools, time, io
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from PIL import Image

ALTURA_MM = 150.0      # altura de impresion si el modelo viene sin escala real o no cabe en la Core One
CELDA_MM = 0.4         # escala de suavizado del color (ancho de boquilla)
SUAVIZADO_MM = 0.8     # radio de suavizado del color sobre la superficie
UNIFORME_MM2 = 2000.0  # zonas de hasta este tamano se funden si el vecino tiene casi el mismo tono
UNIFORME_DE = 10.0     # 'casi el mismo tono' (dE); detalles con mas diferencia se conservan
PROTEGER_FRAC = 0.3    # fraccion minima de detalle para que una zona se considere detalle
DETALLE_DE = 12.0      # contraste local del original que marca 'detalle' (se protege)
SUAVIZADO_PIEL_MM = 2.5 # suavizado fuera de la cara/rasgos: piel y telas lisas quedan uniformes
RASGO_DE = 25.0        # contraste fuerte que define un rasgo (ojo, ceja, labio)
RASGO_DE_2 = 20.0      # segundo intento si no se encontro cara
RASGO_MIN_MM2_2 = 0.3  # segundo intento si no se encontro cara
RASGO_MIN_MM2 = 1.0    # rasgos menores que esto son ruido de textura, no se protegen
RASGO_MAX_MM2 = 150.0  # detalles de hasta este tamano (ojos, labios, cejas) se protegen
CARA_RADIO_MM = 12.0   # rasgos a menos de esta distancia entre si forman una cara
CARA_MIN_RASGOS = 2    # minimo de rasgos vecinos para considerarlo cara
RASGO_MARGEN_MM = 9.0  # margen protegido alrededor de esos rasgos (cubre la cara)
TONO_DE = 6.0          # build-17: diferencia de tono (sin cambio de luz) que marca un detalle de color (areolas, labios)
RASGO_ENTORNO_MM = 6.0 # build-17: radio del entorno con el que se compara cada punto para encontrar rasgos
ZONA_CLASES = 24       # numero de familias de color para agrupar superficies lisas
ZONA_TOL = 12.0        # una cara solo toma el tono de su zona si se parece a su propio color
ZONA_MIN_MM2 = 30.0    # superficies lisas desde este tamano se pintan de un solo tono
ISLA_MIN_MM2 = 2.0     # zonas de color mas pequenas que esto se funden con su vecina
N_VIRTUALES = 105
PASO_RATIO = 0.05
BED = (125.0, 110.0)   # centro de la cama Core One
# colores fisicos (mismos que el proyecto PrusaSlicer de referencia)
FISICOS = ['#009BC3', '#DB5182', '#FFD100', '#F5F5F5', '#1A1A1A']   # C M Y W K  (Prusament PLA ColorMix)
A_MEZCLA = 0.0045      # parametro del modelo de mezcla ajustado a los virtuales de PrusaSlicer

# ---- modos de impresora ----
# fisicos: colores de cada cabezal/ranura | usables: los que el convertidor puede pintar
# mezcla: indices (0-based) de los filamentos que forman los virtuales (siempre C M Y W K)
MODO = {'nombre': 'Prusa MMU3', 'plantilla': 'plantilla_Slic3r_PE.config',
        'fisicos': list(FISICOS), 'usables': [0, 1, 2, 3, 4], 'mezcla': [0, 1, 2, 3, 4], 'sufijo': 'mmu3'}

def dir_exe():
    return os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))

def modo_indx8():
    """INDX 8 con 7 cabezales: C M Y W K + Gold Glitter + Silver Glitter.
    La plantilla de impresora se toma, una sola vez, de un 3MF guardado en PrusaSlicer con 'full color print indx8'."""
    pl = os.path.join(dir_exe(), 'plantilla_indx8.config')
    if not os.path.exists(pl):
        print('\nPrimera vez en modo INDX 8: falta la configuracion de la impresora.')
        print('En PrusaSlicer elige la impresora "full color print indx8", pon un cubo y guarda el proyecto (.3mf).')
        src = os.environ.get('INDX8_3MF') or input('Arrastra aqui ese .3mf y pulsa Enter: ').strip().strip('"')
        with zipfile.ZipFile(src) as z:
            cfg = z.read('Metadata/Slic3r_PE.config').decode('utf-8')
        if len([x for x in (linea_cfg(cfg, 'nozzle_diameter') or '').split(',') if x]) != 7:
            raise SystemExit('ERROR: ese 3MF no tiene 7 cabezales. Guardalo con la impresora "full color print indx8".')
        open(pl, 'w', encoding='utf-8').write(cfg); print('Configuracion INDX 8 guardada; no se volvera a pedir.')
    cfg = open(pl, encoding='utf-8').read()
    cols = (linea_cfg(cfg, 'extruder_colour') or '').split(';')
    fis = [c if c.startswith('#') and len(c) == 7 else '#808080' for c in cols][:7]
    fis[:5] = FISICOS                     # C M Y W K con los mismos valores del modelo de mezcla ajustado
    MODO.update({'nombre': 'Prusa INDX 8', 'plantilla': pl, 'fisicos': fis,
                 'usables': [0, 1, 2, 3, 4], 'mezcla': [0, 1, 2, 3, 4], 'sufijo': 'indx8'})

def linea_cfg(cfg, clave):
    for l in cfg.split('\n'):
        if l.startswith(f'; {clave} = '): return l.split(' = ', 1)[1].strip()
    return None

def base_dir():
    return getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

# ---------------- color ----------------
def h2r(h): return np.array([int(h[i:i+2], 16) for i in (1, 3, 5)]) / 255
def r2h(c): c = np.clip(np.round(c * 255), 0, 255).astype(int); return '#%02X%02X%02X' % tuple(c)
def s2l(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def l2s(c): c = np.clip(c, 0, 1); return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
def lab(srgb):
    lin = s2l(np.clip(srgb, 0, 1))
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)
def base_mezcla(): return np.array([h2r(MODO['fisicos'][i]) for i in MODO['mezcla']])
def base_usable(): return np.array([h2r(MODO['fisicos'][i]) for i in MODO['usables']])
# ---- build-17: motor de color ratdoux (Snapmaker Orca FullSpectrum v0.9.18) ----
_RD = []
def RATDOUX():
    """carga ratdoux.dll (modelo KM/K-S de la v0.9.18); None si no esta (se usa el modelo anterior)"""
    if not _RD:
        lib = None
        import ctypes
        rutas = [os.environ.get('RATDOUX_LIB', '')] + [os.path.join(d, 'ratdoux.dll') for d in (base_dir(), dir_exe())]
        for p in rutas:
            if p and os.path.exists(p):
                try:
                    lib = ctypes.CDLL(p)
                    lib.cm_predict.argtypes = [ctypes.c_char_p, ctypes.c_char_p]; lib.cm_predict.restype = ctypes.c_int
                    break
                except OSError:
                    lib = None
        _RD.append(lib)
    return _RD[0]

def mezcla(R):
    """color predicho de una mezcla por capas: modelo KM/K-S de ratdoux v0.9.18 (o el anterior si falta la DLL)"""
    R = np.atleast_2d(R)
    if RATDOUX() is None:
        L = s2l(base_mezcla()) + A_MEZCLA
        return l2s(np.exp(R @ np.log(L)) - A_MEZCLA)
    hexs = ','.join(MODO['fisicos'][i] for i in MODO['mezcla']).encode()
    out = np.zeros((len(R), 3))
    for k, w in enumerate(R):
        p = RATDOUX().cm_predict(hexs, ','.join(str(int(round(x * 100))) for x in w).encode())
        if p >= 0: out[k] = [(p >> 16) & 255, (p >> 8) & 255, p & 255]
    return out / 255

def _oklab_de_lab(A):
    """Lab (D65) -> Oklab, pasando por sRGB lineal recortado a 0..1 como en ratdoux"""
    fy = (A[:, 0] + 16) / 116; fx = fy + A[:, 1] / 500; fz = fy - A[:, 2] / 200
    f = np.stack([fx, fy, fz], 1)
    xyz = np.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1, 1.08883])
    Mi = np.linalg.inv(np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]))
    lin = np.clip(xyz @ Mi.T, 0, 1)
    l = np.cbrt(lin @ np.array([0.4122214708, 0.5363325363, 0.0514459929]))
    m = np.cbrt(lin @ np.array([0.2119034982, 0.6806995451, 0.1073969566]))
    s = np.cbrt(lin @ np.array([0.0883024619, 0.2817188376, 0.6299787005]))
    return np.stack([0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s], 1)

def dist(A, B):
    """distancia de ratdoux v0.9.18 (Oklab, pesos segun croma del objetivo y castigo a lo mas oscuro), en escala dE.
    A (n,3) objetivo contra B (m,3), ambos en Lab. Sin la DLL se usa la distancia anterior."""
    if RATDOUX() is None: return dist_anterior(A, B)
    t = _oklab_de_lab(np.asarray(A, float)); c = _oklab_de_lab(np.asarray(B, float))
    cf = np.clip((np.hypot(t[:, 1], t[:, 2]) - 0.015) / 0.13, 0, 1)
    wl = np.maximum(1.0, 1 + (0.25 - 1) * cf); wab = np.minimum(4.0, 1.25 + (8 - 1.25) * cf)
    dl = c[None, :, 0] - t[:, None, 0]; da = c[None, :, 1] - t[:, None, 1]; db = c[None, :, 2] - t[:, None, 2]
    e = wl[:, None] * dl * dl + wab[:, None] * (da * da + db * db)
    under = np.maximum(0, t[:, None, 0] - c[None, :, 0] - 0.04)
    e += 4.0 * cf[:, None] * under * under
    return (100 * np.sqrt(e)).astype(np.float32)

def candidatos():
    n = int(round(1 / PASO_RATIO)); out = []
    for k in (2, 3):
        for comb in itertools.combinations(range(5), k):
            for parts in itertools.product(range(1, n), repeat=k):
                if sum(parts) != n: continue
                w = np.zeros(5); w[list(comb)] = np.array(parts) / n; out.append(w)
    for comb in itertools.combinations([0, 1, 2, 4], 2):          # blanco + 1 o 2 colores, paso fino
        for a in range(0, 16):
            for b in range(0, 16):
                if a + b == 0 or a + b > 15: continue
                w = np.zeros(5); w[3] = 1 - (a + b) * 0.02; w[comb[0]] = a * 0.02; w[comb[1]] = b * 0.02; out.append(w)
    W = np.unique(np.round(np.array(out), 4), axis=0); return W, mezcla(W)

def dist_anterior(A, B):
    """distancia tipo CIE94 con mas peso al tono: A (n,3) contra B (m,3) en Lab"""
    dL = A[:, None, 0] - B[None, :, 0]
    Ca = np.hypot(A[:, 1], A[:, 2]); Cb = np.hypot(B[:, 1], B[:, 2])
    dC = Ca[:, None] - Cb[None, :]
    da = A[:, None, 1] - B[None, :, 1]; db = A[:, None, 2] - B[None, :, 2]
    dH2 = np.maximum(da * da + db * db - dC * dC, 0)
    wH = 1 + 1.5 * np.clip(Ca / 15, 0, 1)[:, None]          # en colores con tono definido, el tono pesa mas
    return np.sqrt(dL * dL + dC * dC + wH * wH * dH2).astype(np.float32)

def penal_piel(Lt, Wp):
    """en tonos claros y calidos (piel, pelo rubio) castiga mezclas con cian o negro, que ensucian"""
    calido = (Lt[:, 0] > 55) & (Lt[:, 1] > -3) & (Lt[:, 2] > 3)
    return (calido[:, None] * (40 * Wp[None, :, 0] + 25 * Wp[None, :, 4])).astype(np.float32)

def elegir_paleta(W, rgb, muestras, k):
    """facility location voraz: elige k virtuales que minimizan el error dE sobre los colores del modelo"""
    L = lab(rgb).astype(np.float32); B = lab(base_usable()).astype(np.float32); m = lab(muestras).astype(np.float32)
    cur = dist(m, B).min(1)
    D = np.empty((len(m), len(L)), np.float32)
    for s0 in range(0, len(L), 500):
        D[:, s0:s0 + 500] = dist(m, L[s0:s0 + 500])
    D += penal_piel(m, W)
    Bw = np.eye(5)[MODO['usables']]
    cur = cur + penal_piel(m, Bw).min(1) * 0
    q = np.round(m / 6).astype(int); _, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    peso = 1 / np.sqrt(cnt[inv.ravel()])          # los colores escasos (cara, detalles) pesan mas
    sel = []
    for _ in range(k):
        g = (peso[:, None] * np.maximum(cur[:, None] - D, 0)).sum(0); g[sel] = -1
        j = int(g.argmax()); sel.append(j); cur = np.minimum(cur, D[:, j])
    return W[sel], rgb[sel]

# ---------------- GLB / GLTF ----------------
def _matriz_nodo(n):
    if 'matrix' in n: return np.array(n['matrix'], float).reshape(4, 4).T
    M = np.eye(4)
    t = n.get('translation', [0, 0, 0]); q = n.get('rotation', [0, 0, 0, 1]); sc = n.get('scale', [1, 1, 1])
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M[:3, :3] = R * np.array(sc)[None, :]; M[:3, 3] = t
    return M

def cargar_glb(path):
    """lee modelos .glb / .gltf (los que entregan los generadores 3D con IA): geometria, UV, textura o colores"""
    import json, struct, base64, tempfile
    datos = open(path, 'rb').read(); base = os.path.dirname(path)
    binarios = {}
    if datos[:4] == b'glTF':
        off = 12; js = None
        while off + 8 <= len(datos):
            ln, typ = struct.unpack_from('<I4s', datos, off); off += 8
            ch = datos[off:off + ln]; off += ln
            if typ == b'JSON': js = json.loads(ch.decode('utf-8'))
            elif typ == b'BIN\x00': binarios[0] = ch
    else:
        js = json.loads(datos.decode('utf-8'))
    if js is None: raise SystemExit('ERROR: el archivo no es un GLB/GLTF valido.')
    if 'KHR_draco_mesh_compression' in js.get('extensionsUsed', []) or 'EXT_meshopt_compression' in js.get('extensionsUsed', []):
        raise SystemExit('ERROR: este GLB tiene la geometria comprimida (Draco/meshopt). Exportalo sin compresion o como OBJ.')
    def buffer(i):
        if i not in binarios:
            uri = js['buffers'][i].get('uri', '')
            binarios[i] = base64.b64decode(uri.split(',', 1)[1]) if uri.startswith('data:') else open(os.path.join(base, uri), 'rb').read()
        return binarios[i]
    tipos = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    comps = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
    def acc(i):
        a = js['accessors'][i]; bv = js['bufferViews'][a['bufferView']]
        dt = np.dtype(tipos[a['componentType']]); n = comps[a['type']]
        buf = buffer(bv.get('buffer', 0)); ini = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        paso = bv.get('byteStride', 0) or dt.itemsize * n
        arr = np.ndarray((a['count'], n), dtype=dt, buffer=buf, offset=ini, strides=(paso, dt.itemsize)).astype(float)
        if a.get('normalized') and dt.kind in 'iu':
            arr = arr / float(np.iinfo(dt).max)
        return arr
    def imagen(ii):
        im = js['images'][ii]
        if 'bufferView' in im:
            bv = js['bufferViews'][im['bufferView']]; b = buffer(bv.get('buffer', 0))
            return b[bv.get('byteOffset', 0): bv.get('byteOffset', 0) + bv['byteLength']]
        uri = im.get('uri', '')
        return base64.b64decode(uri.split(',', 1)[1]) if uri.startswith('data:') else open(os.path.join(base, uri), 'rb').read()
    # matrices de cada malla segun la jerarquia de nodos
    mallas = []
    escena = js.get('scenes', [{}])[js.get('scene', 0)] if js.get('scenes') else {'nodes': list(range(len(js.get('nodes', []))))}
    def visitar(ni, M):
        n = js['nodes'][ni]; M2 = M @ _matriz_nodo(n)
        if 'mesh' in n: mallas.append((n['mesh'], M2))
        for h in n.get('children', []): visitar(h, M2)
    for ni in escena.get('nodes', []): visitar(ni, np.eye(4))
    if not mallas: mallas = [(i, np.eye(4)) for i in range(len(js.get('meshes', [])))]
    Vs, Ts, Cs, Fs, imgs = [], [], [], [], []; base_v = 0; hay_uv = False
    for mi, M in mallas:
        for pr in js['meshes'][mi]['primitives']:
            if pr.get('mode', 4) != 4: continue
            at = pr['attributes']; P = acc(at['POSITION'])
            P = (np.c_[P, np.ones(len(P))] @ M.T)[:, :3]
            idx = acc(pr['indices']).astype(np.int64).ravel() if 'indices' in pr else np.arange(len(P))
            F = idx.reshape(-1, 3) + base_v
            mat = js.get('materials', [{}])[pr['material']] if 'material' in pr else {}
            pbr = mat.get('pbrMetallicRoughness', {}); fac = np.array(pbr.get('baseColorFactor', [1, 1, 1, 1]))[:3]
            tx = pbr.get('baseColorTexture')
            if tx is not None and 'TEXCOORD_0' in at:
                src = js['textures'][tx['index']]['source']
                if src not in imgs: imgs.append(src)
                uv = acc(at['TEXCOORD_0'])[:, :2].copy(); uv[:, 1] = 1 - uv[:, 1]; hay_uv = True
            else:
                uv = np.zeros((len(P), 2))
            if 'COLOR_0' in at: col_v = acc(at['COLOR_0'])[:, :3] * fac
            else: col_v = np.tile(fac, (len(P), 1))
            Vs.append(P); Ts.append(uv); Cs.append(col_v); Fs.append(F); base_v += len(P)
    if not Vs: raise SystemExit('ERROR: el GLB no contiene mallas de triangulos.')
    if len(imgs) > 1:
        raise SystemExit('ERROR: este GLB usa varias texturas distintas. Exportalo como OBJ con una sola textura.')
    V = np.vstack(Vs); FV = np.vstack(Fs)
    tex = None; T = None; VC = None
    if hay_uv:
        T = np.vstack(Ts); fuera = (T < 0) | (T > 1); T = np.where(fuera, T - np.floor(T), T)
        d = tempfile.mkdtemp(prefix='fsp_'); tex = os.path.join(d, 'textura_glb.png')
        Image.open(io.BytesIO(imagen(imgs[0]))).convert('RGB').save(tex)
    else:
        VC = np.vstack(Cs)
    return V, VC, T, FV, FV.copy(), tex

# ---------------- OBJ ----------------
def cargar_obj(path):
    with open(path, 'r', errors='ignore') as fh: lines = fh.read().split('\n')
    v = [l[2:] for l in lines if l.startswith('v ')]
    vt = [l[3:] for l in lines if l.startswith('vt ')]
    f = [l[2:].split() for l in lines if l.startswith('f ')]
    mtllib = next((l.split(None, 1)[1].strip() for l in lines if l.startswith('mtllib')), None)
    vv = np.array([x.split() for x in v], dtype=object)
    ncol = max(len(x) for x in vv)
    Varr = np.array([list(map(float, x[:6])) + [0] * (6 - len(x[:6])) for x in vv])
    V = Varr[:, :3]; VC = Varr[:, 3:6] if ncol >= 6 else None
    T = np.array(' '.join(vt).split(), float).reshape(len(vt), -1)[:, :2] if vt else None
    FV = []; FT = []
    for p in f:
        a = [x.split('/') for x in p]
        vi = [int(x[0]) for x in a]; ti = [int(x[1]) if len(x) > 1 and x[1] else 0 for x in a]
        for k in range(1, len(a) - 1):
            FV.append([vi[0], vi[k], vi[k + 1]]); FT.append([ti[0], ti[k], ti[k + 1]])
    FV = np.array(FV); FT = np.array(FT)
    FV = np.where(FV > 0, FV - 1, len(V) + FV); FT = np.where(FT > 0, FT - 1, (len(T) if T is not None else 0) + FT)
    tex = None
    if mtllib and T is not None:
        mp = os.path.join(os.path.dirname(path), mtllib)
        if os.path.exists(mp):
            for l in open(mp, errors='ignore'):
                p = l.strip().split(None, 1)
                if p and p[0].lower() == 'map_kd':
                    img = os.path.join(os.path.dirname(path), p[1].strip().replace('\\', '/').split('/')[-1])
                    if os.path.exists(img): tex = img
    if T is not None:
        # algunos exportadores (p. ej. 3dviewer.net) guardan la coordenada V negativa (-1..0);
        # se aplica la repeticion estandar de textura para que caiga dentro de 0..1 (los modelos normales no cambian)
        fuera = (T < 0) | (T > 1)
        T = np.where(fuera, T - np.floor(T), T)
    return V, VC, T, FV, FT, tex

# ---------------- tramado ----------------
BAYER = np.array([[0,32,8,40,2,34,10,42],[48,16,56,24,50,18,58,26],[12,44,4,36,14,46,6,38],[60,28,52,20,62,30,54,22],
                  [3,35,11,43,1,33,9,41],[51,19,59,27,49,17,57,25],[15,47,7,39,13,45,5,37],[63,31,55,23,61,29,53,21]]) / 64

def asignar(colores, PL, umbral):
    """para cada color: mejor par de la paleta que lo encierra + tramado ordenado"""
    out = np.zeros(len(colores), int); Lc = lab(colores)
    for s0 in range(0, len(Lc), 40000):
        p = Lc[s0:s0 + 40000]; D = np.linalg.norm(p[:, None] - PL[None], axis=2); o = np.argsort(D, 1)[:, :8]; A = PL[o]
        bv = np.full(len(p), 1e9); bi = o[:, 0].copy(); bj = o[:, 0].copy(); bt = np.zeros(len(p))
        for i in range(8):
            for j in range(8):
                if i == j: continue
                a = A[:, i]; c = A[:, j]; v = c - a
                t = np.clip(((p - a) * v).sum(1) / ((v * v).sum(1) + 1e-9), 0, 1)
                err = np.linalg.norm(a + t[:, None] * v - p, axis=1) + 0.3 * np.linalg.norm(v, axis=1) * t * (1 - t)
                m = err < bv; bv[m] = err[m]; bi[m] = o[m, i]; bj[m] = o[m, j]; bt[m] = t[m]
        out[s0:s0 + 40000] = np.where(umbral[s0:s0 + 40000] < bt, bj, bi)
    return out


# ---------------- limpieza sobre la malla (sin tramado: evita el efecto "viruela") ----------------
def vecinos(F):
    """3 vecinos por cara (por arista compartida); -1 si es borde"""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e.sort(1)
    fid = np.tile(np.arange(len(F)), 3); lado = np.repeat([0, 1, 2], len(F))
    k = e[:, 0].astype(np.int64) * (F.max() + 1) + e[:, 1]
    o = np.argsort(k, kind='stable'); ks = k[o]
    N = -np.ones((len(F), 3), np.int64)
    par = np.nonzero(ks[1:] == ks[:-1])[0]
    a, b = o[par], o[par + 1]
    N[fid[a], lado[a]] = fid[b]; N[fid[b], lado[b]] = fid[a]
    return N

def suavizar(Lc, N, pasos, sigma=10.0):
    """suavizado que respeta bordes: no mezcla colores muy distintos (bikini/piel, ojos, labios)"""
    X = Lc.copy(); ok = N >= 0; Nn = np.where(ok, N, 0)
    for _ in range(pasos):
        d2 = ((X[Nn] - X[:, None]) ** 2).sum(2)
        w = ok * np.exp(-d2 / (2 * sigma * sigma))
        X = (X + (X[Nn] * w[..., None]).sum(1)) / (1 + w.sum(1))[:, None]
    return X

def mayoria(idx, N, pasadas=4):
    ok = N >= 0; Nn = np.where(ok, N, 0)
    for _ in range(pasadas):
        L = np.where(ok, idx[Nn], -1)
        a, b, c = L[:, 0], L[:, 1], L[:, 2]
        cand = np.where((a == b) & (a >= 0), a, np.where((b == c) & (b >= 0), b, np.where((a == c) & (a >= 0), a, idx)))
        idx = cand
    return idx

def quitar_islas(idx, N, area, min_area, rondas=40, Lref=None, PL=None, tol=10.0, proteger=None):
    """funde cada zona de color menor que min_area con el color vecino con el que comparte mas borde"""
    n = len(idx); r = np.repeat(np.arange(n), 3); c = N.ravel(); val = c >= 0; r, c = r[val], c[val]
    for _ in range(rondas):
        mism = idx[r] == idx[c]
        g = coo_matrix((np.ones(mism.sum()), (r[mism], c[mism])), shape=(n, n))
        nc, comp = connected_components(g, directed=False)
        a = np.bincount(comp, weights=area, minlength=nc)
        # procesar primero las mas pequenas: solo zonas bajo el umbral
        peq = a < min_area
        if not peq.any(): break
        borde = (~mism) & peq[comp[r]]
        cr, lc = comp[r[borde]], idx[c[borde]]
        if len(cr) == 0: break
        key = cr.astype(np.int64) * 1000 + lc
        uk, cnt = np.unique(key, return_counts=True)
        kc, kl = uk // 1000, uk % 1000
        o = np.lexsort((-cnt, kc)); kc, kl = kc[o], kl[o]
        first = np.r_[True, kc[1:] != kc[:-1]]
        nuevo = np.full(nc, -1); nuevo[kc[first]] = kl[first]
        cambia = peq & (nuevo >= 0)
        if Lref is not None:
            cnt_c = np.bincount(comp, minlength=nc)
            media = np.stack([np.bincount(comp, weights=Lref[:, j], minlength=nc) for j in range(3)], 1) / np.maximum(cnt_c, 1)[:, None]
            dist = np.linalg.norm(media - PL[np.maximum(nuevo, 0)], axis=1)
            cambia &= dist < tol                   # detalle real (ojo, labio, borde) -> se conserva
        if proteger is not None:
            frac = np.bincount(comp, weights=area * proteger, minlength=nc) / np.maximum(a, 1e-9)
            cambia &= frac < PROTEGER_FRAC      # zona que ES detalle (ojo, labio) se conserva; un parche que solo roza un borde, no
        f = cambia[comp]; idx = idx.copy(); idx[f] = nuevo[comp[f]]
    return idx

# ---------------- 3MF ----------------
def codigo(estado):
    if estado == 0: return ''
    if estado < 3: return '%X' % (estado << 2)
    if estado < 17: return '%XC' % (estado - 3)
    return '%02XEC' % (estado - 17)

def escribir_3mf(out, nombre, V, F, estados, WV, PV, miniatura):
    pl = MODO['plantilla'] if os.path.isabs(MODO['plantilla']) else os.path.join(base_dir(), MODO['plantilla'])
    cfg = open(pl, encoding='utf-8').read()
    cfg = '\n'.join(l for l in cfg.split('\n') if not l.startswith('; print_settings_id'))
    buf = io.StringIO()
    buf.write('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
              'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
              'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">\n'
              ' <metadata name="slic3rpe:Version3mf">1</metadata>\n'
              ' <metadata name="slic3rpe:MmPaintingVersion">2</metadata>\n'
              f' <metadata name="Title">{nombre}</metadata>\n'
              ' <metadata name="Application">PrusaSlicer-2.9.6</metadata>\n'
              ' <resources>\n  <object id="1" type="model">\n   <mesh>\n    <vertices>\n')
    buf.write(''.join(f'     <vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>\n' for x, y, z in V))
    buf.write('    </vertices>\n    <triangles>\n')
    cod = np.array([codigo(i) for i in range(int(estados.max()) + 1)], dtype=object)
    buf.write(''.join(f'     <triangle v1="{a}" v2="{b}" v3="{c}" slic3rpe:mmu_segmentation="{s}"/>\n'
                      for (a, b, c), s in zip(F, cod[estados])))
    buf.write('    </triangles>\n   </mesh>\n  </object>\n </resources>\n'
              ' <build>\n  <item objectid="1" transform="1 0 0 0 1 0 0 0 1 0 0 0" printable="1"/>\n </build>\n</model>\n')
    modelcfg = ('<?xml version="1.0" encoding="UTF-8"?>\n<config>\n <object id="1" instances_count="1">\n'
                f'  <metadata type="object" key="name" value="{nombre}"/>\n  <metadata type="object" key="extruder" value="1"/>\n'
                f'  <volume firstid="0" lastid="{len(F) - 1}">\n   <metadata type="volume" key="name" value="{nombre}"/>\n'
                '   <metadata type="volume" key="volume_type" value="ModelPart"/>\n'
                '   <metadata type="volume" key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
                '   <mesh edges_fixed="0" degenerate_facets="0" facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n'
                '  </volume>\n </object>\n</config>\n')
    nf = len(MODO['fisicos'])
    fs = {'physical_extruders': [{'color': h, 'id': i + 1} for i, h in enumerate(MODO['fisicos'])], 'version': 1,
          'virtual_extruders': [{'color': r2h(c), 'components': [{'extruder': MODO['mezcla'][int(e)] + 1, 'ratio': round(float(w[e]), 2)}
                                 for e in np.nonzero(w)[0]], 'id': nf + 1 + k, 'kind': 'fullspectrum'} for k, (w, c) in enumerate(zip(WV, PV))]}
    ct = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
          '<Default Extension="png" ContentType="image/png"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
            '<Relationship Target="/Metadata/thumbnail.png" Id="rel-2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"/>'
            '</Relationships>')
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', ct); z.writestr('_rels/.rels', rels)
        z.writestr('3D/3dmodel.model', buf.getvalue())
        z.writestr('Metadata/Slic3r_PE.config', cfg); z.writestr('Metadata/Slic3r_PE_model.config', modelcfg)
        z.writestr('Metadata/Prusa_Slicer_full_spectrum.json', json.dumps(fs, indent=1))
        z.writestr('Metadata/Prusa_Slicer_wipe_tower_information.xml',
                   '<?xml version="1.0" encoding="utf-8"?>\n<wipe_tower_information position_x="170" position_y="140" rotation_deg="0"/>\n')
        b = io.BytesIO(); miniatura.save(b, 'PNG'); z.writestr('Metadata/thumbnail.png', b.getvalue())

# ---------------- render de vista previa ----------------
def render(V, F, fcol, size=600):
    """vista previa sin huecos: cada triangulo se muestrea segun su tamano en pantalla"""
    P = V[:, [0, 2, 1]] * np.array([1, 1, -1])           # frente: X derecha, Z arriba, -Y hacia camara
    N = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]); N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    lo = P[:, :2].min(0); hi = P[:, :2].max(0); sc = (size - 20) / (hi - lo).max()
    fi = np.where(N[:, 2] > 0)[0]; c = fcol[fi] * np.clip(0.35 + 0.65 * N[fi, 2], 0, 1)[:, None]
    A, B, C = P[F[fi, 0]], P[F[fi, 1]], P[F[fi, 2]]
    lado = np.max([np.linalg.norm((B - A)[:, :2], axis=1), np.linalg.norm((C - B)[:, :2], axis=1),
                   np.linalg.norm((A - C)[:, :2], axis=1)], axis=0) * sc
    k_f = np.clip(np.ceil(lado * 1.5).astype(int), 1, 40)
    img = np.ones((size, size, 3)); zb = np.full(size * size, -1e9); flat = img.reshape(-1, 3)
    for k in np.unique(k_f):
        g = np.nonzero(k_f == k)[0]
        pts = [(i / k, j / k) for i in range(k + 1) for j in range(k + 1 - i)] if k > 1 else [(1/3, 1/3)]
        for u_, v_ in pts:
            p = A[g] + u_ * (B[g] - A[g]) + v_ * (C[g] - A[g])
            x = np.clip(((p[:, 0] - lo[0]) * sc + 10).astype(int), 0, size - 1)
            y = np.clip((size - 1 - ((p[:, 1] - lo[1]) * sc + 10)).astype(int), 0, size - 1)
            pix = y * size + x; z = p[:, 2]; o = np.lexsort((-z, pix)); pu, first = np.unique(pix[o], return_index=True); sel = o[first]
            upd = z[sel] > zb[pu]; zb[pu[upd]] = z[sel][upd]; flat[pu[upd]] = c[g][sel][upd]
    return Image.fromarray((img * 255).astype(np.uint8))


# ---------------- mensajes de progreso ----------------
import threading, builtins
def print(*a, **k):
    """cada mensaje borra antes la linea del contador para que no se mezclen"""
    if k.get('end', '\n') == '\n' and not (a and str(a[0]).startswith('\r')):
        builtins.print('\r' + ' ' * 78 + '\r', end='')
    k.setdefault('flush', True); builtins.print(*a, **k)
class Progreso:
    """muestra el paso actual, una barra y un contador de segundos para que se vea que el programa sigue trabajando"""
    TOTAL = 12
    todos = []
    def __init__(self):
        self.t0 = time.time(); self.n = 0; self.txt = ''; self.ts = time.time()
        Progreso.todos.append(self); self.activo = True; threading.Thread(target=self._latido, daemon=True).start()
    def paso(self, texto):
        self.n += 1; self.txt = texto; self.ts = time.time()
        pct = int(100 * (self.n - 1) / self.TOTAL); barra = '#' * (pct // 5) + '-' * (20 - pct // 5)
        print(f'\r{" " * 70}\r[{barra}] {pct:3d}%  Paso {self.n}/{self.TOTAL}: {texto}', flush=True)
    def _latido(self):
        while self.activo:
            time.sleep(2)
            if self.activo and self.n:
                print(f'\r      trabajando... {int(time.time() - self.ts)} s en este paso | {int(time.time() - self.t0)} s en total', end='', flush=True)
    def fin(self):
        self.activo = False
        print(f'\r{" " * 70}\r[####################] 100%  Terminado en {int(time.time() - self.t0)} s', flush=True)

ULTIMO = {}
# ---------------- principal ----------------
def convertir(path, altura):
    t0 = time.time(); nombre = os.path.splitext(os.path.basename(path))[0]
    PR = Progreso()
    PR.paso('Leyendo el modelo OBJ (los archivos grandes tardan un poco)')
    print(f'Cargando {path} ...')
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.glb', '.gltf'):
        V, VC, T, FV, FT, tex = cargar_glb(path)
    elif ext == '.obj':
        V, VC, T, FV, FT, tex = cargar_obj(path)
    else:
        raise SystemExit(f'ERROR: formato no soportado ({ext}). Usa un archivo .obj, .glb o .gltf.')
    print(f'  {len(FV):,} triangulos | textura: {os.path.basename(tex) if tex else "no"} | color por vertice: {"si" if VC is not None else "no"}')
    # OBJ (Y arriba) -> 3MF (Z arriba), escala y centrado en cama
    V = V @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    alto = V[:, 2].max() - V[:, 2].min()
    # sin escala real o mas alto que la Core One -> se escala a la altura de impresion
    if alto < 20 or alto > 270 or altura: V = V * ((altura or ALTURA_MM) / alto)
    V = V - [(V[:, 0].max() + V[:, 0].min()) / 2 - BED[0], (V[:, 1].max() + V[:, 1].min()) / 2 - BED[1], V[:, 2].min()]
    # color por cara, suavizado a escala de boquilla
    if tex is not None:
        PR.paso('Leyendo la textura y el color de cada cara')
        im = Image.open(tex).convert('RGB'); n = im.size[0]
        i = np.random.default_rng(0).choice(len(FV), min(50000, len(FV)), replace=False)
        a3 = 0.5 * np.linalg.norm(np.cross(V[FV[i, 1]] - V[FV[i, 0]], V[FV[i, 2]] - V[FV[i, 0]]), axis=1)
        t = T[FT[i]]; a2 = 0.5 * np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])) * n * n
        mm_tex = np.sqrt(a3.sum() / max(a2.sum(), 1e-9)); cel = max(1, int(round(CELDA_MM / mm_tex))); g = max(8, n // cel)
        small = np.asarray(im.resize((g, g), Image.BOX), float) / 255
        uv = T[FT].mean(1); gx = np.clip((uv[:, 0] * g).astype(int), 0, g - 1); gy = np.clip(((1 - uv[:, 1]) * g).astype(int), 0, g - 1)
        col = small[gy, gx]; umbral = BAYER[gy % 8, gx % 8]
        full = np.asarray(im, float) / 255
        col_orig = full[np.clip(((1 - uv[:, 1]) * n).astype(int), 0, n - 1), np.clip((uv[:, 0] * n).astype(int), 0, n - 1)]
    elif VC is not None:
        col = VC[FV].mean(1); col_orig = col
        cen = V[FV].mean(1); q = np.floor(cen / CELDA_MM).astype(int); umbral = BAYER[(q[:, 0] + q[:, 2]) % 8, (q[:, 1] + q[:, 2]) % 8]
    else:
        print('ERROR: el OBJ no tiene textura ni color por vertice.'); return
    # vecindad por posicion: el OBJ duplica vertices en las costuras de la textura (UV); sin soldarlos,
    # cada isla de textura quedaba como una zona aparte y recibia su propio tono -> parches
    PR.paso('Uniendo la superficie del modelo')
    _, soldado = np.unique(np.round(V, 4), axis=0, return_inverse=True)
    N = vecinos(soldado.ravel()[FV])
    area = 0.5 * np.linalg.norm(np.cross(V[FV[:, 1]] - V[FV[:, 0]], V[FV[:, 2]] - V[FV[:, 0]]), axis=1)
    lado = np.sqrt(2 * np.median(area)); pasos = int(np.clip((SUAVIZADO_MM / max(lado, 1e-6)) ** 2, 1, 60))
    # mascara de detalle: lo que se aparta de su entorno en el original (ojos, cejas, labios, lineas finas)
    PR.paso('Buscando rasgos por contraste (ojos, cejas, labios, pezones)')
    Lo = lab(col_orig); okN = N >= 0; Nn = np.where(okN, N, 0)
    entorno = suavizar(Lo, N, max(3, pasos // 2), sigma=1e6)
    detalle = np.linalg.norm(Lo - entorno, axis=1) > DETALLE_DE
    for _ in range(3): detalle = detalle | (detalle[Nn] & okN).any(1)
    # build-17: rasgos por contraste, sin buscar la cara. Un rasgo es una zona pequena que se aparta de su
    # entorno amplio en luz/color (ojos, cejas, labios) o solo en tono (areolas, pezones, labios suaves).
    pasos_ent = int(np.clip((RASGO_ENTORNO_MM / max(lado, 1e-6)) ** 2, 3, 400))
    entorno_r = suavizar(Lo, N, pasos_ent, sigma=1e6)
    dif = np.linalg.norm(Lo - entorno_r, axis=1)
    dtono = np.hypot(Lo[:, 1] - entorno_r[:, 1], Lo[:, 2] - entorno_r[:, 2])
    rasgo = (dif > RASGO_DE_2) | (dtono > TONO_DE)
    for _ in range(2): rasgo = rasgo | (rasgo[Nn] & okN).any(1)
    r = np.repeat(np.arange(len(FV)), 3); c = N.ravel(); cv = np.where(c >= 0, c, 0)
    m = (c >= 0) & rasgo[r] & rasgo[cv]
    nc, compd = connected_components(coo_matrix((np.ones(m.sum()), (r[m], c[m])), shape=(len(FV),) * 2), directed=False)
    a_d = np.bincount(compd, weights=area * rasgo, minlength=nc)
    # solo zonas pequenas: los bordes largos (bikini, costuras) y las sombras grandes no son rasgos
    rasgo_ok = rasgo & (a_d[compd] >= RASGO_MIN_MM2_2) & (a_d[compd] < RASGO_MAX_MM2)
    # rellenar el interior de cada rasgo (centro de la areola, iris): zonas sin marcar pequenas y rodeadas
    m2 = (c >= 0) & ~rasgo_ok[r] & ~rasgo_ok[cv]
    nh, comph = connected_components(coo_matrix((np.ones(m2.sum()), (r[m2], c[m2])), shape=(len(FV),) * 2), directed=False)
    a_h = np.bincount(comph, weights=area, minlength=nh)
    rasgo_ok = rasgo_ok | (~rasgo_ok & (a_h[comph] < RASGO_MAX_MM2))
    n_rasgos = len(np.unique(compd[rasgo & rasgo_ok]))
    detalle = detalle | rasgo_ok
    proteger = rasgo_ok
    for _ in range(3): proteger = proteger | (proteger[Nn] & okN).any(1)
    # build-17: la paleta se elige despues de encontrar los rasgos, con sus colores incluidos,
    # para que cada rasgo (ojo, labio, areola) tenga su propio virtual
    PR.paso('Calculando las mezclas de color posibles')
    W, rgb = candidatos()
    m = col[np.random.default_rng(1).choice(len(col), min(12000, len(col)), replace=False)]
    pi_ = np.nonzero(proteger)[0]
    if len(pi_): m = np.vstack([m, col_orig[np.random.default_rng(3).choice(pi_, min(4000, len(pi_)), replace=False)]])
    PR.paso('Eligiendo los 105 virtuales para este modelo')
    WV, PV = elegir_paleta(W, rgb, m, N_VIRTUALES)
    PAL = np.vstack([base_usable(), PV]); PL = lab(PAL)
    nu = len(MODO['usables']); nf = len(MODO['fisicos'])
    id_de = np.array([u + 1 for u in MODO['usables']] + [nf + 1 + k for k in range(len(PV))])
    # color: zona protegida igual que antes (nitido en el detalle); resto con suavizado fuerte y uniforme
    PR.paso('Suavizando el color de piel y telas (el paso mas largo)')
    Lc = suavizar(lab(col), N, pasos)
    Lc[detalle & proteger] = suavizar(Lo, N, 1, sigma=6.0)[detalle & proteger]
    pasos_piel = int(np.clip((SUAVIZADO_PIEL_MM / max(lado, 1e-6)) ** 2, pasos, 200))
    Lp = suavizar(Lc, N, pasos_piel - pasos, sigma=8.0) if pasos_piel > pasos else Lc
    Lc = np.where(proteger[:, None], Lc, Lp)
    Wp = np.vstack([np.eye(5)[MODO['usables']], WV])
    PR.paso('Asignando un color a cada cara')
    idx = np.empty(len(Lc), int)
    for s0 in range(0, len(Lc), 100000):
        blk = Lc[s0:s0 + 100000]
        idx[s0:s0 + 100000] = (dist(blk, PL) + penal_piel(blk, Wp)).argmin(1)
    idx = np.where(proteger, idx, mayoria(idx, N))
    # superficies lisas: se agrupan por continuidad de color y cada zona grande recibe UN solo tono (sin parches)
    # clases de color gruesas (k-medias): la piel es una clase, el traje otra... no se encadenan entre si
    PR.paso('Unificando superficies lisas en un solo tono')
    rng = np.random.default_rng(2); smp = Lc[rng.choice(len(Lc), min(60000, len(Lc)), replace=False)]
    cen = smp[rng.choice(len(smp), ZONA_CLASES, replace=False)]
    for _ in range(15):
        a_ = dist(smp, cen).argmin(1)
        for k in range(ZONA_CLASES):
            if (a_ == k).any(): cen[k] = smp[a_ == k].mean(0)
    clase = np.concatenate([dist(Lc[i:i + 200000], cen).argmin(1) for i in range(0, len(Lc), 200000)])
    cv2 = np.where(c >= 0, c, 0)
    ok2 = (c >= 0) & ~proteger[r] & ~proteger[cv2] & (clase[r] == clase[cv2])
    ns, seg = connected_components(coo_matrix((np.ones(ok2.sum()), (r[ok2], c[ok2])), shape=(len(FV),) * 2), directed=False)
    a_s = np.bincount(seg, weights=area, minlength=ns)
    med = np.stack([np.bincount(seg, weights=area * Lc[:, j], minlength=ns) for j in range(3)], 1) / np.maximum(a_s, 1e-12)[:, None]
    grande = (a_s >= ZONA_MIN_MM2)[seg] & ~proteger
    if grande.any():
        gs = np.nonzero(a_s >= ZONA_MIN_MM2)[0]
        tono = np.zeros(ns, int); tono[gs] = (dist(med[gs], PL) + penal_piel(med[gs], Wp)).argmin(1)
        cerca = np.linalg.norm(Lc - PL[tono[seg]], axis=1) < ZONA_TOL      # nunca pintar una cara con un tono lejano al suyo
        idx = np.where(grande & cerca, tono[seg], idx)
    PR.paso('Quitando manchas y zonas sueltas')
    idx = quitar_islas(idx, N, area, ISLA_MIN_MM2, Lref=Lc, PL=PL, proteger=detalle & proteger)
    # uniformidad fuera de la zona protegida: parches de tono parecido se funden (piel, telas lisas)
    idx = quitar_islas(idx, N, area, UNIFORME_MM2, Lref=Lc, PL=PL, tol=UNIFORME_DE, proteger=proteger)
    if os.environ.get('DEPURAR'): np.savez(os.environ['DEPURAR'], V=V, F=FV, co=col_orig, P=PAL, i=idx)
    estados = id_de[idx]                  # ids de PrusaSlicer: fisicos 1..N, virtuales N+1..
    carpeta = os.path.join(os.path.dirname(path), f'{nombre}_prusa_{MODO["sufijo"]}'); os.makedirs(carpeta, exist_ok=True)
    PR.paso('Creando la vista previa')
    prev = [render(V, FV, col_orig), render(V, FV, PAL[idx])]
    vp = Image.new('RGB', (1220, 640), 'white'); vp.paste(prev[0], (0, 40)); vp.paste(prev[1], (620, 40))
    from PIL import ImageDraw; d = ImageDraw.Draw(vp); d.text((20, 10), 'Original', fill='black'); d.text((640, 10), f'{MODO["nombre"]} - ColorMix ({len(MODO["usables"])} reales + {len(PV)} virtuales)', fill='black')
    vp.save(os.path.join(carpeta, 'vista_previa.png'))
    render(V, FV, np.where(proteger[:, None], np.array([1.0, 0.0, 0.0]), 0.6 + 0.4 * col_orig)).save(os.path.join(carpeta, 'zonas_protegidas.png'))
    ULTIMO.update(V=V, FV=FV, col=col, col_orig=col_orig, PAL=PAL, idx=idx, prot=proteger, det=detalle, Lc=Lc, clase=clase)
    out = os.path.join(carpeta, f'{nombre}_colormix.3mf')
    PR.paso('Escribiendo el archivo 3MF')
    escribir_3mf(out, nombre, V, FV, estados, WV, PV, prev[1].resize((256, 256)))
    dE = np.linalg.norm(lab(col) - PL[idx], axis=1)
    rep = [f'Impresora: {MODO["nombre"]}', f'Archivo: {path}', f'Triangulos: {len(FV):,}', f'Altura: {V[:, 2].max():.1f} mm',
           f'Colores usados: {len(np.unique(idx))} de {len(PAL)}', f'Error de color por cara (dE) medio: {dE.mean():.1f}',
           f'Motor de color: {"ratdoux v0.9.18 (KM/K-S)" if RATDOUX() is not None else "anterior (falta ratdoux.dll)"}',
           f'Rasgos protegidos: {n_rasgos} zonas, {area[proteger].sum():.0f} mm2 (en rojo en zonas_protegidas.png)', f'Salida: {out}', f'Tiempo: {time.time() - t0:.0f} s']
    PR.fin()
    open(os.path.join(carpeta, 'reporte.txt'), 'w', encoding='utf-8').write('\n'.join(rep)); print('\n'.join(rep))

def main():
    args = [a.strip('"') for a in sys.argv[1:]]
    modo = None
    for f in ('--mmu3', '--indx8'):
        if f in args: modo = f; args.remove(f)
    if modo is None:
        print('Elige la impresora:\n  1) Prusa MMU3  (5 filamentos)\n  2) Prusa INDX 8 (7 cabezales: C M Y W K + Gold + Silver)')
        modo = '--indx8' if input('Opcion [1/2]: ').strip() == '2' else '--mmu3'
    if modo == '--indx8': modo_indx8()
    print(f'Modo: {MODO["nombre"]}')
    altura = None
    if args and args[-1].replace('.', '', 1).isdigit(): altura = float(args.pop())
    if not args: args = [input('Arrastra aqui el .obj o .glb y pulsa Enter: ').strip().strip('"')]
    ok = True
    for a in args:
        try: convertir(os.path.abspath(a), altura)
        except SystemExit as e:
            for pr_ in Progreso.todos: pr_.activo = False
            print(f'\n{e}'); ok = False
        except Exception:
            for pr_ in Progreso.todos: pr_.activo = False
            import traceback; traceback.print_exc(); ok = False
    if not os.environ.get('SIN_PAUSA'): input('\nListo. Pulsa Enter para cerrar...')
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
