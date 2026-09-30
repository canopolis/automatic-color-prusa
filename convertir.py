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

ALTURA_MM = 100.0      # altura de impresion si el modelo viene sin escala real
CELDA_MM = 0.4         # escala de suavizado del color (ancho de boquilla)
SUAVIZADO_MM = 0.8     # radio de suavizado del color sobre la superficie
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
def mezcla(R):
    """color predicho de una mezcla por capas (modelo sustractivo ajustado a PrusaSlicer ColorMix)"""
    L = s2l(base_mezcla()) + A_MEZCLA
    return l2s(np.exp(R @ np.log(L)) - A_MEZCLA)

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

def dist(A, B):
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

def quitar_islas(idx, N, area, min_area, rondas=40, Lref=None, PL=None, tol=10.0):
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
    P = V[:, [0, 2, 1]] * np.array([1, 1, -1])           # frente: X derecha, Z arriba, -Y hacia camara
    N = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]); N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    lo = P[:, :2].min(0); hi = P[:, :2].max(0); sc = (size - 20) / (hi - lo).max()
    fi = np.where(N[:, 2] > 0)[0]; c = fcol[fi] * np.clip(0.35 + 0.65 * N[fi, 2], 0, 1)[:, None]
    img = np.ones((size, size, 3)); zb = np.full(size * size, -1e9)
    for w in [[1/3,1/3,1/3],[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]]:
        p = w[0] * P[F[fi, 0]] + w[1] * P[F[fi, 1]] + w[2] * P[F[fi, 2]]
        x = ((p[:, 0] - lo[0]) * sc + 10).astype(int); y = (size - 1 - ((p[:, 1] - lo[1]) * sc + 10)).astype(int)
        pix = y * size + x; z = p[:, 2]; o = np.lexsort((-z, pix)); pu, first = np.unique(pix[o], return_index=True); sel = o[first]
        u = z[sel] > zb[pu]; zb[pu[u]] = z[sel][u]; img.reshape(-1, 3)[pu[u]] = c[sel][u]
    return Image.fromarray((img * 255).astype(np.uint8))

ULTIMO = {}
# ---------------- principal ----------------
def convertir(path, altura):
    t0 = time.time(); nombre = os.path.splitext(os.path.basename(path))[0]
    print(f'Cargando {path} ...')
    V, VC, T, FV, FT, tex = cargar_obj(path)
    print(f'  {len(FV):,} triangulos | textura: {os.path.basename(tex) if tex else "no"} | color por vertice: {"si" if VC is not None else "no"}')
    # OBJ (Y arriba) -> 3MF (Z arriba), escala y centrado en cama
    V = V @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    alto = V[:, 2].max() - V[:, 2].min()
    if alto < 20 or altura: V = V * ((altura or ALTURA_MM) / alto)
    V = V - [(V[:, 0].max() + V[:, 0].min()) / 2 - BED[0], (V[:, 1].max() + V[:, 1].min()) / 2 - BED[1], V[:, 2].min()]
    # color por cara, suavizado a escala de boquilla
    if tex is not None:
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
    print('Eligiendo 105 virtuales para este modelo ...')
    W, rgb = candidatos()
    m = col[np.random.default_rng(1).choice(len(col), min(12000, len(col)), replace=False)]
    WV, PV = elegir_paleta(W, rgb, m, N_VIRTUALES)
    PAL = np.vstack([base_usable(), PV]); PL = lab(PAL)
    nu = len(MODO['usables']); nf = len(MODO['fisicos'])
    id_de = np.array([u + 1 for u in MODO['usables']] + [nf + 1 + k for k in range(len(PV))])
    print('Asignando colores (suavizado + limpieza de manchas) ...')
    N = vecinos(FV)
    area = 0.5 * np.linalg.norm(np.cross(V[FV[:, 1]] - V[FV[:, 0]], V[FV[:, 2]] - V[FV[:, 0]]), axis=1)
    lado = np.sqrt(2 * np.median(area)); pasos = int(np.clip((SUAVIZADO_MM / max(lado, 1e-6)) ** 2, 1, 60))
    Lc = suavizar(lab(col), N, pasos)
    Wp = np.vstack([np.eye(5)[MODO['usables']], WV])
    idx = np.empty(len(Lc), int)
    for s0 in range(0, len(Lc), 100000):
        blk = Lc[s0:s0 + 100000]
        idx[s0:s0 + 100000] = (dist(blk, PL) + penal_piel(blk, Wp)).argmin(1)
    idx = mayoria(idx, N)
    idx = quitar_islas(idx, N, area, ISLA_MIN_MM2, Lref=Lc, PL=PL)
    if os.environ.get('DEPURAR'): np.savez(os.environ['DEPURAR'], V=V, F=FV, co=col_orig, P=PAL, i=idx)
    estados = id_de[idx]                  # ids de PrusaSlicer: fisicos 1..N, virtuales N+1..
    carpeta = os.path.join(os.path.dirname(path), f'{nombre}_prusa_{MODO["sufijo"]}'); os.makedirs(carpeta, exist_ok=True)
    prev = [render(V, FV, col_orig), render(V, FV, PAL[idx])]
    vp = Image.new('RGB', (1220, 640), 'white'); vp.paste(prev[0], (0, 40)); vp.paste(prev[1], (620, 40))
    from PIL import ImageDraw; d = ImageDraw.Draw(vp); d.text((20, 10), 'Original', fill='black'); d.text((640, 10), f'{MODO["nombre"]} - ColorMix ({len(MODO["usables"])} reales + {len(PV)} virtuales)', fill='black')
    vp.save(os.path.join(carpeta, 'vista_previa.png'))
    ULTIMO.update(V=V, FV=FV, col=col, col_orig=col_orig, PAL=PAL, idx=idx)
    out = os.path.join(carpeta, f'{nombre}_colormix.3mf')
    print('Escribiendo 3MF ...')
    escribir_3mf(out, nombre, V, FV, estados, WV, PV, prev[1].resize((256, 256)))
    dE = np.linalg.norm(lab(col) - PL[idx], axis=1)
    rep = [f'Impresora: {MODO["nombre"]}', f'Archivo: {path}', f'Triangulos: {len(FV):,}', f'Altura: {V[:, 2].max():.1f} mm',
           f'Colores usados: {len(np.unique(idx))} de {len(PAL)}', f'Error de color por cara (dE) medio: {dE.mean():.1f}',
           f'Salida: {out}', f'Tiempo: {time.time() - t0:.0f} s']
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
    if not args: args = [input('Arrastra aqui el .obj y pulsa Enter: ').strip().strip('"')]
    ok = True
    for a in args:
        try: convertir(os.path.abspath(a), altura)
        except Exception:
            import traceback; traceback.print_exc(); ok = False
    if not os.environ.get('SIN_PAUSA'): input('\nListo. Pulsa Enter para cerrar...')
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
