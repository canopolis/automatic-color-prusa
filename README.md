# FullSpectrumPrusa
Version aprobada: modo Normal con la cara y la limpieza de manchas aprobadas (sin modo Fisico ni cambios posteriores).
Arrastra un .obj full spectrum (con su .mtl y textura) sobre FullSpectrumPrusa.exe y elige:
1) Prusa MMU3  - C M Y W K + 105 virtuales ColorMix
2) Prusa INDX 8 - C M Y W K + Gold Glitter + Silver Glitter + 105 virtuales
La primera vez en INDX 8 pide un .3mf guardado en PrusaSlicer con la impresora "full color print indx8"
(un cubo basta) y guarda su configuracion junto al exe.
Salida: <nombre>_prusa_mmu3 o <nombre>_prusa_indx8 con el .3mf, vista_previa.png y reporte.txt.

Cambios build-16 (sobre la version aprobada):
- La cara se busca por contraste de TONO (ojos, labios), no por cantidad de manchas: las sombras de
  plumas, pelo o rocas ya no se confunden con la cara (antes podia proteger un ala y dejar la cara borrosa).
- La cara se conserva: ojos, cejas, nariz y labios nitidos; nada la altera despues.
- Piel de un solo color: tono de referencia = piel clara del cuerpo; sombras rodeadas de piel pasan a piel.
- Pezones/areolas (manchas pequenas mas rosadas que la piel) conservan su color.
- Piel y pezones solo se procesan si se detecta una cara: en modelos sin cara el resultado es igual a build-15.
