# HYPOTHESIS LEDGER - RX 7800 XT / Qwen3.8-27B Q2 Investigation
Date: 2026-08-22 | llama.cpp b10524-6-g849798132 | ROCm 7.2.4 | gfx1101
Evidence base: experiments ctx_*_mtp_on_*, T1_mtpon, T3_mtpoff, X_fixshape, X_nmax1,
traces T1/T3/X_fixshape (rocprofv3 runtime-trace CSV), source tree @849798132.

Scale: Rejected < Weak < Tentative < Supported < Strongly supported

---
H1: "Q2 quantized GEMV/GEMM ana decode darboğazidir."
PREDICTION: Trace'te tek bir Q2_K matvec kernel grubunun baskin (%50+) gorulmesi.
EVIDENCE FOR: MMVQ toplami buyuk: %34 GPU @63k OFF, ~%45 @63k ON (w_63k_decode_on_clean.json).
EVIDENCE AGAINST:
  - Agirliklar aslinda Q2_K degil: GGUF census -> FFN=IQ3_XXS(type18), attn=IQ3_S(21),
    lm_head=Q3_K(11). "UD-Q2_K_XL" karisim.
  - Pay daginik: en buyuk tekil grup (IQ3_XXS fused FFN mmvq) bile %31.5 (ON) - %57 (1k OFF fazlari
    icinde) ama bu "tek suclu" degil; cok sayida cagri (150us mean).
  - OFF modda FA %49 ile mmvq'yu geciyor @63k.
ALTERNATIVES: (a) FA kernel verimsizligi baskin, (b) launch/sync.
FALSIFICATION TEST: Yapildi - FA'yi devre disi birakarak (kisa baglam) kiyas: 1k'da mmvq baskin
ama mutlak sureler kucuk; uzun baglamda FA geciyor.
VERDICT: Rejected (primary). Confidence 0.85. Ikincil katki olarak gercek.

---
H2: "Q2 dequantization overhead ana darbogazdir."
PREDICTION: Agirlik dequant kernel'lerinin (to_fp16 vb.) buyuk payi.
EVIDENCE FOR: Yok (agirlik dequant MMVQ/MMA icinde fuse; ayri agirlik dequant trace'te yok).
EVIDENCE AGAINST:
  - quantize_q8_1 (aktivasyon kvant.) sadece %0.9 GPU.
  - ANCAK: KV->F16 tam-cache dequant (dequantize_block_q4_0<__half>) MTP-ON verify'de
    %6.8 GPU @63k - TILE kernel geregi. Bu H8'in sonucu.
ALTERNATIVES: FA secimi, batch sekli.
FALSIFICATION TEST: X_nmax1 (ncols<=2 -> VEC yolu): dequant kayboldu ama performans dustu
(14.6 tok/s) - dequant tek basina darbogaz degil.
VERDICT: Rejected (as stated). Confidence 0.9. KV-dequant varyanti Supported (bkz H8).

---
H3: "Uzun contextte attention/KV cache bandwidth darbogazi baskindir."
PREDICTION: FA sureleri context ile buyuyecek; 63k'da baskin olacak.
EVIDENCE FOR:
  - FA-VEC 2.47ms/call @63.5k vs 81us @1k (30x, 65x KV) - T3 phase 18/19.
  - 63k OFF: FA %49.2 GPU (39.4ms/token; 16 layer x 2.47ms).
  - Etkin KV bantgenisligi ~33GB/s / 624GB/s tepe = %5.3 (82MB/layer-call / 2.47ms).
    Bant genisligi siniri olsaydi 2.1ms/toplam-token yeterdi; olculen 39.4ms -> ~18x verimsiz.
  - Prefill'de flash_attn_tile<256,256,16,2> 78.7ms/call (gec ubatch'ler).
EVIDENCE AGAINST:
  - Kisa baglamda FA ihmal edilebilir (<%1 @1k) - darbogaz context'e bagli, mutlak degil.
  - "Bandwidth darbogazi" ifadesi yanlis: bandwidth YETERSIZ KULLANIM var, bandwidth siniri yok.
ALTERNATIVES: (a) kernel occupancy/compute siniri (RDNA3 dp4a yolu), (b) grid yapisi.
FALSIFICATION TEST: Ayni KV ile f16-KV calistirip VEC'in hizlanmadigini gormek (yapilmadi -
VRAM yetmez; 16k altinda mumkun, not edildi).
VERDICT: Strongly supported (uzun context, kernel-verimsizligi bicimiyle). Confidence 0.95.

---
H4: "MTP verification beklenen hizlanmayi sinirlamaktadir."
PREDICTION: Verify tur maliyeti round-basina yuksek; kabul ile hiz korele.
EVIDENCE FOR:
  - Round maliyeti @63k ON: 126-200ms (hedef verify + catch-up + ~3 draft decode).
  - Kabul-hiz korelasyonu r=+0.54 (n=10); run bazinda acc %51 -> 23.5 tok/s, acc %100 -> 40.8.
  - Teorik 5x yerine efektif ~1.9-2.7x (OFF 11.2 -> ON 21.7-30).
EVIDENCE AGAINST:
  - MTP yine de net kazandiriyor (her yerde OFF > ON hizli).
  - X_nmax1: daha kisa spülasyon DAHA yavas (14.6 tok/s) - sorun "cok verification" degil,
    round basina sabit maliyetlerin token'a bolunmesi.
ALTERNATIVES: (a) draft context'in kendi KV maliyeti, (b) grafik churn (H6).
FALSIFICATION TEST: X_fixshape/X_nmax1 yapildi.
VERDICT: Supported. Confidence 0.8.

---
H5: "GPU VRAM bandwidth'i yetersiz kullanilmaktadir."
EVIDENCE FOR:
  - FA-VEC %5.3 etkin bant genisligi @63k (yukarida).
  - IQ3_XXS fused FFN mmvq: 190us/call, ~57MB okuma -> ~300GB/s (%48) - orta seviye.
  - lm_head Q3_K: 1468us, ~218MB -> 149GB/s (%24).
EVIDENCE AGAINST:
  - Prefill MMQ GEMM'leri yuksever verimli (mulmat_other 40.6% global ama buyuk is).
VERDICT: Strongly supported (decode FA icin), Supported (matvec). Confidence 0.9.

---
H6: "Kernel launch/CPU dispatch/sync overhead onemlidir."
EVIDENCE FOR:
  - 1k ON: GPU busy sadece %74.8 -> duvarin %25'i bosluk.
  - MTP grafik churn'u: T1 @63k 25s pencerede 104,967 hipLaunchKernel vs 662 hipGraphLaunch;
    hipGraphInstantiate 5x20ms; GraphExecUpdate 181x580us. OFF modda ayni pencere:
    9,812 eager / 253 graph. Neden: verify batch boyutu 1-5 arasi degisiyor ->
    "2 ard ardina ozdes cagri" capture sarti bozuluyor (ggml-cuda.cu:4263-4282).
  - hipStreamSynchronize: 22.1s kumulatif / 25s pencere (807us mean) - CPU GPU'yu bekliyor.
  - X_fixshape (sabit sekil): eager launch %40 azaldi, 1k varyans TAMAMEN yok oldu
    (37-41 tok/s bandi vs 23-47), taban yukseldi -> mekanizma dogrulandi.
EVIDENCE AGAINST:
  - 63k OFF: GPU %91 busy -> launch overhead ikincil.
  - API tarafi saf maliyet dusuk (hipLaunchKernel 3-4.5us x N = ~%2 wall).
ALTERNATIVES: sampling CPU maliyeti (olculmedi - bilinmiyor bileseni).
VERDICT: Supported (MTP modunda guclu, kisa contextte baskin). Confidence 0.85.

---
H7: "ROCm/HIP backend'i RX 7800 XT'de mimariye ozgu verimsiz kod yolu kullanıyor."
EVIDENCE FOR:
  - fattn-vec.cuh:74-79: RDNA3 icin quantize-K kooperatif thread sayisi hardcoded 2
    (diger mimarilerde 32); __launch_bounds__(128,1).
  - fattn.cu:515: AMD WMMA-MMA yolu D<=128 sartina bagli -> head_dim=256 hicbir zaman MMA kullanmiyor;
    device guard fattn-mma-f16.cuh:1762 ayni.
  - Sonuc: %5.3 bant genisligi kullanimi (H3/H5).
EVIDENCE AGAINST:
  - Matvec kernels makul calisiyor (%48); prefill GEMM iyi.
  - Vulkan kiyaslama verimiz yok (issue #20934 raporu [REPORTED] benzer sikayet ediyor).
ALTERNATIVES: Genel RDNA3 olgunluk sorunu, rocWMMA kaldirilmasinin ardindan bosluk (#26046).
VERDICT: Supported (FA yolunda Spesifik ve kanitli). Confidence 0.85.

---
H8: "llama.cpp kernel secimi RDNA3 icin optimal degildir."
EVIDENCE FOR:
  - Verify batch (>2 query) -> VEC yolu kapali (fattn.cu:528 Q->ne[1]<=2 sarti) -> TILE+
    TAM-CACHE F16 dequant zorunlu; @63k'da round basina 32x273.8us dequant + tile FA.
  - D=256 -> MMA disi (H7).
EVIDENCE AGAINST:
  - Secim mantigi RDNA3'e ozel degil; NVIDIA'da ayni secim farkli sonuc verir (MMA acik).
VERDICT: Supported. Confidence 0.8.

---
H9: "VRAM kapasitesi veya KV yerlesimi dolayli kayip yaratiyor."
EVIDENCE FOR: VRAM 15.35GB/17.16GB zirve; spill/GTT artisi yok; clock saglikli.
EVIDENCE AGAINST: Yukaridaki tum veriler; f16-KV alternatifi bu kartta 64k icin SIGMIYOR
(+~9.1GB iki context icin) - kapasite siniri VAR ama performans kaybi yaratmiyor.
VERDICT: Rejected. Confidence 0.9.

---
H10: "29.34 tok/s'nin onemli kismi sampling veya CPU tarafinda kayboluyor."
EVIDENCE FOR: 1k ON %25 bosluk (sampling+launch+sync zinciri); backend_sampling=false ->
sampling CPU'da her token.
EVIDENCE AGAINST:
  - Memcpy D2H toplam <192ms/25s pencere (<%1) - logits kopyalama sucLU degil.
  - 63k OFF GPU %91 busy - CPU tarafindan kayip minimal.
  - Sampling'in ayristirilmis olcumu yapilmadi (CPU profiler gerekir) -> BILINMEYEN bilesen.
VERDICT: Tentative (kisa context MTP'de katkili; buyukluk bilinmiyor). Confidence 0.4.

---
H11: "Mean speculative length=3 nedeniyle pipeline yeterince verimli degil."
EVIDENCE FOR: Gercek drafts/round 2.4-3.26, kabul %53-100 degisken; tok/round 2.7-5.0.
EVIDENCE AGAINST: X_nmax1 (length=1) DAHA KOTU: 14.6 vs 21.7-30 tok/s @63k. Kisa spülasyon
sabit maliyeti daha az tokene boluyor. Sorun uzunluk degil, degiskenlik+maliyet.
VERDICT: Rejected (ifadenin biçimi). Confidence 0.75.

---
H12: "Darbogaz tek bir kernel degil; bilesik."
SYSTEM-LEVEL SYNTHESIS:
  - Uzun context: FA-VEC/TILE verimsizligi (%39-49 GPU) + MTP grafik churn'u + KV dequant.
  - Kisa context: CPU launch/sync/sampling zinciri (%25 bosluk) + matvec tavanı.
  - Matvec'ler her durumda yuksek payda ama tekil sucLU yok.
VERDICT: Strongly supported. Confidence 0.9.

---
H13: "gfx11'de byte-wise sign uygulamasinin (__vcmpne4/__vsub4) VALU emulasyon zinciri, IQ3_XXS/IQ3_S MMVQ decode kernel'inin ana maliyetidir."
PREDICTION: Native u32x24 carpma (v_mul_u32_u24_e32) ile ikame, dispatch-basi VALU
  talimatlarinin ve kernel surelerinin buyuk kismini kaldirir; occupanciyi bozmadan.
EVIDENCE FOR:
  - PMU (rocprofv3, type18 n=1, test-backend-ops perf): SQ_INSTS_VALU 21.01M ->
    6.32M (-69.9%)/dispatch; sure 105.20 -> 45.34 us (-56.9%); GRBM_GUI_ACTIVE
    276,607 -> 124,819 (-54.9%).
  - Disassembly: baseline 16x v_perm_b32 + 32x v_sub_nc_i16 + v_lshlrev_b16/
    v_cndmask_b32_e64 zinciri; patched'da v_mul_u32_u24_e32 (0x204081) + xor/add.
  - Op-level: iq3_xxs n=1 114.28 -> 51.33 us/run (-55.1%); iq3_s n=1 113.65 ->
    57.32 (-49.6%); n=512 flat (GEMM path etkilenmedi).
  - VGPR: type18 n=1 80 -> 48; type21 n=1 80 -> 40 (occupancy baskisi azalmis,
    artmamis).
EVIDENCE AGAINST:
  - Full-model duvar oncesi/sonrasi olculmedi (build-baseline kontamine, GPU dolu)
    -> kernel kazanminin wall'a gecis oranini dogrulanmadi.
  - 31.62 tok/s uretim sayisi tek basina ve kullanicinin raporu (log yok); MTP
    durumu bilinmiyor.
ALTERNATIVES: (a) gather chain hoist (EXP-006: +4.6% ort, restore edildi),
  (b) LDS staging (EXP-022: -2.5%, REVERT), (c) grid tablosu repack (deferred).
FALSIFICATION TEST: T-9 - patched build'de rocprofv3 occupancy (resident waves/SIMD)
  + Phase-3 gate benchmark (MTP OFF, 5 rep, sabit seed, 256 tok, 1k/16k/63k ctx).
  Occupancy dususe veya wall kazaniminin %15'in altinda kalmasi haliinde falsified.
VERDICT: Tentative. Confidence 0.7.

---
USER'S 29.34 tok/s OBSERVATION IN CONTEXT:
- ~6.3k prompt ile olculmus; bizim 6.9k ON bandimiz 27.9-49.8 tok/s (median ~36).
- 29.34 bandin alt-orta kisitesi: dusuk-kabul bir dongude tipik deger. Anormal degil.
