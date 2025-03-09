
*----------------------------
* Run 58: Vg = 1.56 V (Ascending)
*----------------------------
.OPTIONS NUMDGT=15
.OPTIONS POST=2
.hdl "constants.vams"
.hdl "disciplines.vams"
.hdl "fbfet_gru_final.va"

.param Lich = 0.6953285598*40.8598567936371
.param wf   = 0.3425383017*4.83312357942386
.param Vd   = 0.5
.param Vg_val = 1.56

Vclk   n_clk 0 PULSE(0 1 0 1n 1n 0.5u 1u)
VLich  n_Lich 0 DC 0.6953285598
Vwf    n_wf   0 DC 0.3425383017
Vd     n_Vd   0 DC 0.5
Vg     n_Vg   0 DC 1.56

.reinit opfile_run57.op

X1 n_vout n_clk n_Lich n_wf n_Vd n_Vg h0_0 h0_1 h0_2 h0_3 h0_4 h0_5 h0_6 h0_7 \
   h1_0 h1_1 h1_2 h1_3 h1_4 h1_5 h1_6 h1_7 fbfet_gru

.dc Vg 1.48 1.56 0.08

.write opfile_run58.op

.print I(X1.vout) V(n_Vg) V(X1.h0_0) V(X1.h0_1) V(X1.h0_2) V(X1.h0_3) V(X1.h0_4) V(X1.h0_5) V(X1.h0_6) V(X1.h0_7) V(X1.h1_0) V(X1.h1_1) V(X1.h1_2) V(X1.h1_3) V(X1.h1_4) V(X1.h1_5) V(X1.h1_6) V(X1.h1_7)

.end
