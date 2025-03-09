
*----------------------------
* Run 49: Vg = 0.8399999999999999 V
*----------------------------
.OPTIONS NUMDGT=12
.OPTIONS POST=2
.hdl "constants.vams"
.hdl "disciplines.vams"
.hdl "fbfet_gru_final.va"

.param Lich = 0.6953285598*40.8598567936371
.param wf   = 0.3425383017*4.83312357942386
.param Vd   = 0.5
.param Vg_val = 0.8399999999999999

Vclk   n_clk 0 PULSE(0 1 0 1n 1n 0.5u 1u)
VLich  n_Lich 0 DC 0.6953285598
Vwf    n_wf   0 DC 0.3425383017
Vd     n_Vd   0 DC 0.5
Vg     n_Vg   0 DC 0.8399999999999999

.reinit opfile_run48.op

X1 n_vout n_clk n_Lich n_wf n_Vd n_Vg h0_0 h0_1 h0_2 h0_3 h0_4 h0_5 h0_6 h0_7 \
   h1_0 h1_1 h1_2 h1_3 h1_4 h1_5 h1_6 h1_7 fbfet_gru

.dc Vg 0.8399999999999999 0.8399999999999999 1

.write opfile_run49.op

.end
