import torch
import numpy as np
from model.Transient_model import FBFETGRU  # FBFET_GRU 모델 정의 파일

# -----------------------------
# 1. 모델 로드 및 파라미터 추출
# -----------------------------
model_path = "./checkpoint/FBFET_mixed_GRU/8_2layer/1e-5_0.075_0.075_1.0/8_model_epoch_1000.pt" 
input_dim = 4      # GRU 입력 차원 (예: Lich, wf, Vd, Vg)
hidden_dim = 8     # GRU 은닉 상태 차원 (또한 regressor 첫 번째 층의 입출력 차원)
num_layers = 2
output_dim = 1     # regressor 최종 출력 차원

model = FBFETGRU(input_dim, num_layers, hidden_dim, output_dim)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)

# --- GRU 파라미터 추출 (두 레이어) ---
# GRU Layer 0
w_ih_l0 = state_dict["gru.weight_ih_l0"].cpu().detach().numpy()   # shape: (3*hidden_dim, input_dim)
w_hh_l0 = state_dict["gru.weight_hh_l0"].cpu().detach().numpy()   # shape: (3*hidden_dim, hidden_dim)
bias_ih_l0 = state_dict["gru.bias_ih_l0"].cpu().detach().numpy()    # shape: (3*hidden_dim,)
bias_hh_l0 = state_dict["gru.bias_hh_l0"].cpu().detach().numpy()    # shape: (3*hidden_dim,)

# GRU Layer 1
w_ih_l1 = state_dict["gru.weight_ih_l1"].cpu().detach().numpy()   # shape: (3*hidden_dim, hidden_dim)
w_hh_l1 = state_dict["gru.weight_hh_l1"].cpu().detach().numpy()   # shape: (3*hidden_dim, hidden_dim)
bias_ih_l1 = state_dict["gru.bias_ih_l1"].cpu().detach().numpy()    # shape: (3*hidden_dim,)
bias_hh_l1 = state_dict["gru.bias_hh_l1"].cpu().detach().numpy()    # shape: (3*hidden_dim,)

# GRU의 각 층은 3개의 게이트(update, reset, candidate)를 갖습니다.
# 각 게이트의 크기는 hidden_dim입니다.
# 편향은 bias_ih와 bias_hh를 더한 값으로 사용합니다.
bias_l0 = bias_ih_l0 + bias_hh_l0
bias_l1 = bias_ih_l1 + bias_hh_l1

# --- Regressor 파라미터 추출 ---
# regressor는 nn.Sequential로 구성되어 있으며,
# 구성: [Linear(hidden_dim->hidden_dim), ReLU, Linear(hidden_dim->output_dim)]
linear1 = model.regressor[0]
linear2 = model.regressor[2]

W1_reg = linear1.weight.detach().cpu().numpy()  # shape: (hidden_dim, hidden_dim)
b1_reg = linear1.bias.detach().cpu().numpy()      # shape: (hidden_dim,)

W2_reg = linear2.weight.detach().cpu().numpy()    # shape: (output_dim, hidden_dim)
b2_reg = linear2.bias.detach().cpu().numpy()        # shape: (output_dim,)

# -----------------------------
# 2. Verilog‑A 코드 문자열 생성
# -----------------------------
va_lines = []

# 헤더 및 모듈 선언
va_lines.append('`include "disciplines.vams"')
va_lines.append('`include "constants.vams"')
va_lines.append("")
va_lines.append("module fbfet_gru(vout, clk, Lich, wf, Vd, Vg, h0_0, h0_1, h0_2, h0_3, h0_4, h0_5, h0_6, h0_7, h1_0, h1_1, h1_2, h1_3, h1_4, h1_5, h1_6, h1_7);")
va_lines.append("    input clk, Lich, wf, Vd, Vg, h0_0, h0_1, h0_2, h0_3, h0_4, h0_5, h0_6, h0_7, h1_0, h1_1, h1_2, h1_3, h1_4, h1_5, h1_6, h1_7;")
va_lines.append("    output vout, h0_0, h0_1, h0_2, h0_3, h0_4, h0_5, h0_6, h0_7, h1_0, h1_1, h1_2, h1_3, h1_4, h1_5, h1_6, h1_7;")
va_lines.append("    electrical vin, clk, Lich, wf, Vd, Vg, vout;")
va_lines.append("")

# GRU 각 레이어의 은닉 상태 노드 선언 (layer0와 layer1)
for l in range(num_layers):
    for i in range(hidden_dim):
        va_lines.append(f"    electrical h{l}_{i};")
va_lines.append("")

# regressor의 중간 신호들을 모듈 상단에서 사전에 선언 (real 변수)
va_lines.append("    // Regressor intermediate signal declarations")
for i in range(hidden_dim):
    va_lines.append(f"    real reg_layer1_{i};")
    va_lines.append(f"    real reg_layer1_out_{i};")
for k in range(output_dim):
    va_lines.append(f"    real reg_layer2_{k};")
va_lines.append(f"    real reg_out_0;")
va_lines.append("")

# 함수 정의: sigmoid와 relu (verilog-A 내에서 사용)
va_lines.append("    // Function definitions")
va_lines.append("    function real sigmoid;")
va_lines.append("        input x;")
va_lines.append("        begin")
va_lines.append("            sigmoid = 1.0 / (1.0 + exp(-x));")
va_lines.append("        end")
va_lines.append("    endfunction")
va_lines.append("")
va_lines.append("    function real relu;")
va_lines.append("        input x;")
va_lines.append("        begin")
va_lines.append("            if (x > 0) begin")
va_lines.append("                relu = x;")
va_lines.append("            end else begin")
va_lines.append("                relu = 0;")
va_lines.append("            end")
va_lines.append("        end")
va_lines.append("    endfunction")
va_lines.append("")

# 내부 변수 선언 (GRU 연산용)
va_lines.append("    // Internal real variable declarations for GRU Layer 0")
temp_vars = [f"z0_{i}" for i in range(hidden_dim)]
va_lines.append("    real " + ", ".join(temp_vars) + ";")
temp_vars = [f"r0_{i}" for i in range(hidden_dim)]
va_lines.append("    real " + ", ".join(temp_vars) + ";")
temp_vars = [f"htilde0_{i}" for i in range(hidden_dim)]
va_lines.append("    real " + ", ".join(temp_vars) + ";")
temp_vars = [f"h0_new_{i}" for i in range(hidden_dim)]
va_lines.append("    real " + ", ".join(temp_vars) + ";")
va_lines.append("")
va_lines.append("    // Internal real variable declarations for GRU Layer 1")
temp_vars = [f"z1_{i}" for i in range(hidden_dim)]
va_lines.append("    real " + ", ".join(temp_vars) + ";")
temp_vars = [f"r1_{i}" for i in range(hidden_dim)]
va_lines.append("    real " + ", ".join(temp_vars) + ";")
temp_vars = [f"htilde1_{i}" for i in range(hidden_dim)]
va_lines.append("    real " + ", ".join(temp_vars) + ";")
temp_vars = [f"h1_new_{i}" for i in range(hidden_dim)]
va_lines.append("    real " + ", ".join(temp_vars) + ";")
va_lines.append("")
va_lines.append("    // Regressor output")
va_lines.append("    real y;")
va_lines.append("")

# --- GRU Layer 0 파라미터 ---
for gate in range(3):
    for i in range(hidden_dim):
        for j in range(input_dim):
            name = f"W_ih_l0_{gate}_{i}_{j}"
            va_lines.append(f"    parameter real {name} = {w_ih_l0[gate*hidden_dim + i, j]};")
for gate in range(3):
    for i in range(hidden_dim):
        for j in range(hidden_dim):
            name = f"W_hh_l0_{gate}_{i}_{j}"
            va_lines.append(f"    parameter real {name} = {w_hh_l0[gate*hidden_dim + i, j]};")
for gate in range(3):
    for i in range(hidden_dim):
        name = f"b_l0_{gate}_{i}"
        va_lines.append(f"    parameter real {name} = {bias_l0[gate*hidden_dim + i]};")
va_lines.append("")

# --- GRU Layer 1 파라미터 ---
for gate in range(3):
    for i in range(hidden_dim):
        for j in range(hidden_dim):
            name = f"W_ih_l1_{gate}_{i}_{j}"
            va_lines.append(f"    parameter real {name} = {w_ih_l1[gate*hidden_dim + i, j]};")
for gate in range(3):
    for i in range(hidden_dim):
        for j in range(hidden_dim):
            name = f"W_hh_l1_{gate}_{i}_{j}"
            va_lines.append(f"    parameter real {name} = {w_hh_l1[gate*hidden_dim + i, j]};")
for gate in range(3):
    for i in range(hidden_dim):
        name = f"b_l1_{gate}_{i}"
        va_lines.append(f"    parameter real {name} = {bias_l1[gate*hidden_dim + i]};")
va_lines.append("")

# --- Regressor 파라미터 (두 개의 선형층) ---
# 첫 번째 선형층: 입력 및 출력 차원 모두 hidden_dim
va_lines.append("    // Regressor first linear layer parameters")
for i in range(hidden_dim):
    for j in range(hidden_dim):
        va_lines.append(f"    parameter real W_reg1_{i}_{j} = {W1_reg[i][j]};")
for i in range(hidden_dim):
    va_lines.append(f"    parameter real b_reg1_{i} = {b1_reg[i]};")
va_lines.append("")
# 두 번째 선형층: 출력 차원 output_dim, 입력 차원 hidden_dim
va_lines.append("    // Regressor second linear layer parameters")
for k in range(output_dim):
    for j in range(hidden_dim):
        va_lines.append(f"    parameter real W_reg2_{k}_{j} = {W2_reg[k][j]};")
for k in range(output_dim):
    va_lines.append(f"    parameter real b_reg2_{k} = {b2_reg[k]};")
va_lines.append("")

# -----------------------------
# 3. Analog 블록: GRU 및 Regressor 연산 구현
# -----------------------------
va_lines.append("    analog begin")
va_lines.append("            // --- GRU Layer 0 ---")
va_lines.append("            // 입력 벡터: {V(Lich), V(wf), V(Vd), V(Vg)}")
for i in range(hidden_dim):
    # update gate z0 계산
    expr_z_parts = []
    for j, sig in enumerate(["V(Lich)", "V(wf)", "V(Vd)", "V(Vg)"]):
        expr_z_parts.append(f"(W_ih_l0_0_{i}_{j} * {sig})")
    expr_z_sum = " + ".join(expr_z_parts)
    expr_z = f"sigmoid({expr_z_sum} + W_hh_l0_0_{i}_{i} * V(h0_{i}) + b_l0_0_{i})"
    va_lines.append(f"            z0_{i} = {expr_z};")
    
    # reset gate r0 계산
    expr_r_parts = []
    for j, sig in enumerate(["V(Lich)", "V(wf)", "V(Vd)", "V(Vg)"]):
        expr_r_parts.append(f"(W_ih_l0_1_{i}_{j} * {sig})")
    expr_r_sum = " + ".join(expr_r_parts)
    expr_r = f"sigmoid({expr_r_sum} + W_hh_l0_1_{i}_{i} * V(h0_{i}) + b_l0_1_{i})"
    va_lines.append(f"            r0_{i} = {expr_r};")
    
    # candidate hidden state htilde0 계산
    expr_hcand_parts = []
    for j, sig in enumerate(["V(Lich)", "V(wf)", "V(Vd)", "V(Vg)"]):
        expr_hcand_parts.append(f"(W_ih_l0_2_{i}_{j} * {sig})")
    expr_hcand_sum = " + ".join(expr_hcand_parts)
    expr_hcand = f"tanh({expr_hcand_sum} + W_hh_l0_2_{i}_{i} * (r0_{i} * V(h0_{i})) + b_l0_2_{i})"
    va_lines.append(f"            htilde0_{i} = {expr_hcand};")
    
    # hidden state 업데이트: h0_new = (1 - z0) * V(h0) + z0 * htilde0
    va_lines.append(f"            h0_new_{i} = (1 - z0_{i}) * V(h0_{i}) + z0_{i} * htilde0_{i};")
    va_lines.append(f"            V(h0_{i}) <+ h0_new_{i};\n")
va_lines.append("")
va_lines.append("            // --- GRU Layer 1 ---")
va_lines.append("            // 입력은 GRU Layer 0의 은닉 상태: V(h0_0), V(h0_1), ..., V(h0_7)")
for i in range(hidden_dim):
    expr_z_parts = []
    for j in range(hidden_dim):
        expr_z_parts.append(f"(W_ih_l1_0_{i}_{j} * h0_new_{j})")
    expr_z_sum = " + ".join(expr_z_parts)
    expr_z = f"sigmoid({expr_z_sum} + W_hh_l1_0_{i}_{i} * V(h1_{i}) + b_l1_0_{i})"
    va_lines.append(f"            z1_{i} = {expr_z};")
    
    expr_r_parts = []
    for j in range(hidden_dim):
        expr_r_parts.append(f"(W_ih_l1_1_{i}_{j} * h0_new_{j})")
    expr_r_sum = " + ".join(expr_r_parts)
    expr_r = f"sigmoid({expr_r_sum} + W_hh_l1_1_{i}_{i} * V(h1_{i}) + b_l1_1_{i})"
    va_lines.append(f"            r1_{i} = {expr_r};")
    
    expr_hcand_parts = []
    for j in range(hidden_dim):
        expr_hcand_parts.append(f"(W_ih_l1_2_{i}_{j} * h0_new_{j})")
    expr_hcand_sum = " + ".join(expr_hcand_parts)
    expr_hcand = f"tanh({expr_hcand_sum} + W_hh_l1_2_{i}_{i} * (r1_{i} * V(h1_{i})) + b_l1_2_{i})"
    va_lines.append(f"            htilde1_{i} = {expr_hcand};")
    
    va_lines.append(f"            h1_new_{i} = (1 - z1_{i}) * V(h1_{i}) + z1_{i} * htilde1_{i};")
    va_lines.append(f"            V(h1_{i}) <+ h1_new_{i};\n")
va_lines.append("")
va_lines.append("            // --- Regressor (Output Layer) ---")
va_lines.append("            // 첫 번째 regressor 선형층 + ReLU (입력: GRU Layer 1 은닉 상태)")
for i in range(hidden_dim):
    terms = " + ".join([f"W_reg1_{i}_{j} * h1_new_{j}" for j in range(hidden_dim)])
    va_lines.append(f"            reg_layer1_{i} = b_reg1_{i} + {terms};")
    va_lines.append(f"            reg_layer1_out_{i} = relu(reg_layer1_{i});")
va_lines.append("")
va_lines.append("            // 두 번째 regressor 선형층 + ReLU")
for k in range(output_dim):
    terms = " + ".join([f"W_reg2_{k}_{j} * reg_layer1_out_{j}" for j in range(hidden_dim)])
    va_lines.append(f"            reg_layer2_{k} = b_reg2_{k} + {terms};")
    va_lines.append(f"            reg_out_{k} = relu(reg_layer2_{k});")
    va_lines.append(f"            y = reg_out_{k};")
    va_lines.append(f"            V(vout) <+ y;")
va_lines.append("    end")
va_lines.append("endmodule")

# -----------------------------
# 4. 최종 Verilog‑A 코드 파일 저장
# -----------------------------
va_code = "\n".join(va_lines)
with open("fbfet_gru_final.va", "w") as f:
    f.write(va_code)

print("Verilog-A code has been generated and saved as 'fbfet_gru_final.va'.")
