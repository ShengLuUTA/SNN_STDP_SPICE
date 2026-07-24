clc; clear; fclose('all');
roots = pwd;

%% ================= PATH =================
hspice_path = 'C:\Synopsys\Hspice_J-2014.09-2\WIN64\hspice.com';
cd(roots);
rootpath = pwd;

%% ================= TEMP DIR =================
workdir = '.\temp';
if ~exist(workdir, 'dir')
    mkdir(workdir);
end
cd(workdir);

%% ================= GENERATE SPICE =================
M = 4; N = 64; timestep = 0.2; simulationtime = 100;
% generate_spice('stdp.sp',rootpath);
generate_spice_model_inhabi('stdp.sp', rootpath, N, M, 1.0e-9, 0.01, timestep, simulationtime)

%% ================= RUN HSPICE =================
cmd = sprintf('"%s" stdp.sp -o out', hspice_path);
system(cmd);
cd(roots);

%% ================= READ RESULT =================
% res = read_rram_lis('out.lis');
[res, ~] = import_data([rootpath, '\temp\out.lis']);

%% ================= PLOT =================
neuron_stride = 2 + N;   % 每个神经元占用的列数
for ineuron = 1:M

    figure(ineuron);

    base = 1 + (ineuron-1)*neuron_stride;   % 每个神经元的起始列

    time_col = 1;
    vout_col = base + 1;
    vmem_col = base + 2;
    rram_cols = (base + 3):(base + 2 + N);

    % ==== 1. RRAM 权重热力图 ====
    subplot(3,1,1);
    weights = reshape(1 ./ res(end, rram_cols), 8, 8);
    imagesc(weights);
    colorbar;
    title(sprintf('Neuron %d - RRAM Weight Map', ineuron));
    axis equal tight;
    colormap jet;

    % ==== 2. RRAM 电导变化 ====
    subplot(3,1,2);
    plot(res(:, time_col), 1 ./ res(:, rram_cols));
    title(sprintf('Neuron %d - RRAM Conductance Evolution', ineuron));
    xlabel('Time');
    ylabel('Conductance (1/R)');

    % ==== 3. 膜电压 ====
    subplot(3,1,3);
    plot(res(:, time_col), res(:, vout_col:vmem_col));
    title(sprintf('Neuron %d - Membrane Voltage', ineuron));
    xlabel('Time');
    ylabel('V_{mem}');
end

% plot_results(res);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ================= READ LIS =================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function data = read_rram_lis(filename)

fid = fopen(filename, 'r');
if fid < 0
    error('Cannot open file.');
end

data_blocks = {};
current_block = [];
reading_block = false;
expected_cols = 0;

while true
    line = fgetl(fid);
    if ~ischar(line), break; end

    % Detect header line
    if contains(line, 'time') && contains(line, 'voltage')
        reading_block = true;
        current_block = [];
        expected_cols = 0;   % reset
        continue;
    end

    % Detect end of block
    if reading_block && (isempty(strtrim(line)) || strcmp(strtrim(line), 'y'))
        if ~isempty(current_block)
            data_blocks{end+1} = current_block; %#ok<AGROW>
        end
        reading_block = false;
        continue;
    end

    % Read numeric rows
    if reading_block
        nums = sscanf(line, '%f')';

        % First numeric row determines number of columns
        if expected_cols == 0
            expected_cols = numel(nums);
        end

        % Accept rows with matching column count
        if numel(nums) == expected_cols
            current_block = [current_block; nums]; %#ok<AGROW>
        end
    end
end

fclose(fid);

%% Combine all blocks
time = data_blocks{1}(:,1);
all_R = [];

for k = 1:length(data_blocks)
    block = data_blocks{k};
    all_R = [all_R, block(:,2:end)]; %#ok<AGROW>
end

data = [time, all_R];
end


function generate_spice_model_inhabi(filename, rootpath, N, M, base_gap, gap_sigma, timestep, simulationtime)

if nargin < 1, filename = 'stdp.sp'; end
if nargin < 2, rootpath = pwd; end
if nargin < 3, N = 3; end
if nargin < 4, M = 2; end
if nargin < 5, gap_sigma = 0.05; end   % 默认 5% 高斯扰动

fid = fopen(filename, 'w');
rootpath = strrep(rootpath, '\', '/');

%% ===== Header =====
fprintf(fid, '.title SNN + RRAM STDP (Gaussian gap variation + lateral connections)\n');
fprintf(fid, '.option post ingold=1\n');
fprintf(fid, '.option runlvl=2\n\n');

fprintf(fid, ".include '%s/TLV2372.LIB'\n", rootpath);
fprintf(fid, ".include '%s/cmp.sp'\n", rootpath);
fprintf(fid, ".include '%s/mnist_digit1_20samples_7.sp'\n", rootpath);
fprintf(fid, ".LIB '%s/sm046005-1j.hspice' typical\n", rootpath);
fprintf(fid, ".hdl '%s/edge_to_pulse.va'\n", rootpath);
fprintf(fid, ".hdl '%s/rram.va'\n", rootpath);
fprintf(fid, ".hdl '%s/spikegen.va'\n", rootpath);
fprintf(fid, ".hdl '%s/follow_res.va'\n", rootpath);
% fprintf(fid, ".hdl '%s/lateral_bus.va'\n\n", rootpath);

fprintf(fid, '.global vdd gnd\n\n');

%% ===== Power =====
fprintf(fid, 'VDD  vdd  0 1.2\n');
fprintf(fid, 'VEE  vee  0 -1.2\n');
fprintf(fid, 'VDDL vddl 0 1.2\n');
fprintf(fid, 'VDDC vddc 0 0.53\n\n');

%% ===== Top-level neuron instances =====
fprintf(fid, '* === M Neuron Instances ===\n');

% base_gap = 1.4e-9;

for m = 1:M
    % 高斯扰动 gap_ini_base
    gap_value = base_gap * (1 + gap_sigma * randn());

    fprintf(fid, 'XNEU%d ', m);
    for i = 1:N
        fprintf(fid, 'vin%d ', i);
    end

    fprintf(fid, ...
        'vdd vee vddl vddc vout_SRC%d node2_%d single_neuron gap_ini_base=%gn\n', ...
        m, m, gap_value);
end
fprintf(fid, '\n');

%% ===== Lateral connections =====
fprintf(fid, '* === Lateral Connections: vout_SRC_m -> node2_k (k != m) ===\n');
% 
% for m = 1:M
%     for k = 1:M
%         if k ~= m
%             fprintf(fid, 'Dlat_%d_%d vout_SRC%d node2_%d DI_1N4001G\n', m, k, k, m);
%         end
%     end
% end
% fprintf(fid, '\n');

for m = 1:M
    for k = 1:M
        if k ~= m
            fprintf(fid, 'M_%d_%d node2_%d vout_SRC%d XNEU%d.node1 gnd nmos_3p3 L=0.35u W=20u\n', m,k,m,k,m);
        end
    end
end
fprintf(fid, '\n');

%% ===== Lateral bus instance =====
% fprintf(fid, '* === Lateral Inhibition Bus ===\n');
% 
% % Build node list: vout_SRC1 ... vout_SRCM node2_1 ... node2_M
% xbus_line = 'XBUS';
% 
% % vout_SRC[1..M]
% for m = 1:M
%     xbus_line = sprintf('%s vout_SRC%d', xbus_line, m);
% end
% 
% % node2_[1..M]
% for m = 1:M
%     xbus_line = sprintf('%s node2_%d', xbus_line, m);
% end
% 
% % Append subckt name and parameter M
% xbus_line = sprintf('%s lateral_bus M=%d', xbus_line, M);
% 
% fprintf(fid, '%s\n\n', xbus_line);


%% ===== .ic for all neurons =====
fprintf(fid, '* === Initial Conditions ===\n.ic \\\n');

for m = 1:M
    fprintf(fid, '  V(XNEU%d.vout)=0 V(XNEU%d.vout_SRC)=0 \\\n', m, m);
    fprintf(fid, '  V(XNEU%d.node1)=0 V(XNEU%d.node2)=0 V(XNEU%d.node3)=0 \\\n', m, m, m);
    fprintf(fid, '  V(XNEU%d.node4)=0 V(XNEU%d.node6)=0 V(XNEU%d.node7)=0 V(XNEU%d.node8)=0', ...
        m, m, m, m);

    if m < M
        fprintf(fid, ' \\\n');
    else
        fprintf(fid, '\n\n');
    end
end

%% ===== .tran =====
fprintf(fid, '.tran %.2fus %.2fus\n', timestep, simulationtime);

%% ===== .print for all neurons =====
fprintf(fid, '* === Print All Neurons ===\n.print ');

for m = 1:M
    fprintf(fid, 'V(XNEU%d.vout_SRC) V(XNEU%d.node2) ', m, m);
    for i = 1:N
        fprintf(fid, 'V(XNEU%d.Xrram%d.R_out) ', m, i);
    end
end
fprintf(fid, '\n\n');

%% ===== Subcircuit Definition =====
fprintf(fid, '.subckt single_neuron ');
for i = 1:N
    fprintf(fid, 'vin%d ', i);
end
fprintf(fid, 'vdd vee vddl vddc vout_SRC node2\n');

fprintf(fid, '* --- Input channels ---\n');
for i = 1:N
    fprintf(fid, 'EIN%d  TOP%d  0 vin%d 0 1\n',  i, i, i);
    fprintf(fid, 'EINV%d vinc%d 0 vin%d 0 -1\n', i, i, i);
    fprintf(fid, 'Xrw%d  vinc%d node1 R_out%d follow_res\n', i, i, i);
    fprintf(fid, 'Xspk%d TOP%d TOP_SRC%d spikegen vth=0.8 a=-0.20 b=0.0 Tr=3us\n', i, i, i);

    % 使用 gap_ini_base
    fprintf(fid, 'Xrram%d TOP_SRC%d vout_SRC R_out%d rram gap_ini=gap_ini_base tstep=%.2fus\n\n', ...
        i, i, i, timestep);
end

fprintf(fid, '* --- Neuron core ---\n');
fprintf(fid, 'XU1 node3 node1 vdd vee node2 TLV2372\n');
fprintf(fid, 'C1 node1 node2 1n\n');
fprintf(fid, 'R5 node3 gnd 10k\n');
fprintf(fid, 'R1 node2 node4 4.7k\n');
% fprintf(fid, 'C2 node2 gnd 100p\n');
fprintf(fid, 'XU2 node4 gnd vdd vee node6 TLV2372\n');
fprintf(fid, 'R4 node6 node7 1k\n');
fprintf(fid, 'R3 node7 node8 100\n');
fprintf(fid, 'D1 node8 node1 DI_1N4001G\n');
fprintf(fid, 'R2 node4 node7 9.4k\n');
fprintf(fid, 'XCMP node7 gnd cmpout vddl vddc cmp\n');
fprintf(fid, 'X1 cmpout vout edge_to_pulse vth=0.5 vhigh=1.0 width=1u\n');
fprintf(fid, 'Xspkout vout vout_SRC spikegen vth=0.8 a=-0.20 b=0.0 Tr=3us\n');

fprintf(fid, '.ends single_neuron\n\n');

fprintf(fid, '.subckt current_mirror vout_SRC TOP_SRC1 node1_inv\n');
fprintf(fid, 'M_1 vout_SRC vout_SRC 0 0 nmos_3p3 L=0.35u W=20u\n');
fprintf(fid, 'R_1 TOP_SRC1 TOP_SRC2 500k\n');
fprintf(fid, 'M_2 TOP_SRC2 vout_SRC node1_inv 0 nmos_3p3 L=0.35u W=20u\n');
fprintf(fid, '.ends current_mirror\n\n');


fprintf(fid, '.end\n');

fclose(fid);
fprintf('✔ 生成成功：%s（N=%d 输入，M=%d 神经元，高斯σ=%.1f%%，含侧向连接）\n', ...
    filename, N, M, gap_sigma*100);
end


function data = read_rram_lis_fast(filename)

% ====== 1. 读取整个文件 ======
txt = fileread(filename);

% ====== 2. 找到所有数据块（以 time 开头，以空行结束） ======
% 匹配：
%   time ...\n
%   数字行...
%   空行结束
block_pattern = 'time[^\n]*\n((?:[ \t]*[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?[^\n]*\n)+)';
blocks = regexp(txt, block_pattern, 'tokens');

if isempty(blocks)
    error('未找到任何数据块');
end

% ====== 3. 解析每个 block ======
num_blocks = numel(blocks);
parsed = cell(1, num_blocks);

for k = 1:num_blocks
    block_text = blocks{k}{1};

    % 将 block 中所有数字一次性读出
    nums = sscanf(block_text, '%f');

    % 自动推断列数
    % 取第一行的列数
    first_line = regexp(block_text, '([^\n]+)', 'match', 'once');
    ncol = numel(sscanf(first_line, '%f'));

    % reshape 成矩阵
    parsed{k} = reshape(nums, ncol, []).';
end

% ====== 4. 拼接所有 block ======
time = parsed{1}(:,1);
all_R = [];

for k = 1:num_blocks
    all_R = [all_R, parsed{k}(:,2:end)];
end

data = [time, all_R];
end
