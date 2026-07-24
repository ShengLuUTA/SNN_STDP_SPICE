%% ================================
%  MNIST → 8x8 → Rate-based Poisson Spike Train (with labels)
%  ================================

clc; clear; close all;

%% Parameters
duration_us = 100;      % total duration (us)
dt_us       = 1;        % timestep (us)
T           = duration_us / dt_us;

%% Step 1: Download MNIST images + labels
fprintf('Checking MNIST raw files...\n');

imgURL   = 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz';
lblURL   = 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz';

imgGZ    = 'train-images-idx3-ubyte.gz';
lblGZ    = 'train-labels-idx1-ubyte.gz';

imgFile  = 'train-images-idx3-ubyte';
lblFile  = 'train-labels-idx1-ubyte';

% --- Images ---
if ~isfile(imgFile)
    fprintf('MNIST image file missing. Downloading...\n');
    websave(imgGZ, imgURL);
    gunzip(imgGZ);
else
    fprintf('MNIST image file exists. Skipping download.\n');
end

% --- Labels ---
if ~isfile(lblFile)
    fprintf('MNIST label file missing. Downloading...\n');
    websave(lblGZ, lblURL);
    gunzip(lblGZ);
else
    fprintf('MNIST label file exists. Skipping download.\n');
end

%% Step 2: Load MNIST images
fprintf('Loading MNIST images...\n');

fid = fopen(imgFile, 'rb');
magic    = fread(fid, 1, 'int32', 0, 'ieee-be');
numImages= fread(fid, 1, 'int32', 0, 'ieee-be');
numRows  = fread(fid, 1, 'int32', 0, 'ieee-be');
numCols  = fread(fid, 1, 'int32', 0, 'ieee-be');

images = fread(fid, inf, 'unsigned char');
fclose(fid);

images = reshape(images, numCols, numRows, numImages);
images = permute(images, [2 1 3]);  % 28x28xN

%% Step 3: Load MNIST labels
fprintf('Loading MNIST labels...\n');

fid = fopen(lblFile, 'rb');
magicLbl = fread(fid, 1, 'int32', 0, 'ieee-be');
numLabels= fread(fid, 1, 'int32', 0, 'ieee-be');
labels   = fread(fid, inf, 'unsigned char');
fclose(fid);

if numLabels ~= numImages
    error('Label count does not match image count.');
end

%% Step 4: Resize to 8x8 and save dataset
if isfile('mnist_8x8_dataset.mat')
    fprintf('8x8 dataset already exists. Loading...\n');
    load('mnist_8x8_dataset.mat', 'images8', 'labels8');
else
    fprintf('Resizing to 8x8 and saving dataset...\n');

    images8 = zeros(8, 8, numImages, 'double');

    for i = 1:numImages
        images8(:,:,i) = imresize(images(:,:,i), [8 8], 'bilinear');
    end

    images8 = images8 / 255;
    labels8 = labels;   % store labels alongside images

    save('mnist_8x8_dataset.mat', 'images8', 'labels8', '-v7.3');
    fprintf('Saved 8x8 dataset to mnist_8x8_dataset.mat\n');
end

%% Step 5: Rate-based Poisson encoding (max 50 spikes per pixel)
if isfile('mnist_spike_trains.mat')
    fprintf('Spike-train file already exists. Skipping generation.\n');
    load('mnist_spike_trains.mat', 'spikes', 'labels_spike');
else
    fprintf('Generating rate-based Poisson spike trains...\n');

    num_pixels = 64;
    max_spikes = 40;     % pixel value 1 → 50 spikes total
    spikes = zeros(num_pixels, T, numImages, 'logical');

    for n = 1:numImages
        img     = images8(:,:,n);
        img_vec = reshape(img, [], 1);   % 64x1

        lambda = img_vec * max_spikes;   % expected spike count
        p_spike = lambda / T;            % per-timestep probability

        for t = 1:T
            spikes(:, t, n) = rand(num_pixels, 1) < p_spike;
        end
    end

    labels_spike = labels;  % attach labels to spike dataset

    save('mnist_spike_trains.mat', 'spikes', 'images8', 'labels_spike', '-v7.3');
    fprintf('Saved spike trains to mnist_spike_trains.mat\n');
end

% if isfile('mnist_spike_trains.mat')
%     fprintf('Spike-train file already exists. Skipping generation.\n');
%     load('mnist_spike_trains.mat', 'spikes', 'labels_spike');
% else
%     fprintf('Generating UNIFORM (non-random) spike trains...\n');
% 
%     num_pixels = 64;
%     max_spikes = 40;     % pixel value 1 → 40 spikes total
%     spikes = false(num_pixels, T, numImages);
% 
%     for n = 1:numImages
%         img     = images8(:,:,n);
%         img_vec = reshape(img, [], 1);   % 64x1
% 
%         % 每个像素的脉冲数量（均匀分布）
%         spike_count = round(img_vec * max_spikes);
% 
%         for i = 1:num_pixels
%             if spike_count(i) > 0
%                 % 在 [1, T] 内均匀分布 spike_count(i) 个脉冲
%                 spike_times = round(linspace(1, T, spike_count(i)));
%                 spikes(i, spike_times, n) = true;
%             end
%         end
%     end
% 
%     labels_spike = labels;
%     save('mnist_spike_trains.mat', 'spikes', 'images8', 'labels_spike', '-v7.3');
%     fprintf('Saved UNIFORM spike trains to mnist_spike_trains.mat\n');
% end

%% Step 6: Example visualization
figure;
subplot(1,2,1);
imshow(images8(:,:,1), []);
title('8x8 Image');

subplot(1,2,2);
imagesc(spikes(:,:,100));
xlabel('Time step');
ylabel('Neuron Index');
title('Rate-based Poisson Spike Train');
colormap(gray);

% generate_sp_file(1, 'mnist_spike_trains.mat', 'mnist_sample_1.sp', 1.20, 0.00);

generate_sp_file_samples('mnist_spike_trains.mat', ...
    'mnist_digit1_20samples_7.sp', ...
    1.2, 0.0, [0 4 7], 20);

function generate_sp_file(idx, matfile, outfile, V_high, V_low)

    % Load spike data
    data = load(matfile);
    spikes = data.spikes;          % [64 x 100 x N]
    labels = data.labels_spike;    % labels
    % Extract one sample
    spike_sample = spikes(:,:,idx);   % [64 x 100]
    label = labels(idx);
    [num_pixels, T] = size(spike_sample);   % T = 100
    % Open output .sp file
    fid = fopen(outfile, 'w');
    fprintf(fid, "* HSPICE PWL file for MNIST sample %d (label %d)\n", idx, label);
    fprintf(fid, "* Each spike point = 1us, expanded to 10 points of 0.1us\n\n");
    % Generate 64 PWL sources
    for px = 1:num_pixels
        fprintf(fid, "Vin%d vin%d 0 PWL(\n", px, px);
        for t = 1:T
            base_time = (t-1) * 1.0;   % each spike point = 1us
            Vout = V_low + spike_sample(px,t) * (V_high - V_low);
            % Expand each 1us interval into 10 × 0.1us samples
            for k = 0:9
                time_us = base_time + 0.1 * k;
                fprintf(fid, "+ %.1fu %.2fV\n", time_us, Vout);
            end
        end
        fprintf(fid, ")\n\n");
    end
    fclose(fid);
end

function generate_sp_file_samples(matfile, outfile, V_high, V_low, target, numsample)

% Load spike data
data = load(matfile);
spikes = data.spikes;          % [64 x 100 x N]
labels = data.labels_spike;    % labels

num_pixels = size(spikes, 1);   % 64
T = size(spikes, 2);            % 100 (each = 1us)

% ---- Balanced sampling: each class picks numsample samples ----
idx_list = [];
label_list = [];

for c = target
    idx_c = find(labels == c);

    if length(idx_c) < numsample
        error('Not enough samples for class %d.', c);
    end

    % pick first numsample (or use randperm for random)
    idx_pick = idx_c(1:numsample);

    idx_list = [idx_list; idx_pick];
    label_list = [label_list; labels(idx_pick)];
end

% ---- Shuffle while avoiding consecutive identical labels ----
valid = false; limit_time = 0;
while ~valid && limit_time < 100
    order = randperm(length(idx_list));
    idx_shuffled = idx_list(order);
    label_shuffled = label_list(order);
    % Check for consecutive identical labels
    if all(label_shuffled(1:end-1) ~= label_shuffled(2:end))
        valid = true;
    end
    limit_time = limit_time + 1;
end

num_samples = length(idx_shuffled);

% ---- Write SPICE PWL file ----
fid = fopen(outfile, 'w');
fprintf(fid, "* HSPICE PWL file for MNIST samples of digits: %s\n", mat2str(target));
fprintf(fid, "* Balanced sampling: %d samples per class\n", numsample);
fprintf(fid, "* Total samples = %d\n", num_samples);
fprintf(fid, "* Order shuffled with no consecutive identical labels\n", num_samples);
fprintf(fid, "* Each sample = 100us, total = %dus\n", 100 * num_samples);
fprintf(fid, "* Each spike point = 1us, expanded to 10 points of 0.1us\n\n");

% ---- Generate 64 PWL sources ----
for px = 1:num_pixels
    fprintf(fid, "Vin%d vin%d 0 PWL(\n", px, px);

    global_time = 0.0;

    for s = 1:num_samples
        spike_sample = spikes(:, :, idx_shuffled(s));

        for t = 1:T
            Vout = V_low + spike_sample(px, t) * (V_high - V_low);

            for k = 0:9
                time_us = global_time + 0.1 * k;
                fprintf(fid, "+ %.1fu %.2fV\n", time_us, Vout);
            end

            global_time = global_time + 1.0;
        end
    end

    fprintf(fid, ")\n\n");
end

fclose(fid);
end

