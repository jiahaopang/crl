% Test the "cascade residual learning" (CRL) network on the KITTI dataset
% 
% We found that testing at higher resolution improves the accuracy of the 
% disparities, at a cost of more time and memory consumption. Due to memory
% limitation, in this script, we divide a stereo pair into two and estimate
% two disparity maps. Then the two disparity maps are merged into one as 
% the final results.

close all;
clear;
clc

%% initialization
addpath ../matlab % add the paths of caffe
addpath ../tools
caffe.set_mode_gpu();
caffe.set_device(0);
caffe.reset_all();
width_test = 1792;
height_test = 320;
path_left = './kitti_test/image_2/';
path_right = './kitti_test/image_3/';
data_left = dir(path_left);
data_right = dir(path_right);
result_folder = 'disp_0';

%% load model
model = './deploy_kitti.prototxt';
weights = './crl.caffemodel';
net = caffe.Net(model, weights, 'test'); % create net and load weights
coord_x=repmat(0 : width_test - 1, [height_test 1])';
coord_y=repmat((0 : height_test - 1)', [1 width_test])';
coord=zeros(width_test, height_test, 2);
coord(:,:,1) = coord_x;
coord(:,:,2) = coord_y; % construct the canonical coordinates for the Remap layer
net.blobs('zero').set_data(zeros(width_test, height_test));
net.blobs('coord_canon').set_data(coord);
n_data = size(data_left, 1) - 2;
time = 0;
trans_n = 6; % transition height: 2 * trans_n
alpha = repmat(linspace(1, 0, trans_n * 2)', [1 width_test]);
dealpha = 1 - alpha;
mkdir(result_folder);

%% test the model
for i = 1 : n_data
    imgL = imread([path_left, data_left(i + 2).name]);
    imgR = imread([path_right, data_left(i + 2).name]);
    [height_inter, width_inter, ~] = size(imgL);
    height_shrink = round(height_test * width_inter / width_test);
    height_asmb = round(width_test * height_inter / width_inter);
    disp_est_asmb = zeros(height_asmb, width_test);
    height_ovlp = round(height_test - height_asmb / 2);
    imgL_up = imgL(1 : height_shrink, :, :);
    imgL_down = imgL(end - height_shrink + 1 : end, :, :);
    imgR_up = imgR(1 : height_shrink, :, :);
    imgR_down = imgR(end - height_shrink + 1 : end, :, :);

    %% pre-processings on the images: resize, permute, etc.
    imgL_p_up = imresize(imgL_up(:, :, [3, 2, 1]), [height_test width_test], 'nearest'); % permute channels from RGB to BGR
    imgR_p_up = imresize(imgR_up(:, :, [3, 2, 1]), [height_test width_test], 'nearest');
    imgL_p_up = permute(imgL_p_up, [2, 1, 3]);
    imgR_p_up = permute(imgR_p_up, [2, 1, 3]);
    imgL_p_down = imresize(imgL_down(:, :, [3, 2, 1]), [height_test width_test], 'nearest'); % permute channels from RGB to BGR
    imgR_p_down = imresize(imgR_down(:, :, [3, 2, 1]), [height_test width_test], 'nearest');
    imgL_p_down = permute(imgL_p_down, [2, 1, 3]);
    imgR_p_down = permute(imgR_p_down, [2, 1, 3]);
    
    %% estimate two disparity maps
    tic;
    net.blobs('img0').set_data(imgL_p_up);
    net.blobs('img1').set_data(imgR_p_up);
    net.forward_prefilled();
    disp_est_up = -transpose(net.blobs('predict_flow2_s2').get_data());
    net.blobs('img0').set_data(imgL_p_down);
    net.blobs('img1').set_data(imgR_p_down);
    net.forward_prefilled();
    disp_est_down = -transpose(net.blobs('predict_flow2_s2').get_data());
    
    %% assemble the final estimate
    disp_est_asmb(1 : height_test - height_ovlp - trans_n, :) = disp_est_up(1 : height_test - height_ovlp - trans_n, :); % for upper
    disp_est_asmb(height_test - height_ovlp + 1 + trans_n: end, :) = disp_est_down(end - ... % for below
        (height_asmb - height_test + height_ovlp) + 1 + trans_n: end, :);
    disp_est_asmb(height_test - height_ovlp - trans_n + 1 : height_test - height_ovlp + trans_n, :) = ...
        disp_est_up(height_test - height_ovlp - trans_n + 1 : height_test - height_ovlp + trans_n, :) .* alpha + ...
        disp_est_down(end - (height_asmb - height_test + height_ovlp) - trans_n + 1: end - ...
        (height_asmb - height_test + height_ovlp) + trans_n, :) .* dealpha;
    disp_est_asmb = imresize(disp_est_asmb, [height_inter width_inter], 'nearest') * width_inter / width_test;
    time = time + toc;
    
    %% write the result
    imwrite(uint16(disp_est_asmb * 256), [result_folder, '/', ...
        data_left(i + 2).name(1 : end - 4), '.png'], 'BitDepth', 16);
    display(i);
end
time = time / n_data;
display(time);
caffe.reset_all();
