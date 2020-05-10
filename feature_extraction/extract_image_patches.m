% function img_samples = extract_image_patches(image, pos, scales, features, gparams, extract_info)
function img_samples = extract_image_patches(image, pos, scales,extract_info)
% argin：extrat_info 里面samplesize是采样大小，采样完了在get_pixels里会被压缩到input_size

% Sample image patches at given position and scales. Then extract features
% from these patches.
% Requires that cell size and image sample size is set for each feature.


num_scales = length(scales);
num_sizes = length(extract_info.img_sample_sizes);

% Extract image patches
% img_samples = cell(num_sizes,1);
% for sz_ind = 1:num_sizes
%     img_sample_sz = extract_info.img_sample_sizes{sz_ind};
%     img_input_sz = extract_info.img_input_sizes{sz_ind};
%     img_samples{sz_ind} = zeros(img_input_sz(1), img_input_sz(2), size(image,3), num_scales, 'uint8');
%     for scale_ind = 1:num_scales
%         img_samples{sz_ind}(:,:,:,scale_ind) = get_pixels(image, pos, round(img_sample_sz*scales(scale_ind)),img_input_sz);
%     end
% end

img_sample_sz = extract_info.img_sample_sizes{1};
img_input_sz = extract_info.img_input_sizes{1};
img_samples= zeros(img_input_sz(1), img_input_sz(2), size(image,3), 'uint8');
img_samples(:,:,:) = get_pixels(image, pos, round(img_sample_sz*scales(1)),img_input_sz);
