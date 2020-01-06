function results = tracker(params)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get sequence info
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return;
end

% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end

global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;

% Load learning parameters
admm_max_iterations = params.max_iterations;
init_penalty_factor = params.init_penalty_factor;
max_penalty_factor = params.max_penalty_factor;
penalty_scale_step = params.penalty_scale_step;

init_target_sz = target_sz;

if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false 
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account 
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor(base_target_sz * params.search_area_scale);
        % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2);
        % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2];
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'exact');

% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = unique(feature_info.data_sz, 'rows', 'stable');
feature_cell_sz = unique(feature_info.min_cell_size, 'rows', 'stable');
num_feature_blocks = size(feature_sz, 1);

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);

% Size of the extracted feature maps
feature_sz_cell = mat2cell(feature_sz, ones(1,num_feature_blocks), 2);
filter_sz = feature_sz;
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size
[output_sz, k1] = max(filter_sz, [], 1);
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% Construct the Gaussian label function 
yf = cell(numel(num_feature_blocks), 1);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma = sqrt(prod(floor(base_target_sz/feature_cell_sz(i)))) * params.output_sigma_factor;
    rg           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]     = ndgrid(rg,cg);
    y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
    yf{i}           = fft2(y); 
end

% Compute the cosine windows 
cos_window = cellfun(@(sz) hann(sz(1))*hann(sz(2))', feature_sz_cell, 'uniformoutput', false);

reg_window = cell(num_feature_blocks, 1);
for i = 1:num_feature_blocks
    reg_scale = floor(base_target_sz/params.feature_downsample_ratio(i));
    use_sz = filter_sz_cell{i};
    reg_window{i} = ones(use_sz) * params.reg_window_max;
    range = zeros(numel(reg_scale), 2);
    
    % determine the target center and range in the regularization windows
    for j = 1:numel(reg_scale)
        range(j,:) = [0, reg_scale(j) - 1] - floor(reg_scale(j) / 2);
    end
    center = floor((use_sz + 1)/ 2) + mod(use_sz + 1,2);
    range_h = (center(1)+ range(1,1)) : (center(1) + range(1,2));
    range_w = (center(2)+ range(2,1)) : (center(2) + range(2,2));
    
    reg_window{i}(range_h, range_w) = params.reg_window_min;
end

% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;

% Use the translation filter to estimate the scale
scale_sigma = sqrt(params.num_scales) * params.scale_sigma_factor;
ss = (1:params.num_scales) - ceil(params.num_scales/2);
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
ysf = single(fft(ys));
if mod(params.num_scales,2) == 0
    scale_window = single(hann(params.num_scales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(params.num_scales));
end
ss = 1:params.num_scales;
scaleFactors = params.scale_step.^(ceil(params.num_scales/2) - ss);
if params.scale_model_factor^2 * prod(params.init_sz) > params.scale_model_max_area
    params.scale_model_factor = sqrt(params.scale_model_max_area/prod(params.init_sz));
end

if prod(params.init_sz) > params.scale_model_max_area
    params.scale_model_factor = sqrt(params.scale_model_max_area/prod(params.init_sz));
end
scale_model_sz = floor(params.init_sz * params.scale_model_factor);

% set maximum and minimum scales
min_scale_factor = params.scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(params.scale_step));
max_scale_factor = params.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(params.scale_step));

seq.time = 0;

cf_f = cell(num_feature_blocks, 1);

% Allocate 
scores_fs_feat = cell(1,1,num_feature_blocks);
while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end
    
    tic();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we
    % just want to initialize the tracker there
    if seq.frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);
            
            % sample_scale = currentScaleFactor*scaleFactors;
            sample_scale = currentScaleFactor * 1;
            
            xt = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info);
            
            % Do windowing of features
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);
            
            % Compute convolution for each feature block in the Fourier domain
            % and the sum over all blocks.
            scores_fs_feat{k1} = gather(sum(bsxfun(@times, conj(cf_f{k1}), xtf{k1}), 3));
            scores_fs_sum = scores_fs_feat{k1};
            for k = block_inds
                scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(cf_f{k}), xtf{k}), 3));
                scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                scores_fs_sum = scores_fs_sum +  scores_fs_feat{k};
            end
            
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);
            
            responsef_padded = resizeDFT2(scores_fs, output_sz);
            response = ifft2(responsef_padded, 'symmetric');
            [disp_row, disp_col, ~] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, output_sz);
                                    
            translation_vec = [disp_row, disp_col] .* (img_support_sz./output_sz) * currentScaleFactor;
           
            % update position
            old_pos = pos;
            pos = sample_pos + translation_vec;
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
                        
            xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
            xsf = fft(xs,[],2);
            scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + params.scale_lambda) ));
            % find the maximum scale response
            recovered_scale = find(scale_response == max(scale_response(:)), 1);
            % update the scale
            currentScaleFactor = currentScaleFactor * scaleFactors(recovered_scale);
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            iter = iter + 1;
            
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    sample_pos = round(pos);
    
    % extract image region for training sample;
    xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
    
    if seq.frame == 1
        
        tracking_result.center_pos = double(pos);
        tracking_result.target_size = double(target_sz);
        
    end
    
    do_sa_target_size = 20;
    
    if min(tracking_result.target_size) <= do_sa_target_size
        
        reg_window_input{1} = reg_window{1} ;
        
    else
        
        Saliency_extract_info.img_input_sizes{1,1}=round(tracking_result.target_size*2);           
        Saliency_extract_info.img_sample_sizes{1,1}=round(tracking_result.target_size*2);

        rgbpatch = extract_image_patches(im, sample_pos, 1, Saliency_extract_info);
                
        [origin_row , origin_col,~]=size(rgbpatch);
        
        do_sa_in_size = 32;
        
        rgbpatch = imresize(rgbpatch, [do_sa_in_size , do_sa_in_size]);
                
        %Spectral Residual
        myFFT = fft2(rgb2gray(rgbpatch));
        myLogAmplitude = log(abs(myFFT));
        myPhase = angle(myFFT);
        mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', 3), 'replicate');
        saliencyMap = abs(ifft2(exp(mySpectralResidual + 1i * myPhase))).^2;
        
        %After Effect
        saliencyMap = mat2gray(imfilter(saliencyMap, fspecial('gaussian', [10, 10], 2.5)));

        salpatch = imresize(saliencyMap, [ origin_row , origin_col ] );
       
        patchscale{1} = size(salpatch) ;
        window = cellfun(@(sz) hann(sz(1))*hann(sz(2))', patchscale, 'uniformoutput', false);
        salpatch = salpatch .* window{1} ;
        
        sal_resize_scale=tracking_result.target_size/currentScaleFactor/feature_cell_sz * 2;
        salpatch_in_feature_size = imresize(salpatch , sal_resize_scale );
        
        m_in_feature_size = salpatch_in_feature_size;
                
        bi_thrs = params.bi_thrs ;
       
        m_in_feature_size(salpatch_in_feature_size(:) <= bi_thrs ) = 10000;
        m_in_feature_size(salpatch_in_feature_size(:) >  bi_thrs ) = 0.001;

        [m_mask_row,m_mask_col]=size(m_in_feature_size);
        
        m_input = 10000 * ones(filter_sz);
        
        minput_rowmin = round((filter_sz(1)-m_mask_row)/2);
        minput_rowmax = minput_rowmin + m_mask_row - 1;
        
        minput_colmin = round((filter_sz(2)-m_mask_col)/2);
        minput_colmax = minput_colmin + m_mask_col - 1;
        
        try
            m_input(minput_rowmin:minput_rowmax,minput_colmin:minput_colmax) = m_in_feature_size;

            reg_window_input{1} = 10000*ones(filter_sz);
            
            reg_window_input{1}( m_input(:) <=1 & reg_window{1}(:) <=1 ) = 0.001;
        catch
            reg_window_input{1} = reg_window{1};
        end

    end
    
    % do windowing of features
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
    
    % compute the fourier series
    xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
    
    if seq.frame == 1
        model_xf = xlf{1};
    else
        model_xf = (1 - params.Lrate) * model_xf + params.Lrate * xlf{1};
    end
   
    % our method
    for k = 1: numel(xlf)
        
        
        model_xf = xlf{k};
        
        % intialize the variables
        g_f = single(zeros(size(model_xf)));
        w_f = g_f;
        h_f = g_f;
        rho  = init_penalty_factor(k);
        gamma_max = max_penalty_factor(k);
        gamma_scale_step = penalty_scale_step(k);
        
        % use the GPU mode
        if params.use_gpu
            model_xf = gpuArray(model_xf);
            g_f = gpuArray(g_f);
            w_f = gpuArray(w_f);
            h_f = gpuArray(h_f);
            reg_window_input{k} = gpuArray(reg_window_input{k});
            yf{k} = gpuArray(yf{k});
            %             sp_mask_resized=gpuArray(sp_mask_resized);
        end
        
        % pre-compute the variables
        S_xx = sum(conj(model_xf) .* model_xf, 3);
        
        % solve via ADMM algorithm
        iter = 1;
        
        lambda = 1;
        
        
        while (iter <= admm_max_iterations)
            
            f0_f = fft2( ifft2(w_f) .* reg_window_input{k} );
            
            % subproblem f
            Pool__extract_info = S_xx + rho;
            Sf0x_f = sum(conj(model_xf) .* f0_f, 3);
            Shx_f = sum(conj(model_xf) .* h_f, 3);
            
            g_f = ((1 / rho  * bsxfun(@times,  yf{k}, model_xf)) -  h_f + w_f ) - ...
                bsxfun(@rdivide,( 1/ rho * bsxfun(@times, model_xf, (S_xx .*  yf{k})) - ...
                (bsxfun(@times, model_xf, Shx_f)) + (bsxfun(@times, model_xf, Sf0x_f))), Pool__extract_info);
            
            %   subproblem g
            w_f = fft2(argmin_w(reg_window_input{k}, rho, lambda, real(reg_window_input{k} .* ifft2(rho * g_f+ h_f)), w_f));
            
            %   update h
            h_f = h_f +  g_f - w_f ;
            
            %   update gamma
            rho = min(gamma_scale_step * rho, gamma_max);
            
            iter = iter+1;
        end
        
        % save the trained filters
        if seq.frame == 1
            cf_f{k} = reg_window_input{k} .* w_f;
        else
            cf_f{k} = (1 - params.Lrate) *  cf_f{k} + params.Lrate * ( reg_window_input{k} .* w_f ) ;            
        end
               
    end
    
    %% Upadate Scale
    xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
    xsf = fft(xs,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    
    if seq.frame == 1
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        sf_den = (1 - params.learning_rate_scale) * sf_den + params.learning_rate_scale * new_sf_den;
        sf_num = (1 - params.learning_rate_scale) * sf_num + params.learning_rate_scale * new_sf_num;
    end
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;
    
    
    
    %% save position
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    
    
    %calculate FPS
    seq.time = seq.time + toc();
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % visualization
    if params.visualization
        figure(1);
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        
        imagesc(im_to_show);
        hold on;
        rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
        text(10, 10, [int2str(seq.frame) '/'  int2str(size(seq.image_files, 1))], 'color', [0 1 1]);
        hold off;
        axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        
        drawnow
    end
    fprintf([int2str(seq.frame) ,'/' , int2str(size(seq.image_files, 1)),'\n']);    
end

[~, results] = get_sequence_results(seq);

disp(['fps: ' num2str(results.fps)])

