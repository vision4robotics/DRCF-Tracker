% This demo script runs the DRCF tracker on the included video.

% Add paths
setup_paths();

%  Load video information
base_path  =  './sequences';
%video  = choose_video(base_path);
video = 'wakeboard5';

video_path = [base_path '/' video];
[seq, gt_boxes] = load_video_info(video_path);

% Run DRCF
results = run_DRCF(seq);

% pd_boxes = results.res;
% thresholdSetOverlap = 0: 0.05 : 1;
% success_num_overlap = zeros(1, numel(thresholdSetOverlap));
% res = calcRectInt(gt_boxes, pd_boxes);
% for t = 1: length(thresholdSetOverlap)
%     success_num_overlap(1, t) = sum(res > thresholdSetOverlap(t));
% end
% cur_AUC = mean(success_num_overlap) / size(gt_boxes, 1);
FPS_vid = results.fps;
display([video  '---->' '   FPS:   ' num2str(FPS_vid) ]);