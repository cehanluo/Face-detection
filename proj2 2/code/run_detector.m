% Starter code prepared by James Hays for CS 143, Brown University
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector(test_scn_path, w, b, feature_params, mu, sigma)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);
%cell_per_temp = feature_params.template_size/feature_params.hog_cell_size;
cell_size=feature_params.hog_cell_size;
window_size=feature_params.template_size;
pixel_per_step = 6;
%scales = [1.2,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.15,0.1,0.06];  % initial scale
scales = [1.15,0.95,0.8,0.65,0.5,0.35,0.2,0.1,0.06];
%scales = [1.25,1.1,0.95,0.8,0.65,0.5,0.35,0.2,0.1,0.06];

for i = 1:length(test_scenes)
    
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img=imread( fullfile( test_scn_path, test_scenes(i).name ));
    img=single(img)/255;
    cur_bboxes=zeros(0,4);
    cur_confidences=zeros(0,1);
    cur_image_ids=cell(0,1);
    cur_num=1;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
    % calculate hog features for current image
    for scale=scales
        img_cur=imresize(img,scale);
        nxsteps=floor((size(img_cur,2)-window_size)/pixel_per_step);
        nysteps=floor((size(img_cur,1)-window_size)/pixel_per_step);
        for xb=1:nxsteps
            for yb=1:nysteps
                ypos=int16((yb-1)*pixel_per_step+1);
                ypos1=int16(ypos+window_size-1);
                xpos=int16((xb-1)*pixel_per_step+1);
                xpos1=int16(xpos+window_size-1);
                img_win=img_cur(ypos:ypos1,xpos:xpos1,:);
                hog_feat=vl_hog(im2single(img_win),cell_size);
                hog_feat=hog_feat(:)';
                hog_feat_norm=(hog_feat-mu)./sigma;
                confidence=hog_feat_norm*w+b;
                if confidence>0.75
                    box=int32([xpos,ypos,xpos1,ypos1]/scale);
                    cur_bboxes=[cur_bboxes;box;];
                    cur_confidences=[cur_confidences; confidence;];
                    cur_image_ids{cur_num,1}=test_scenes(i).name;
                    cur_num=cur_num+1;
                end
            end
        end
    end


    
    
    
    
        
  
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
 
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
end




