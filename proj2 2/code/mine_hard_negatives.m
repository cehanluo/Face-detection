function features_hneg = mine_hard_negatives(non_face_scn_path, w, b, feature_params, mu, sigma, loop, num)
% 'bboxes' is Nx4, N is the number of non-overlapping detections, and each
% row is [x_min, y_min, x_max, y_max]
% 'confidences' is the Nx1 (final cascade node) confidence of each
% detection.
% 'image_ids' is the Nx1 image names for each detection.

%this lists the ground truth bounding boxes for the test set.

mine_scenes = dir( fullfile( non_face_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
features_hneg = [];
confidences = zeros(0,1);
image_ids = cell(0,1);
cell_per_temp = feature_params.template_size/feature_params.hog_cell_size;
cell_size=feature_params.hog_cell_size;
window_size=feature_params.template_size;
pixel_per_step = 4;
%scales = [1.2,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.15,0.1,0.06];
%scales = [1,1.5,2,2.5,3,3.5,4,4.5];
scales = [1];
j=1;
%h= randperm(length(mine_scenes),50);
for i = 1:length(mine_scenes)
    fprintf('Mining faces in %s\n', mine_scenes(i).name)
    img=imread( fullfile( non_face_scn_path, mine_scenes(i).name ));
    img=single(img)/255;
    cur_bboxes=zeros(0,4);
    cur_confidences=zeros(0,1);
    cur_image_ids=cell(0,1);
    cur_num=1;
    cur_hog=[];
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
                if confidence>0.5
                    imwindow=imresize(img_win,[36,36]);
                    %imwrite(imwindow, sprintf('../data/neg_0.886_4pi/average_precision3pi_loop%s_mine_%s_%s.jpg', num2str(loop), num2str(num), num2str(j)));
                    box=int32([xpos,ypos,xpos1,ypos1]/scale);
                    cur_bboxes=[cur_bboxes;box;];
                    cur_confidences=[cur_confidences; confidence;];
                    cur_image_ids{cur_num,1}=mine_scenes(i).name;
                    cur_hog=[cur_hog;hog_feat;];
                    cur_num=cur_num+1;
                    j=j+1;
                end
            end
        end
    end
    
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));
    
    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
    cur_hog         = cur_hog(        is_maximum,:);
    
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
    features_hneg= [features_hneg;cur_hog];
    for n=1:size(cur_bboxes,1)
        xwin=cur_bboxes(n,1);
        ywin=cur_bboxes(n,2);
        xwin1=cur_bboxes(n,3);
        ywin1=cur_bboxes(n,4);
        imwindow=img(ywin:ywin1,xwin:xwin1,:);
        imwindow=imresize(imwindow,[36,36]);
        imwrite(imwindow, sprintf('../data/neg_mine/mine_%s_im_%s_%s.jpg', num2str(num), num2str(i), num2str(n)));
    end
end





