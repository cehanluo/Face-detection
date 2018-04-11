% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

% There is a confusiton that if I need to calculate hog of all images in
% the directory before "get random negative". Now, I only do random before
% hog calculation.

%[~,~,~] = mkdir('../data/neg_extracted');
image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);
cell_per_temp = feature_params.template_size/feature_params.hog_cell_size;
D = cell_per_temp^2 * 31;
features_neg = zeros(num_samples,D,'double'); % initialize a num_images*1116 matrix, element=0
window_size=feature_params.template_size;
cell_size=feature_params.hog_cell_size;
j = 1; % control total sample size
scale=1;
flag = 0;
num_per_img = ceil(num_samples/num_images);

for i=1:num_images
    img_neg=image_files(i); % get negative image position number in image_files
    img_path=fullfile(non_face_scn_path,img_neg.name);
    img=imread(img_path); % imread chosen negative image
    % calculate hog features for current image
    window=window_size*scale;
    w=size(img,2);
    h=size(img,1);
    for r=1:num_per_img
        x = rand();
        y = rand();
        ypos=ceil(y*(h-window));
        ypos1=int16(ypos+window-1);
        xpos=ceil(x*(w-window));
        xpos1=int16(xpos+window-1);
        img_cur=img(ypos:ypos1,xpos:xpos1,:);
        img_cur=imresize(img_cur,[36,36]);
        %imwrite(img_cur, sprintf('../data/neg_extracted/loop%s_%s.jpg', num2str(j)));
        hog_feat=vl_hog(im2single(img_cur),cell_size);
        hog_feat=hog_feat(:)';
        features_neg(j,:)=hog_feat;
        j=j+1;
        if j>num_samples
            flag=1;
            break;
        end
    end
    if flag==1
        break;
    end
end

