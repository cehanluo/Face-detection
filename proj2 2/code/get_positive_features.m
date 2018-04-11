% Starter code prepared by James Hays for CS 143, Brown University
% This function should return all positive training examples (faces) from
% 36x36 images in 'train_path_pos'. Each face should be converted into a
% HoG template according to 'feature_params'. For improved performance, try
% mirroring or warping the positive training examples.

function features_pos = get_positive_features(train_path_pos, feature_params)
% 'train_path_pos' is a string. This directory contains 36x36 images of
%   faces
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.


% 'features_pos' is N by D matrix where N is the number of faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files);  % num_images equals to N, number of faces
D = (feature_params.template_size/feature_params.hog_cell_size)^2 * 31;
features_pos = zeros(num_images,D,'double'); % initialize a num_images*1116 matrix, element=0

for i = 1:num_images
    img_pro = image_files(i);
    img_path = fullfile(train_path_pos, img_pro.name);
    img = imread(img_path);
    channel = size(img, 3);
    if channel > 1
        img = rgb2gray(img);
    end
    hog = vl_hog(im2single(img),feature_params.hog_cell_size); % return a 6*6*31 matrix
    %imhog = vl_hog('render', hog,'verbose');
    
    hog = hog(:)'; % transform 6*6*31 matrix into 1*1116 matrix
    features_pos(i, :) = hog;
    %fprintf("%d",feature_pos);
end
    
    
% placeholder to be deleted
% features_pos = rand(100, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);