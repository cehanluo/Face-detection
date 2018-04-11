data_path = '../data/';
image_path =  fullfile(data_path, 'lfw_faces/cut/'); 
image_files = dir( fullfile( image_path, '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files);  % num_images equals to N, number of faces
%frame_extracted = [];

for i=1:num_images
    file_name=image_files(i).name;
    im=imread(fullfile(image_path,file_name));
    im_m=im(50:210,55:200,:);
    im_m=imresize(im_m,[36,36]);
    imwrite(im_m, sprintf('../data/lfw_faces/cut_result/cut_%s.jpg', num2str(i)));
end
    