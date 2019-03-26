function [] = Q4()
myFolder = 'face_data/landmark_87/';
filePattern = fullfile(myFolder, '*.dat');
datFiles = dir(filePattern);

landmarkMatrix = zeros(numel(datFiles), (87*2));

for nn = 1:length(datFiles)
%for nn = 1:length(bmpFiles)
    baseFileName = datFiles(nn).name;
    fullFileName = fullfile(myFolder, baseFileName);
    %fprintf(1, 'reading file %s\n', fullFileName);
    
    noisyLandmarkData = dlmread(fullFileName);
    landmarkData = noisyLandmarkData(2:size(noisyLandmarkData,1),:);
    %imageDataSingleRow = imresize(imageData, [256 256]);
    landmarkMatrix(nn,:) = double(landmarkData(:).');
end

tr_set = landmarkMatrix(1:150,:);
tt_set = landmarkMatrix(151:177,:);

mean_landmark = mean(tr_set,1);

size(tr_set);
size(tt_set);

myFolder = 'face_data/face/';
filePattern = fullfile(myFolder, '*.bmp');
bmpFiles = dir(filePattern);

imageMatrix = zeros(numel(bmpFiles), (256*256));
original_imageMatrix = zeros(numel(bmpFiles), (256*256));


for nn = 1:length(bmpFiles)
%for nn = 1:length(bmpFiles)
    baseFileName = bmpFiles(nn).name;
    fullFileName = fullfile(myFolder, baseFileName);
    %fprintf(1, 'reading file %s\n', fullFileName);
    imageData = imread(fullFileName);
    
    original_imageDataSingleRow = imresize(imageData, [256 256]);
    original_imageMatrix(nn,:) = double(original_imageDataSingleRow(:).');
    
    for j = 1:87
        original_landmark = [landmarkMatrix(nn,j), landmarkMatrix(nn,(j+87))];
        desired_landmark =  [mean_landmark(j), mean_landmark(j+87)];
        imageData = warpImage_kent(imageData, original_landmark, desired_landmark);
    end
    imageDataSingleRow = imresize(imageData, [256 256]);
    imageMatrix(nn,:) = double(imageDataSingleRow(:).');
    
end

tr_set = imageMatrix(1:150,:);
tt_set = imageMatrix(151:177,:);

size(tr_set);
size(tt_set);

tr_mu = mean(tr_set,1);
f_mean_subracted_tr_set = tr_set - repmat(tr_mu,size(tr_set,1),1);

top_k = 20;
[coeff_f, score_f, latent_f] = pca(f_mean_subracted_tr_set);

% step 1:
actual_tr_set = landmarkMatrix(1:150,:);
actual_tt_set = landmarkMatrix(1:27,:);

top_k = 10;

mean_subracted_tr_set = actual_tr_set - repmat(mean_landmark,size(actual_tr_set,1),1);
[coeff, score, latent] = pca(mean_subracted_tr_set);

first_k_landmark_eigen_vectors = coeff(:,1:top_k);
first_k_landmark_eigen_values = latent(1:10);

% for i=1:size(actual_tt_set,1)
%     actual_tt_set(i,:) = actual_tt_set(i,:) - mean_landmark;
% end
% 
% actual_tt_set_NEW = actual_tt_set * coeff(:,1:top_k);
% actual_tt_set_NEW = actual_tt_set_NEW*coeff(:,1:top_k)';
% 
%  for i = 1:size(actual_tt_set_NEW,1)
%     subplot(6,5,i);
%     eig =  actual_tt_set_NEW(i,:)+mean_landmark;
%     plot(eig(1:(size(eig,2)/2)) , 256-eig((size(eig,2)/2)+1:size(eig,2)));
%     title(['Reconstructed Landmark Image ',num2str(i)]);
% end


% step 2:

tr_set = imageMatrix(1:150,:);
tt_set = imageMatrix(151:177,:);

size(tr_set);
size(tt_set);

tr_mu = mean(tr_set,1);
f_mean_subracted_tr_set = tr_set - repmat(tr_mu,size(tr_set,1),1);

top_k = 10;
[coeff_f, score_f, latent_f] = pca(f_mean_subracted_tr_set);

first_k_appearance_eigen_vectors = coeff_f(:,1:top_k);
first_k_appearance_eigen_values = latent_f(1:10);

% original_face_tr_set = original_imageMatrix(1:150,:);
% original_face_tt_set = original_imageMatrix(151:177,:);
% 
% original_face_tt_set_NEW = original_face_tt_set * coeff_f(:,1:10);
% original_face_tt_set_NEW = original_face_tt_set_NEW*coeff_f(:,1:10)';


% generate new test image by random sampling of eigen values

for ind = 1:20

new_coeff_landmark = [];
new_coeff_appearance = [];

new_landmark_eigen_vectors = zeros(size(first_k_landmark_eigen_vectors(:,1),1), size(first_k_landmark_eigen_vectors(:,1),2));
new_appearance_eigen_vectors = zeros(size(first_k_appearance_eigen_vectors(:,1),1), size(first_k_appearance_eigen_vectors(:,1),2));

for i = 1:10
    new_coeff_landmark(i) = normrnd(0,sqrt(first_k_landmark_eigen_values(i)));
    new_coeff_appearance(i) = normrnd(0,sqrt(first_k_appearance_eigen_values(i)));
    new_landmark_eigen_vectors = new_landmark_eigen_vectors + new_coeff_landmark(i)*first_k_landmark_eigen_vectors(:,i);
    new_appearance_eigen_vectors = new_appearance_eigen_vectors + new_coeff_appearance(i)*first_k_appearance_eigen_vectors(:,i);
end
    
new_landmark_eigen_vectors = new_landmark_eigen_vectors' +  mean_landmark;
reshaped_image = reshape(new_appearance_eigen_vectors'+tr_mu,[256 256]);

for j = 1:87
        original_landmark = [mean_landmark(j), mean_landmark(j+87)];
        desired_landmark =  [new_landmark_eigen_vectors(j), new_landmark_eigen_vectors(j+87)];
        reshaped_image = warpImage_kent(reshaped_image, original_landmark, desired_landmark);
end

%warpedImage = warpImage_kent(reshaped_image, original_landmark, desired_landmark);
%imshow(reshaped_image,[]);
    
subplot(4,5,ind);
imshow(reshaped_image,[]);
title(['Synthesized Image ',num2str(ind)]);

end
