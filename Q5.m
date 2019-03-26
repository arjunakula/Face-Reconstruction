function [] = Q5()

myFolder = 'face_data/male_landmark_87/';
filePattern = fullfile(myFolder, '*.txt');
datFiles = dir(filePattern);

landmarkMatrix_male = zeros(numel(datFiles), (87*2));

for nn = 1:length(datFiles)
%for nn = 1:length(bmpFiles)
    baseFileName = datFiles(nn).name;
    fullFileName = fullfile(myFolder, baseFileName);
    %fprintf(1, 'reading file %s\n', fullFileName);
    
    noisyLandmarkData = dlmread(fullFileName);
    landmarkData = noisyLandmarkData(1:size(noisyLandmarkData,1),:);
    %imageDataSingleRow = imresize(imageData, [256 256]);
    landmarkMatrix_male(nn,:) = double(landmarkData(:).');
end

landmark_tr_set_male = landmarkMatrix_male(1:78,:);
landmark_tt_set_male = landmarkMatrix_male(79:88,:);

landmark_tr_set_male_mu = mean(landmark_tr_set_male,1);

myFolder = 'face_data/female_landmark_87/';
filePattern = fullfile(myFolder, '*.txt');
datFiles = dir(filePattern);

landmarkMatrix_female = zeros(numel(datFiles), (87*2));

for nn = 1:length(datFiles)
%for nn = 1:length(bmpFiles)
    baseFileName = datFiles(nn).name;
    fullFileName = fullfile(myFolder, baseFileName);
    %fprintf(1, 'reading file %s\n', fullFileName);
    
    noisyLandmarkData = dlmread(fullFileName);
    landmarkData = noisyLandmarkData(1:size(noisyLandmarkData,1),:);
    %imageDataSingleRow = imresize(imageData, [256 256]);
    landmarkMatrix_female(nn,:) = double(landmarkData(:).');
end

landmark_tr_set_female = landmarkMatrix_female(1:75,:);
landmark_tt_set_female = landmarkMatrix_female(76:85,:);

landmark_tr_set_female_mu = mean(landmark_tr_set_female,1);


myFolder = 'face_data/male_face/';
filePattern = fullfile(myFolder, '*.bmp');
bmpFiles = dir(filePattern);

imageMatrix_male = zeros(numel(bmpFiles), (256*256));
original_imageMatrix_male = zeros(numel(bmpFiles), (256*256));

for nn = 1:length(bmpFiles)
%for nn = 1:length(bmpFiles)
    baseFileName = bmpFiles(nn).name;
    fullFileName = fullfile(myFolder, baseFileName);
    %fprintf(1, 'reading file %s\n', fullFileName);
    imageData = imread(fullFileName);
    
    original_imageDataSingleRow = imresize(imageData, [256 256]);
    original_imageMatrix_male(nn,:) = double(original_imageDataSingleRow(:).');
    
    nn
    for j = 1:87
        original_landmark = [landmarkMatrix_male(nn,j), landmarkMatrix_male(nn,(j+87))];
        desired_landmark =  [landmark_tr_set_male_mu(j), landmark_tr_set_male_mu(j+87)];
        imageData = warpImage_kent(imageData, original_landmark, desired_landmark);
    end
    imageDataSingleRow = imresize(imageData, [256 256]);
    imageMatrix_male(nn,:) = double(imageDataSingleRow(:).');
end

warped_faces_tr_set_male = imageMatrix_male(1:78,:);
warped_faces_tt_set_male = imageMatrix_male(79:88,:);

faces_tr_set_male = original_imageMatrix_male(1:78,:);
faces_tt_set_male = original_imageMatrix_male(79:88,:);

myFolder = 'face_data/female_face/';
filePattern = fullfile(myFolder, '*.bmp');
bmpFiles = dir(filePattern);

imageMatrix_female = zeros(numel(bmpFiles), (256*256));
original_imageMatrix_female = zeros(numel(bmpFiles), (256*256));

for nn = 1:length(bmpFiles)
%for nn = 1:length(bmpFiles)
    baseFileName = bmpFiles(nn).name;
    fullFileName = fullfile(myFolder, baseFileName);
    %fprintf(1, 'reading file %s\n', fullFileName);
    imageData = imread(fullFileName);
    
    original_imageDataSingleRow = imresize(imageData, [256 256]);
    original_imageMatrix_female(nn,:) = double(original_imageDataSingleRow(:).');
    
    nn
    for j = 1:87
        original_landmark = [landmarkMatrix_female(nn,j), landmarkMatrix_female(nn,(j+87))];
        desired_landmark =  [landmark_tr_set_female_mu(j), landmark_tr_set_female_mu(j+87)];
        imageData = warpImage_kent(imageData, original_landmark, desired_landmark);
    end
    imageDataSingleRow = imresize(imageData, [256 256]);
    imageMatrix_female(nn,:) = double(imageDataSingleRow(:).');
end

warped_faces_tr_set_female = imageMatrix_female(1:75,:);
warped_faces_tt_set_female = imageMatrix_female(76:85,:);

faces_tr_set_female = original_imageMatrix_female(1:75,:);
faces_tt_set_female = original_imageMatrix_female(76:85,:);


% perform training for lda

% get x-matrix and y-matrix

tr_X = warped_faces_tr_set_male;
tr_X  = [tr_X; warped_faces_tr_set_female];

tr_Y = ones(1,size(warped_faces_tr_set_male,1));
tr_Y = [tr_Y, 2*(ones(1,size(warped_faces_tr_set_female,1)))];

tt_X = warped_faces_tt_set_male;
tt_X  = [tt_X; warped_faces_tt_set_female];

tt_Y = ones(1,size(warped_faces_tt_set_male,1));
tt_Y = [tt_Y, 2*(ones(1,size(warped_faces_tt_set_female,1)))];

% reduce dimensions from N to (N-c) using pca
tr_mu = mean(tr_X,1);
mean_subracted_tr_X = tr_X - repmat(tr_mu,size(tr_X,1),1);
mean_subracted_tt_X = tt_X - repmat(tr_mu,size(tt_X,1),1);

N = size(tr_X,1);
c = max(tr_Y);

top_k = N-c;
[coeff, score, latent] = pca(mean_subracted_tr_X);

new_tr_X = mean_subracted_tr_X * coeff(:,1:top_k);

num_components = 1;
Lda = lda(new_tr_X', tr_Y', num_components);

new_tt_X = mean_subracted_tt_X * coeff(:,1:top_k);
test_proj = Lda.W' * new_tt_X';
train_proj = Lda.W'*new_tr_X';

correct = 0;
for i = 1:20
     C = knn(train_proj, tr_Y', test_proj(i), 1);
      if (C == tr_Y(i))
          correct = correct + 1
      end
end

accuracy = (correct*1.0)/20

% fisher face visualization
imshow(reshape((coeff(:,1:top_k)*Lda.W)'+tr_mu,[256,256]),[])