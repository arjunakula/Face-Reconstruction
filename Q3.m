
function [] = Q3()
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
    
    nn
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

first_k_eigen_vectors = coeff(:,1:top_k);

for i=1:size(actual_tt_set,1)
    actual_tt_set(i,:) = actual_tt_set(i,:) - mean_landmark;
end

actual_tt_set_NEW = actual_tt_set * coeff(:,1:top_k);
actual_tt_set_NEW = actual_tt_set_NEW*coeff(:,1:top_k)';

 for i = 1:size(actual_tt_set_NEW,1)
    subplot(6,5,i);
    eig =  actual_tt_set_NEW(i,:)+mean_landmark;
    plot(eig(1:(size(eig,2)/2)) , 256-eig((size(eig,2)/2)+1:size(eig,2)));
    title(['Reconstructed Landmark Image ',num2str(i)]);
end


% step 2:

tt_set = imageMatrix(151:177,:);

for i=1:size(tt_set,1)
    tt_set(i,:) = tt_set(i,:) - tr_mu;
end

tt_set_NEW = tt_set * coeff_f(:,1:10);
tt_set_NEW = tt_set_NEW*coeff_f(:,1:10)';

for i = 1:size(tt_set_NEW,1)
    subplot(6,5,i);
    imshow(reshape(tt_set_NEW(i,:)+tr_mu,[256 256]),[]);
    title(['Reconstructed Test Image ',num2str(i)]);
end

% step 3

for i = 1:27
    i
    img = reshape(tt_set_NEW(i,:)+tr_mu,[256 256]);
    for j = 1:87
        original_landmark = [mean_landmark(j), mean_landmark(j+87)];
        desired_landmark =  [actual_tt_set_NEW(i,j)+mean_landmark(j), actual_tt_set_NEW(i,(j+87))+mean_landmark(j+87)];
        img = warpImage_kent(img, original_landmark, desired_landmark);
    end
    tt_set_NEW(i,:) = double(img(:).');
end


for i = 1:size(tt_set_NEW,1)
    subplot(6,5,i);
    imshow(reshape(tt_set_NEW(i,:),[256 256]),[]);
    title(['Reconstructed and Warped Test Image ',num2str(i)]);
end

x_axis = [];
y_axis = [];

o_tt_set = tt_set;
for i=1:size(o_tt_set,1)
    o_tt_set(i,:) = o_tt_set(i,:) + tr_mu;
end


ind = 1
for egv = 1:30:size(coeff_f,2)

    egv
    tt_set_NEW = tt_set * coeff_f(:,1:egv);
    tt_set_NEW = tt_set_NEW*coeff_f(:,1:egv)';
    
    for i = 1:27
    i
    img = reshape(tt_set_NEW(i,:)+tr_mu,[256 256]);
    for j = 1:87
        original_landmark = [mean_landmark(j), mean_landmark(j+87)];
        desired_landmark =  [actual_tt_set_NEW(i,j)+mean_landmark(j), actual_tt_set_NEW(i,(j+87))+mean_landmark(j+87)];
        img = warpImage_kent(img, original_landmark, desired_landmark);
    end
    tt_set_NEW(i,:) = double(img(:).');
    end

    %(not required for reconstruction) padding with zeros to match dimensions of original data and projected data
    %tt_set_NEW = [tt_set_NEW zeros(size(tt_set_NEW,1), ((256*256) - egv))];

    recons_diff =  (o_tt_set - tt_set_NEW);

    norm_recons_diff = (sum(recons_diff.^2, 2));

    avg_recons_error = sum(norm_recons_diff)/((size(recons_diff,2))*size(norm_recons_diff,1));
    
    %avg_recons_error = norm(recons_diff(:))/sqrt(size(recons_diff,1)*size(recons_diff,2));
    
    %/size(norm_recons_diff,1)
    %egv;
    %avg_recons_error;
    x_axis(ind) = egv;
    y_axis(ind) = avg_recons_error;
    ind = ind+1;
end

plot(x_axis,y_axis,'Color',1/255*[205 0 0],'LineWidth',2);
set(gca,'FontSize',14);
title('Reconstruction Error (MSE) vs # Eigen faces', 'FontSize', 17);
xlabel('#Eigen faces','FontSize',16);