function [] = Q2()

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

size(tr_set);
size(tt_set);

tr_mu = mean(tr_set,1);

plot(tr_mu(1:(size(tr_mu,2)/2)) , 256-tr_mu((size(tr_mu,2)/2)+1:size(tr_mu,2)));

mean_subracted_tr_set = tr_set - repmat(tr_mu,size(tr_set,1),1);

top_k = 5;
[coeff, score, latent] = pca(mean_subracted_tr_set);

first_k_eigen_vectors = coeff(:,1:top_k);

for i = 1:top_k
    subplot(2,3,i);
    %imshow(reshape(first_k_eigen_vectors(:,i)',[256 256]),[]);
    eig =  first_k_eigen_vectors(:,i)'+tr_mu;
    plot(eig(1:(size(eig,2)/2)) , 256-eig((size(eig,2)/2)+1:size(eig,2)));
    title(['Eigen-Warping ',num2str(i)]);
end

%eig1 = coeff(:,1:1)'+tr_mu;
%plot(eig1(1:(size(eig1,2)/2)) , 256-eig1((size(eig1,2)/2)+1:size(eig1,2)));

tt_set_ORIGINAL = tt_set;

for i=1:size(tt_set,1)
    tt_set(i,:) = tt_set(i,:) - tr_mu;
end

 tt_set_5_eigen = tt_set * coeff(:,1:5);
 tt_set_5_eigen = tt_set_5_eigen*coeff(:,1:5)';
 
 for i = 1:size(tt_set_5_eigen,1)
    subplot(6,5,i);
    eig =  tt_set_5_eigen(i,:)+tr_mu;
    plot(eig(1:(size(eig,2)/2)) , 256-eig((size(eig,2)/2)+1:size(eig,2)));
    title(['Reconstructed Test Image ',num2str(i)]);
end

x_axis = zeros(1, size(coeff,2));
y_axis = zeros(1, size(coeff,2));

for egv = 1:size(coeff,2)

    tt_set_NEW = tt_set * coeff(:,1:egv);
    tt_set_NEW = tt_set_NEW*coeff(:,1:egv)';

    %(not required for reconstruction) padding with zeros to match dimensions of original data and projected data
    %tt_set_NEW = [tt_set_NEW zeros(size(tt_set_NEW,1), ((256*256) - egv))];

    recons_diff =  (tt_set - tt_set_NEW);

    norm_recons_diff = sqrt(sum(recons_diff.^2, 2));

    avg_recons_error = sum(norm_recons_diff)/size(norm_recons_diff,1);
    
    %avg_recons_error = norm(recons_diff(:))/sqrt(size(recons_diff,1)*size(recons_diff,2));
    
    %/size(norm_recons_diff,1)
    %egv;
    % avg_recons_error;
    x_axis(egv) = egv;
    y_axis(egv) = avg_recons_error;
    
end

plot(x_axis,y_axis,'Color',1/255*[205 0 0],'LineWidth',2);
set(gca,'FontSize',14);
title('Reconstruction Error (MSE) vs # Eigen Warpings', 'FontSize', 17);
xlabel('# Eigen Warpings','FontSize',16);