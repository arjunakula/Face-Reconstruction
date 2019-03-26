function [] = Q1()

myFolder = 'face_data/face/';
filePattern = fullfile(myFolder, '*.bmp');
bmpFiles = dir(filePattern);

imageMatrix = zeros(numel(bmpFiles), (256*256));

for nn = 1:length(bmpFiles)
%for nn = 1:length(bmpFiles)
    baseFileName = bmpFiles(nn).name;
    fullFileName = fullfile(myFolder, baseFileName);
    %fprintf(1, 'reading file %s\n', fullFileName);
    imageData = imread(fullFileName);
    imageDataSingleRow = imresize(imageData, [256 256]);
    imageMatrix(nn,:) = double(imageDataSingleRow(:).');
end

tr_set = imageMatrix(1:150,:);
tt_set = imageMatrix(151:177,:);

size(tr_set);
size(tt_set);

tr_mu = mean(tr_set,1);

imshow(reshape(tr_mu,[256 256]),[]);

mean_subracted_tr_set = tr_set - repmat(tr_mu,size(tr_set,1),1);

top_k = 20;
[coeff, score, latent] = pca(mean_subracted_tr_set);

first_k_eigen_vectors = coeff(:,1:top_k);

for i = 1:top_k
    subplot(5,4,i);
    imshow(reshape(first_k_eigen_vectors(:,i)',[256 256]),[]);
    title(['Eigen-Face ',num2str(i)]);
end

tt_set_ORIGINAL = tt_set;

for i=1:size(tt_set,1)
    tt_set(i,:) = tt_set(i,:) - tr_mu;
end

tt_set_20_eigen = tt_set * coeff(:,1:20);
tt_set_20_eigen = tt_set_20_eigen*coeff(:,1:20)';

for i = 1:size(tt_set_20_eigen,1)
    subplot(6,5,i);
    imshow(reshape(tt_set_20_eigen(i,:)+tr_mu,[256 256]),[]);
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

    norm_recons_diff = (sum(recons_diff.^2, 2));

    avg_recons_error = sum(norm_recons_diff)/((size(recons_diff,2))*size(norm_recons_diff,1));
    
    %avg_recons_error = norm(recons_diff(:))/sqrt(size(recons_diff,1)*size(recons_diff,2));
    
    %/size(norm_recons_diff,1)
    %egv;
    %avg_recons_error;
    x_axis(egv) = egv;
    y_axis(egv) = avg_recons_error;
    
end

plot(x_axis,y_axis,'Color',1/255*[205 0 0],'LineWidth',2);
set(gca,'FontSize',14);
title('Reconstruction Error (MSE) vs # Eigen faces', 'FontSize', 17);
xlabel('#Eigen faces','FontSize',16);