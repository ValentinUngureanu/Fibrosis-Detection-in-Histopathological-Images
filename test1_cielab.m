clc
clear
close all

%% === SETĂRI ===
modelPath = 'models/unet_model_cielab9.mat'; 
testImgPath = 'split_1/train/images/fara fibroza.tif';
imageSize = [256 256];

%% === ÎNCARCĂ MODELUL ===
load(modelPath, 'net');

%% === CITIRE IMAGINE ===
I_orig = imread(testImgPath);
if size(I_orig,3) == 4, I_orig = I_orig(:,:,1:3); end
if size(I_orig,3) == 1, I_orig = repmat(I_orig, [1 1 3]); end

I_cielab = rgb2lab(imresize(I_orig, imageSize));
I_input = im2single(I_cielab);

%% === PREDICȚIE ===
mask = semanticseg(I_input, net);
maskBin = mask == "fibrosis";

%% === CALCUL % ===
percentage = 100 * sum(maskBin(:)) / numel(maskBin);

%% === AFIȘARE ===
overlayed = imoverlay(imresize(I_orig, imageSize), maskBin, [1 1 0]);

figure;
if percentage < 3
    imshow(imresize(I_orig, imageSize)); title('Fără fibroză');
else
    imshow(overlayed); title(sprintf('Fibroză: %.2f%% din imagine', percentage));
end
