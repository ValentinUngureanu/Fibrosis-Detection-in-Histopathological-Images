%% === Test pe o imagine ===

% === Setări
modelPath = 'models/unet_model_20250503_202301.mat';  % ← înlocuiește cu numele modelului tău
testImgPath = 'split/test/images/fara fibroza.tif';           % ← imagine test
imageSize = [256 256];

% === Încarcă modelul
load(modelPath, 'net');

% === Citește imaginea
I_original = imread(testImgPath);

% Conversie grayscale (doar pentru predicție)
if size(I_original, 3) == 4
    I_original = I_original(:,:,1:3);
end
if size(I_original, 3) == 3
    I_gray = rgb2gray(I_original);
else
    I_gray = I_original;
end
I_resized = im2single(imresize(I_gray, imageSize));

% === Predictie mască
predictedMask = semanticseg(I_resized, net);

% === Creează mască binară
maskBinary = predictedMask == "fibrosis";

% === Calculează procentajul zonei cu fibroză
fibrosisPixels = sum(maskBinary(:));
totalPixels = numel(maskBinary);
percentage = 100 * fibrosisPixels / totalPixels;

% === Suprapune mască peste imaginea originală redimensionată
I_display = imresize(I_original, imageSize);
overlayed = imoverlay(I_display, maskBinary, [1 1 0]);  % ← galben aprins

% === Afișare imagine + procentaj
figure;
imshow(overlayed);
title(sprintf('Zona de fibroză: %.2f%% din imagine', percentage));
