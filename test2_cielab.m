clc
clear
close all

%% === Setări ===
imageSize = [256 256];
modelPath = 'models/unet_model_cielab9.mat';  
imgDir = 'split_1/val/images';
maskDir = 'split_1/val/masks_rgb';

load(modelPath, 'net');
imgFiles = dir(fullfile(imgDir, '*.tif'));

diceScores = []; iouScores = [];
allGT = []; allPred = [];

for i = 1:length(imgFiles)
    % === Citire ===
    name = imgFiles(i).name;
    I = imread(fullfile(imgDir, name));
    M = imread(fullfile(maskDir, name));
    if size(I,3) == 4, I = I(:,:,1:3); end
    I = rgb2lab(imresize(I, imageSize));
    I = im2single(I);

    GT = imbinarizeGreenMask(imresize(M, imageSize));

    P = semanticseg(I, net);
    P = P == "fibrosis";

    allGT = [allGT; GT(:)];
    allPred = [allPred; P(:)];

    inter = sum(P(:) & GT(:));
    diceScores(end+1) = 2*inter/(sum(P(:))+sum(GT(:))+eps);
    iouScores(end+1) = jaccard(P, GT);
end

%% === Metrici globale ===
TP = sum(allGT & allPred);
FP = sum(~allGT & allPred);
FN = sum(allGT & ~allPred);
TN = sum(~allGT & ~allPred);

precision = TP/(TP+FP+eps);
recall = TP/(TP+FN+eps);
f1 = 2*precision*recall/(precision+recall+eps);
acc = (TP+TN)/(TP+TN+FP+FN);

fprintf('\n📊 Evaluare:\n→ Dice: %.2f%%\n→ IoU: %.2f%%\n→ Precision: %.2f%%\n→ Recall: %.2f%%\n→ F1: %.2f%%\n→ Acc: %.2f%%\n',...
    mean(diceScores)*100, mean(iouScores)*100, precision*100, recall*100, f1*100, acc*100);

%% === ROC și PR ===
[fpRate, tpRate, ~, auc] = perfcurve(allGT, allPred, 1);
[prec, rec, ~] = perfcurve(allGT, allPred, 1, 'xCrit','reca','yCrit','prec');

figure; plot(fpRate, tpRate); title(sprintf('ROC (AUC = %.2f)', auc)); grid on;
figure; plot(rec, prec); title('PR Curve'); grid on;

figure; confusionchart(confusionmat(allGT, allPred), {'Sănătos','Fibroză'});
%% === FUNCȚIE – binarizare mască RGB ===
function BW = imbinarizeGreenMask(RGB)
    if size(RGB,3) ~= 3
        RGB = cat(3, RGB, RGB, RGB);
    end
    R = RGB(:,:,1);
    G = RGB(:,:,2);
    B = RGB(:,:,3);
    BW = (R < 50) & (G > 200) & (B < 50);  % zonă de fibroză = verde
end
