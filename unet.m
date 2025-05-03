%% === SETĂRI ===
trainImgDir = 'split/train/images';
trainMaskDir = 'split/train/masks_rgb';
valImgDir = 'split/val/images';
valMaskDir = 'split/val/masks_rgb';

imageSize = [256 256];
classes = ["background", "fibrosis"];
labelIDs = [0, 1];

%% === READ IMAGE FUNC ===
readImageFcn = @(x) im2single(imresize(handleImage(imread(x)), imageSize));

%% === READ MASK FUNC ===
readMaskFcn = @(x) imbinarizeGreenMask(imresize(imread(x), imageSize));

%% === DATASTORES ===
trainImages = imageDatastore(trainImgDir, 'ReadFcn', readImageFcn);
valImages = imageDatastore(valImgDir, 'ReadFcn', readImageFcn);
trainMasks = pixelLabelDatastore(trainMaskDir, classes, labelIDs, 'ReadFcn', readMaskFcn);
valMasks = pixelLabelDatastore(valMaskDir, classes, labelIDs, 'ReadFcn', readMaskFcn);

%% === UNET ===
lgraph = unetLayers(imageSize, 2);  % 2 clase: background și fibrosis

%% === TRAINING OPTIONS ===
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', combine(valImages, valMasks), ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');  % fără checkpoint-uri

%% === TRAIN ===
[net, info] = trainNetwork(combine(trainImages, trainMasks), lgraph, options);

%% === SALVARE MODEL FINAL ÎN FOLDERUL "models" ===
modelDir = 'models';
if ~exist(modelDir, 'dir')
    mkdir(modelDir);
end
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
modelName = ['unet_model_' timestamp '.mat'];
modelPath = fullfile(modelDir, modelName);
save(modelPath, 'net', 'info');
disp(['✅ Modelul a fost salvat în: ' modelPath]);

%% === FUNCȚIE INTERNĂ – imagine ===
function I = handleImage(I)
    if size(I,3) == 4
        I = I(:,:,1:3);  % ignoră canalul alpha
    end
    if size(I,3) == 3
        I = rgb2gray(I);  % conversie grayscale
    end
end

%% === FUNCȚIE INTERNĂ – mască ===
function BW = imbinarizeGreenMask(RGB)
    if size(RGB,3) ~= 3
        RGB = cat(3, RGB, RGB, RGB);
    end
    R = RGB(:,:,1);
    G = RGB(:,:,2);
    B = RGB(:,:,3);
    BW = (R < 50) & (G > 200) & (B < 50);  % prag pentru verde
end
