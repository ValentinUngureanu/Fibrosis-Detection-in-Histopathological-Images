    clc
    clear
    close all
    
    %% === SETĂRI ===
    trainImgDir = 'split_1/train/images';
    trainMaskDir = 'split_1/train/masks_rgb';
    valImgDir = 'split_1/val/images';
    valMaskDir = 'split_1/val/masks_rgb';
    
    imageSize = [256 256];
    classes = ["background", "fibrosis"];
    labelIDs = [0, 1];
    
    %% === FUNCȚII DE CITIRE ===
    readImageFcn = @(x) im2single(imresize(handleImageLab(imread(x)), imageSize));
    readMaskFcn  = @(x) uint8(imresize(imbinarizeGreenMask(imread(x)), imageSize));
    
    %% === DATASTORES ===
    trainImages = imageDatastore(trainImgDir, 'ReadFcn', readImageFcn);
    valImages   = imageDatastore(valImgDir,   'ReadFcn', readImageFcn);
    
    trainMasks = pixelLabelDatastore(trainMaskDir, classes, labelIDs, 'ReadFcn', readMaskFcn);
    valMasks   = pixelLabelDatastore(valMaskDir,   classes, labelIDs, 'ReadFcn', readMaskFcn);
    
    %% === AUGMENTARE ===
    augmenter = imageDataAugmenter( ...
        'RandRotation', [-20 20], ...
        'RandXReflection', true, ...
        'RandYReflection', true, ...
        'RandXScale', [0.9 1.1], ...
        'RandYScale', [0.9 1.1]);
    
    dsTrain = pixelLabelImageDatastore(trainImages, trainMasks, 'DataAugmentation', augmenter);
    dsVal   = pixelLabelImageDatastore(valImages, valMasks);
    
    %% === DEFINIRE UNET ===
    lgraph = unetLayers([imageSize 3], numel(classes), 'EncoderDepth', 4);
    
    %% === ANRENARE ===
    options = trainingOptions('adam', ...
        'InitialLearnRate', 1e-3, ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 8, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', dsVal, ...
        'ValidationFrequency', 100, ...
        'Verbose', true, ...
        'Plots', 'training-progress');
    
    [net, info] = trainNetwork(dsTrain, lgraph, options);
    
    %% === SALVARE ===
    modelDir = 'models';
    if ~exist(modelDir, 'dir')
        mkdir(modelDir);
    end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    modelName = ['unet_model_cielab_' timestamp '.mat'];
    save(fullfile(modelDir, modelName), 'net', 'info');
    
    %% === FUNCȚII ===
    function lab = handleImageLab(I)
        if size(I,3) == 4
            I = I(:,:,1:3);
        elseif size(I,3) == 1
            I = repmat(I, [1 1 3]);
        end
        lab = rgb2lab(I);
    end
    
    function BW = imbinarizeGreenMask(RGB)
        if size(RGB,3) ~= 3
            RGB = repmat(RGB, [1 1 3]);
        end
        BW = (RGB(:,:,1) < 50) & (RGB(:,:,2) > 200) & (RGB(:,:,3) < 50);
    end
