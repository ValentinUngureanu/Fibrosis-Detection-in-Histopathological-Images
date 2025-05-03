% === Numele imaginilor sănătoase (fără fibroză)
numeImagini = { ...
    'fara fibroza.tif', ...
    'infarct septic.tif', ...
    'pericardita fibrinoasa-necesitatea coloratiilor speciale.tif', ...
    'rejet transplant.tif', ...
    'amiloid- cum identifica algoritmul- necesitatea coloratiilor speciale.jpg' ...
};

% === Căile către imagini și unde salvăm măștile
imgFolder = 'split/train/images';
maskFolder = 'split/train/masks_rgb';

for i = 1:length(numeImagini)
    imgName = numeImagini{i};
    [~, name, ext] = fileparts(imgName);  % separăm nume și extensie
    ext = lower(ext);

    % Citește imaginea originală
    imgPath = fullfile(imgFolder, imgName);
    I = imread(imgPath);

    % Dacă e .jpg, salveaz-o ca .tif cu același nume
    if strcmp(ext, '.jpg')
        newName = [name '.tif'];
        imwrite(I, fullfile(imgFolder, newName));  % salvare .tif
        imgName = newName;  % actualizează pentru mască
        fprintf('📂 Conversie: %s → %s\n', imgName, newName);
    end

    % Creează mască neagră RGB (0,0,0)
    emptyMask = uint8(zeros(size(I,1), size(I,2), 3));

    % Salvează masca cu extensia .tif (acum uniform)
    imwrite(emptyMask, fullfile(maskFolder, imgName));
end

disp('✅ Conversia și generarea măștilor s-au încheiat.');
