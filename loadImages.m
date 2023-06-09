% Funktion zum Laden der Bilder
function loadImages(~, ~)
    global h;

    [file, path] = uigetfile('*.jpg', 'MultiSelect', 'on');
    if isequal(file,0) || isequal(path,0)
        disp('Benutzer hat Auswahl abgebrochen')
        return
    end

    if iscell(file)
        for k = 1:length(file)
            images{k} = imread(fullfile(path, file{k}));
        end
    else
        images{1} = imread(fullfile(path, file));
    end

    % GUI Update zur Bestätigung
    set(h, 'String', 'Images Loaded');
end