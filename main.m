% Globale Variablen zum Speichern der Bilder und des Fortschrittsbalkens
global images h;

% GUI Initialisierung
f = figure('Name', '3D Model Reconstruction', 'Color', 'white');
set(f, 'menubar', 'none');  % Menüleiste ausblenden
set(f, 'NumberTitle', 'off');  % Fenstertitel ausblenden
set(f, 'Position', [200, 200, 600, 400]);  % Fensterposition und Größe festlegen
set(gca, 'Position', [0.2 0.2 0.6 0.6]);  % Achsenposition und Größe festlegen
set(gca, 'FontName', 'Arial', 'FontSize', 14);

% Button zum Laden von Bildern
uicontrol('Style', 'pushbutton', 'String', 'Load Images', ...
    'Position', [20, 350, 100, 30], 'Callback', @loadImages);

% Button zum Starten der 3D-Rekonstruktion
uicontrol('Style', 'pushbutton', 'String', 'Start 3D Reconstruction', ...
    'Position', [20, 300, 160, 30], 'Callback', @start3DReconstruction);

% Fortschrittsbalken
h = uicontrol('Style', 'text', 'String', '', ...
    'Position', [20, 260, 160, 30]);

% Bildlade-Funktion
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
    set(h, 'String', 'Images Loaded');
end

% 3D-Rekonstruktionsfunktion
function start3DReconstruction(~, ~)
    global h;
    set(h, 'String', 'Starting 3D Reconstruction...');
    % Hier würden Sie den Code zur Durchführung der 3D-Rekonstruktion hinzufügen.
    % Angenommen, 'create3DModel' ist Ihre Funktion zur 3D-Rekonstruktion und gibt Koordinaten X, Y und Z zurück.
    [X, Y, Z] = create3DModel(images, K); % 'K' ist die Kamera-Kalibrierungsmatrix
    
    % Da der eigentliche Code für die 3D-Rekonstruktion fehlt, erstellen wir eine 3D-Kugel als Platzhalter.
    display3DModel([X, Y, Z]);
    set(h, 'String', '3D Reconstruction Completed');
end


function model = create3DModel(images, K)
    % Es wird angenommen, dass images ein Zellarray ist, in dem jedes Element ein Bild ist
    % Finden Sie Punkt-Korrespondenzen zwischen den Bildern
    Ftp1 = harris_detector(images{1});
    Ftp2 = harris_detector(images{2});
    correspondences = point_correspondence(images{1}, images{2}, Ftp1, Ftp2);
    
    % Anwenden des Eight-Point-Algorithmus
    EF = epa(correspondences, K);
    
    % Wiederherstellung der Kamera-Extrinsik und 3D-Punktwolke
    [T1, T2, R1, R2, U, V] = TR_from_E(EF);
    [T_cell, R_cell, d_cell, x1, x2] = reconstruction(T1, T2, R1, R2, correspondences, K);
    
    % 3D-Model erstellen
    model = cell2mat(d_cell);
end


function display3DModel(model)
    % Es wird angenommen, dass das Modell eine nx3-Matrix ist, wobei n die Anzahl der 3D-Punkte ist
    figure;
    scatter3(model(:,1), model(:,2), model(:,3), '.');
    title('3D Model');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
end
