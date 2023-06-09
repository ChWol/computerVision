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
    % [X, Y, Z] = create3DModel(images, K); % 'K' ist die Kamera-Kalibrierungsmatrix
    
    % Da der eigentliche Code für die 3D-Rekonstruktion fehlt, erstellen wir eine 3D-Kugel als Platzhalter.
    %display3DModel(model);
    [X, Y, Z] = sphere;
    surf(X, Y, Z);
    set(h, 'String', '3D Reconstruction Completed');
end


function model = create3DModel(image, K)

    % Initialisierung der 3D-Punktwolke
    pointCloud = [];

    % Merkmalsextraktion aus dem Bild (Beispiel: SURF-Features)
    points = detectSURFFeatures(image);

    % Anpassung der Merkmale zwischen den Bildern
    % Da dies in dieser Aufgabe auf ein einzelnes Bild beschränkt ist, kann dieser Schritt nicht umgesetzt werden

    % Schätzen der Kameraposition und -orientierung
    % Dieser Schritt kann nur durchgeführt werden, wenn Sie mehrere Bilder mit ihren entsprechenden K-Matrizen haben

    % Triangulieren Sie 3D-Punkte (zum Beispiel mit der Funktion triangulate)
    % In diesem Szenario können wir keine Triangulation durchführen, da wir nur ein Bild und eine K-Matrix haben

    % Erzeugen Sie das Modell aus der Punktwolke
    model = pointCloud;

end

function plot = display3DModel(model)
% Nehmen wir an, dass "pointCloud" Ihre 3D-Punktwolke ist und aus einem nx3-Matrix besteht
% wobei n die Anzahl der Punkte ist und jede Zeile die (x, y, z)-Koordinaten eines Punktes darstellt.

x = pointCloud(:,1); % x-Koordinaten
y = pointCloud(:,2); % y-Koordinaten
z = pointCloud(:,3); % z-Koordinaten

% Erzeugen Sie das Scatter-Plot
scatter3(x, y, z, 'filled')
title('3D Rekonstruktion');
xlabel('X');
ylabel('Y');
zlabel('Z');

end
