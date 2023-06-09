% Globale Variablen zum Speichern der Bilder und des Fortschrittsbalkens
global images h;

% GUI Initialisierung
f = figure('Name', '3D Model Reconstruction', 'Color', 'white');
set(f, 'menubar', 'none');
set(f, 'NumberTitle', 'off');
set(f, 'Position', [100, 100, 800, 600]);  % Fensterposition und Größe festlegen

% Titel-Header
uicontrol('Style', 'text', 'String', '3D Model Reconstruction', ...
    'Position', [100, 550, 600, 50], 'BackgroundColor', 'blue', ...
    'ForegroundColor', 'white', 'FontSize', 20, 'HorizontalAlignment', 'center');

% Beschreibung
uicontrol('Style', 'text', 'String', 'This tool allows you to reconstruct 3D models from 2D images.', ...
    'Position', [100, 500, 600, 30], 'FontSize', 14, 'HorizontalAlignment', 'center');

% Button zum Laden von Bildern
uicontrol('Style', 'pushbutton', 'String', 'Load Images', ...
    'Position', [100, 450, 200, 30], 'Callback', @loadImages);

% Button zum Starten der 3D-Rekonstruktion
uicontrol('Style', 'pushbutton', 'String', 'Start 3D Reconstruction', ...
    'Position', [500, 450, 200, 30], 'Callback', @start3DReconstruction);

% Fortschrittsbalken
h = uicontrol('Style', 'text', 'String', '', ...
    'Position', [100, 400, 600, 30], 'HorizontalAlignment', 'center');

% Button zum Exportieren des 3D-Modells
uicontrol('Style', 'pushbutton', 'String', 'Export 3D Model', ...
    'Position', [100, 350, 200, 30], 'Callback', @export3DModel);

% Button zum Anzeigen des Code-Repositories
uicontrol('Style', 'pushbutton', 'String', 'See Code', ...
    'Position', [500, 350, 200, 30], 'Callback', @openCodeRepository);

% 3D Modell Platzhalter
ax = axes('Position', [0.2, 0.05, 0.6, 0.3]);
axis(ax, 'off'); % Achsen ausschalten


% Utils Funktionen
% Export-Funktion
function export3DModel(~, ~)
    global model;  % Angenommen, das Modell ist eine globale Variable
    
    % Erstelle ein PointCloud-Objekt aus dem Modell
    ptCloud = pointCloud(model(:, 1:3), 'Color', model(:, 4:6));
    
    % Fragt den Benutzer nach einem Dateinamen und Speicherort
    [file, path] = uiputfile('*.ply', 'Save 3D Model As');
    if isequal(file, 0) || isequal(path, 0)
        disp('User cancelled export')
        return
    end
    outputFile = fullfile(path, file);
    
    % Speichert die Punktwolke im .ply-Format
    pcwrite(ptCloud, outputFile, 'Encoding', 'ascii');
    disp(['3D model saved as ', outputFile]);
end

% Öffnet das Code-Repository im Webbrowser
function openCodeRepository(~, ~)
    web('https://github.com/ChWol/computerVision', '-browser');  % Ersetzen Sie dies durch die tatsächliche URL Ihres Code-Repositories
end
