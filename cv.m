% Computer Vision Funktionen

% 3D-Rekonstruktionsfunktion
function start3DReconstruction(~, ~)
    global h;
    set(h, 'String', 'Starting 3D Reconstruction...');

    pause(0.5); % kurz pausieren, damit die GUI aktualisiert wird
    
    % Hier würden Sie den Code zur Durchführung der 3D-Rekonstruktion hinzufügen.
    % Angenommen, 'create3DModel' ist Ihre Funktion zur 3D-Rekonstruktion und gibt Koordinaten X, Y und Z zurück.
    [X, Y, Z] = create3DModel(images, K); % 'K' ist die Kamera-Kalibrierungsmatrix
    
    % Da der eigentliche Code für die 3D-Rekonstruktion fehlt, erstellen wir eine 3D-Kugel als Platzhalter.
    display3DModel([X, Y, Z]);
    set(h, 'String', '3D Reconstruction Completed');
end