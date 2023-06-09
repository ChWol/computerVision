% 3D Rekonstruktionsfunktion
function start3DReconstruction(~, ~)
    global h images;
    set(h, 'String', 'Starting 3D Reconstruction...');

    pause(1); % kurz pausieren, damit die GUI aktualisiert wird

    % STL-Datei lesen
    model = stlread('output.stl');

    % Vertices und Faces extrahieren
    V = model.Points;
    F = model.ConnectivityList;

    % 3D-Modell anzeigen
    ax = gca;
    cla(ax, 'reset');
    ax.Projection = 'perspective';
    hold(ax, 'on');
    
    % 3D-Modell anzeigen
    p = patch(ax, 'Faces', F, 'Vertices', V);
    p.FaceColor = [0.8 0.8 1.0];
    p.EdgeColor = 'none';
    
    % Beleuchtungsmodell festlegen
    camlight(ax, 'headlight');
    material(ax, 'dull');
    lighting(ax, 'gouraud');
    
    % Sichtbarkeit und Interaktion einstellen
    ax.Visible = 'off';
    rotate3d(ax, 'on');

    
    % Hier würden Sie den Code zur Durchführung der 3D-Rekonstruktion hinzufügen.
    % Angenommen, 'create3DModel' ist Ihre Funktion zur 3D-Rekonstruktion und gibt Koordinaten X, Y und Z zurück.
    % [X, Y, Z] = create3DModel(images, K); % 'K' ist die Kamera-Kalibrierungsmatrix
    
    % Da der eigentliche Code für die 3D-Rekonstruktion fehlt, erstellen wir eine 3D-Kugel als Platzhalter.
    %display3DModel([X, Y, Z]);
    %[X,Y,Z] = sphere;
    %surf(X,Y,Z);
    set(h, 'String', '3D Reconstruction Completed');
end