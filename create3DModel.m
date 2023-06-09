% Funktion zum Erstellen des 3D Modells
function model = create3DModel(images, K)
    % Es wird angenommen, dass images ein Zellarray ist, in dem jedes Element ein Bild ist
    % Finden Sie Punkt-Korrespondenzen zwischen den Bildern
    Ftp1 = harris_detector(images{1});
    Ftp2 = harris_detector(images{2});
    correspondences = point_correspondence(images{1}, images{2}, Ftp1, Ftp2);

    % Nehmen wir an, dass getColors eine Funktion ist, die RGB-Werte aus den Bildern extrahiert
    colors = getColors(images, correspondences);
    
    % Anwenden des Eight-Point-Algorithmus
    EF = epa(correspondences, K);
    
    % Wiederherstellung der Kamera-Extrinsik und 3D-Punktwolke
    [T1, T2, R1, R2, U, V] = TR_from_E(EF);
    [T_cell, R_cell, d_cell, x1, x2] = reconstruction(T1, T2, R1, R2, correspondences, K);
    
    % 3D-Model erstellen
    model = cell2mat(d_cell);
    % Hinzufügen der Farben zur Modellmatrix
    model = [model, colors];
end