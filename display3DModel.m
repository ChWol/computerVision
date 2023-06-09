% Funktion zum Darstellen des errechneten Modells
function display3DModel(model)
    % Es wird angenommen, dass das Modell eine nx4-Matrix ist, wobei n die Anzahl der 3D-Punkte ist und die vierte Spalte die Farben darstellt
    figure;
    scatter3(model(:,1), model(:,2), model(:,3), '.', 'CData', model(:,4:end));
    title('3D Model');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');

    % Aktivieren Sie das Drehen und Zoomen der Szene
    rotate3d on;
    zoom on;
end