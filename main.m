function main

% Erzeugen einer GUI für die Bildauswahl
[filename, pathname] = uigetfile({'*.jpg;*.png'}, 'Bitte wählen Sie ein Bild aus');

if isequal(filename,0) || isequal(pathname,0)
   disp('Benutzer hat Auswahl abgebrochen')
else
   disp(['Benutzer hat ausgewählt ', fullfile(pathname, filename)])
   % Bild einlesen
   image = imread(fullfile(pathname, filename));
   
   % Transformation zum 3D-Modell
   % Hier müssen Sie Ihre eigenen Funktionen zur Erzeugung des 3D-Modells einfügen
   % Diese Funktionen können z.B. auf Algorithmen aus der Computer Vision Toolbox basieren
   % model = create3DModel(image);
   
   % Anzeigen des 3D-Modells
   % Dies könnte eine Funktion sein, die Sie schreiben, um das Modell in einem MATLAB-Figure-Fenster anzuzeigen
   % display3DModel(model);
end

end
