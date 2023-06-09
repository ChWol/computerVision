%1.1
function gray_image = rgb_to_gray(input_image)
    % This function is supposed to convert a RGB-image to a grayscale image.
    % If the image is already a grayscale image directly return it.
    if length(size(input_image)) == 3
        rgb = double(input_image);
        gray_image = 0.299 * rgb(:, :, 1) + 0.587 * rgb(:, :, 2) + 0.114 * rgb(:, :, 3);
        gray_image = uint8(gray_image);
    else
        gray_image = input_image;
end

%1.2
function [Fx, Fy] = sobel_xy(input_image)
    % In this function you have to implement a Sobel filter 
    % that calculates the image gradient in x- and y- direction of a grayscale image.
    x = [1 0 -1; 2 0 -2; 1 0 -1];
    y = [1 2 1; 0 0 0; -1 -2 -1];
    
    Fx = conv2(input_image, x, 'same');
    Fy = conv2(input_image, y, 'same');
    
end

%1.3
function [segment_length, k, tau, do_plot] = harris_detector(input_image, varargin)
    % In this function you are going to implement a Harris detector that extracts features
    % from the input_image.

    %% Input parser
    parser = inputParser;
    addRequired(parser, 'input_image');
    addOptional(parser, 'segment_length', 15, @(x) isnumeric(x) && (x > 1) && rem(x,2) == 1);
    addOptional(parser, 'k', 0.05, @(x) isnumeric(x) && (x >= 0) && (x <= 1));
    addOptional(parser, 'tau', 1e6, @(x) isnumeric(x) && (x > 0));
    addOptional(parser, 'do_plot', false, @(x) islogical(x));
    parse(parser,input_image,varargin{:});
    
    segment_length = parser.Results.segment_length;
    k = parser.Results.k;
    tau = parser.Results.tau;
    do_plot = parser.Results.do_plot;
    
    features = {segment_length, k, tau, do_plot};
end

%1.4
function [Ix, Iy, w, G11, G22, G12] = harris_detector(input_image, varargin)
    % In this function you are going to implement a Harris detector that extracts features
    % from the input_image.
    
    %% Input parser from task 1.3
    % segment_length    size of the image segment
    % k                 weighting between corner- and edge-priority
    % tau               threshold value for detection of a corner
    % do_plot           image display variable
    input_parser
    
    %% Preparation for feature extraction
    % Check if it is a grayscale image
    if length(size(input_image)) ~= 2
        error("Image format has to be NxMx1")
    else
        gray = double(input_image);
    
    % Approximation of the image gradient
    [Ix, Iy] = sobel_xy(gray);
    
    % Weighting
    std_dev = segment_length/5;
    bound = segment_length/2 - 0.5;
    format = linspace(-bound, bound, segment_length);
    w = (1/sqrt(2*pi*(std_dev^2))) * exp(-(format.^2)/(2*std_dev));
    w = w/sum(w);

    % Harris Matrix G
    G11 = conv2((Ix.*Ix), w'*w, 'same');
    G22 = conv2((Iy.*Iy), w'*w, 'same');
    G12 = conv2((Ix.*Iy), w'*w, 'same');
end

%1.5
function [H, corners, features] = harris_detector(input_image, varargin)
    % In this function you are going to implement a Harris detector that extracts features
    % from the input_image.
    
    %% Input parser from task 1.3
    % segment_length    size of the image segment
    % k                 weighting between corner- and edge-priority
    % tau               threshold value for detection of a corner
    % do_plot           image display variable
    input_parser

    %% Preparation for feature extraction from task 1.4
    % Ix, Iy            image gradient in x- and y-direction
    % w                 weighting vector
    % G11, G12, G22     entries of the Harris matrix
    image_preprocessing
    
    %% Feature extraction with the Harris measurement
    dimension = size(G11);
    H = zeros(dimension);
    for i=(1:dimension(1))
        for j=(1:dimension(2))
            g = [G11(i,j) G12(i,j); G12(i,j) G22(i,j)];
            H(i,j) = det(g) - (k*(trace(g)^2));
        end
    end
    
    corners = H;
    threshold = ceil(segment_length/2);
    corners(1:threshold,:) = 0;
    corners(:,1:threshold) = 0;
    corners(dimension(1)-threshold:dimension(1),:) = 0;
    corners(:,dimension(2)-threshold:dimension(2)) = 0;
    corners(corners<tau) = 0;
    
    [row, col] = find(corners);
    features = [col row]';
end

%1.6
function features = harris_detector(input_image, varargin)
    % In this function you are going to implement a Harris detector that extracts features
    % from the input_image.
    
    %% Input parser from task 1.3
    % segment_length    size of the image segment
    % k                 weighting between corner- and edge-priority
    % tau               threshold value for detection of a corner
    % do_plot           image display variable
    input_parser

    %% Preparation for feature extraction from task 1.4
    % Ix, Iy            image gradient in x- and y-direction
    % w                 weighting vector
    % G11, G12, G22     entries of the Harris matrix
    image_preprocessing
    
    %% Feature extraction with the Harris measurement from task 1.5
    % features          detected features
    % corners           matrix containing the value of the Harris measurement for each pixel
    harris_measurement
    
    %% Plot
    if do_plot
        figure
        imshow(input_image)
        hold on
        plot(features(1,:), features(2,:), 'LineStyle','none','Marker','x','MarkerEdgeColor','y')
end

%1.7
function [min_dist, tile_size, N] = harris_detector(input_image, varargin)
    % In this function you are going to implement a Harris detector that extracts features
    % from the input_image.
    
    %% Input parser
    parser = inputParser;
    addRequired(parser, 'input_image');
    addOptional(parser, 'segment_length', 15, @(x) isnumeric(x) && (x > 1) && rem(x,2) == 1);
    addOptional(parser, 'k', 0.05, @(x) isnumeric(x) && (x >= 0) && (x <= 1));
    addOptional(parser, 'tau', 1e6, @(x) isnumeric(x) && (x > 0));
    addOptional(parser, 'do_plot', false, @(x) islogical(x));
    addOptional(parser, 'min_dist', 20, @(x) isnumeric(x) && (x >= 1));
    addOptional(parser, 'tile_size', 200, @(x) isnumeric(x));
    addOptional(parser, 'N', 5, @(x) isnumeric(x) && (x >= 1));
    parse(parser,input_image,varargin{:});
    
    segment_length = parser.Results.segment_length;
    k = parser.Results.k;
    tau = parser.Results.tau;
    do_plot = parser.Results.do_plot;
    min_dist = parser.Results.min_dist;
    tile_size = parser.Results.tile_size;
    N = parser.Results.N;
    
    if (length(tile_size) == 1) 
        tile_size = [tile_size, tile_size];
    
end

%1.8
function Cake = cake(min_dist)
    % The cake function creates a "cake matrix" that contains a circular set-up of zeros
    % and fills the rest of the matrix with ones. 
    % This function can be used to eliminate all potential features around a stronger feature
    % that don't meet the minimal distance to this respective feature.
    Cake = logical(zeros((2*min_dist) + 1));
    mid = min_dist + 1;
    for i=(1:length(Cake))
        for j=(1:length(Cake))
            diff_i = abs(i - mid);
            diff_j = abs(j - mid);
            if ((diff_i^2+diff_j^2) <= min_dist^2)
                Cake(i,j) = false;
            else
                Cake(i,j) = true;
            end
        end
    end
end

%1.9
function [corners, sorted_index] = harris_detector(input_image, varargin)
    % In this function you are going to implement a Harris detector that extracts features
    % from the input_image.
    
    %% Input parser from task 1.7
    % segment_length    size of the image segment
    % k                 weighting between corner- and edge-priority
    % tau               threshold value for detection of a corner
    % do_plot           image display variable
    % min_dist          minimal distance of two features in pixels
    % tile_size         size of the tiles
    % N                 maximal number of features per tile
    input_parser_new

    %% Preparation for feature extraction from task 1.4
    % Ix, Iy            image gradient in x- and y-direction
    % w                 weighting vector
    % G11, G12, G22     entries of the Harris matrix
    image_preprocessing
    
    %% Feature extraction with the Harris measurement from task 1.5
    % corners           matrix containing the value of the Harris measurement for each pixel         
    % features          detected features
    harris_measurement
    
    %% Feature preparation
    dimension = size(corners);
    corners = [zeros(dimension(1), min_dist), corners, zeros(dimension(1), min_dist)];
    dimension = size(corners);
    corners = [zeros(min_dist, dimension(2)); corners; zeros(min_dist, dimension(2))];
    
    corners_2 = reshape(corners, [], 1);
    [ranked, index] = sort(corners_2, 'descend');
    ranked = ranked(ranked > 0);
    sorted_index = index(1:length(ranked));
end

%1.10
function [acc_array, features] = harris_detector(input_image, varargin)
    % In this function you are going to implement a Harris detector that extracts features
    % from the input_image.
    
    %% Input parser from task 1.7
    % segment_length    size of the image segment
    % k                 weighting between corner- and edge-priority
    % tau               threshold value for detection of a corner
    % do_plot           image display variable
    % min_dist          minimal distance of two features in pixels
    % tile_size         size of the tiles
    % N                 maximal number of features per tile
    input_parser_new

    %% Preparation for feature extraction from task 1.4
    % Ix, Iy            image gradient in x- and y-direction
    % w                 weighting vector
    % G11, G12, G22     entries of the Harris matrix
    image_preprocessing
    
    %% Feature extraction with the Harris measurement from task 1.5
    % corners           matrix containing the value of the Harris measurement for each pixel         
    harris_measurement
    
    %% Feature preparation from task 1.9
    %corners            Harris measurement for each pixel respecting the minimal distance
    %sorted_index       Index list of features sorted descending by thier strength
    feature_preprocessing
    
    %% Accumulator array
    dimension = size(corners);
    acc_dim_y = floor(dimension(1)/tile_size(1));
    acc_dim_x = floor(dimension(2)/tile_size(2));
    accumulation = zeros(acc_dim_y, acc_dim_x);
    
    for i = (1:length(sorted_index))
        residual = mod(sorted_index(i), dimension(1));
        if residual == 0
            pos_y = dimension(1);
            pos_x = sorted_index(i)/dimension(1);
        else
            pos_x = floor(sorted_index(i)/dimension(1)) + 1;
            pos_y = residual;
        end
        
        bound_y_low = 0;
        bound_y_up = 0;
        bound_x_low = 0;
        bound_x_up = 0;
        
        for j = (1:acc_dim_y)
            for k=(1:acc_dim_x)
                bound_y_low = tile_size(1) * (j-1);
                bound_y_up = tile_size(1) * j;
                bound_x_low = tile_size(2) * (k-1);
                bound_x_up = tile_size(2) * k;
                if (pos_y > bound_y_low) && (pos_y <= bound_y_up)
                    if (pos_x > bound_x_low) && (pos_x <= bound_x_up)
                        accumulation(j,k) = accumulation(j,k) + 1;
                    end
                end
            end
        end
    end
    
    acc_temp = reshape(accumulation, [], 1);
    final_length = 0;
    for i = (1:length(acc_temp))
        tile_space = min(N, acc_temp(i));
        final_length = final_length + tile_space;
    end
        
    acc_array = zeros(acc_dim_y,acc_dim_x);
    features = zeros(2,final_length);
end

%1.11
function features = harris_detector(input_image, varargin)
    % In this function you are going to implement a Harris detector that extracts features
    % from the input_image.
    
    %% Input parser from task 1.7
    % segment_length    size of the image segment
    % k                 weighting between corner- and edge-priority
    % tau               threshold value for detection of a corner
    % do_plot           image display variable
    % min_dist          minimal distance of two features in pixels
    % tile_size         size of the tiles
    % N                 maximal number of features per tile
    input_parser_new

    %% Preparation for feature extraction from task 1.4
    % Ix, Iy            image gradient in x- and y-direction
    % w                 weighting vector
    % G11, G12, G22     entries of the Harris matrix
    image_preprocessing
    
    %% Feature extraction with the Harris measurement from task 1.5
    % corners           matrix containing the value of the Harris measurement for each pixel               
    harris_measurement
    
    %% Feature preparation from task 1.9
    % sorted_index      sorted indices of features in decreasing order of feature strength
    feature_preprocessing
    
    %% Accumulator array from task 1.10
    % acc_array         accumulator array which counts the features per tile
    % features          empty array for storing the final features
    accumulator_array
    
    %% Feature detection with minimal distance and maximal number of features per tile
    size_corners = size(corners);
    [acc_dim_y, acc_dim_x] = size(acc_array);
    addIndex = 0;
    dist_mask = cake(min_dist);
    for i = (1:length(sorted_index))
        
        [current_pos_y, current_pos_x] = ind2sub(size_corners, sorted_index(i));
        current_pos = [current_pos_y, current_pos_x];
        if (corners(current_pos(1), current_pos(2)) ~=0)
            
            corners((current_pos(1) - min_dist):(current_pos(1) + min_dist), (current_pos(2) - min_dist):(current_pos(2) + min_dist)) = corners((current_pos(1) - min_dist):(current_pos(1) + min_dist), (current_pos(2) - min_dist):(current_pos(2) + min_dist)) .* dist_mask;
            
            img_pos = current_pos - (ones(1,2) * min_dist);
            for j=(1:acc_dim_y)
                for k=(1:acc_dim_x)
                    bound_y_low = tile_size(1) * (j-1);
                    bound_y_up = tile_size(1) * j;
                    bound_x_low = tile_size(2) * (k-1);
                    bound_x_up = tile_size(2) * k;
                    if (img_pos(1) > bound_y_low) && (img_pos(1) <= bound_y_up)
                        if (img_pos(2) > bound_x_low) && (img_pos(2) <= bound_x_up)
                            if (acc_array(j,k) < N)
                                acc_array(j,k) = acc_array(j,k) + 1;
                                addIndex = addIndex + 1;
                                features(:, addIndex) = [img_pos(2); img_pos(1)];
                             end
                        end
                    end
                end
            end
        end
    end
    
    features = features(:, 1:addIndex);
    
    % Plot Routine
    plotting
end

%2.1
function [window_length, min_corr, do_plot, Im1, Im2] = point_correspondence(I1, I2, Ftp1, Ftp2, varargin)
    % In this function you are going to compare the extracted features of a stereo recording
    % with NCC to determine corresponding image points.
    
    %% Input parser
    parser = inputParser;
    addOptional(parser, 'window_length', 25, @(x) isnumeric(x) && (x > 1) && rem(x,2) == 1);
    addOptional(parser, 'min_corr', 0.95, @(x) isnumeric(x) && ((x > 0) && (x < 1)));
    addOptional(parser, 'do_plot', false, @(x) islogical(x));
    parse(parser,varargin{:});
    
    Im1 = double(I1);
    Im2 = double(I2);
    window_length = parser.Results.window_length;
    min_corr = parser.Results.min_corr;
    do_plot = parser.Results.do_plot;
end

%2.2
function [no_pts1, no_pts2, Ftp1, Ftp2] = point_correspondence(I1, I2, Ftp1, Ftp2, varargin)
    % In this function you are going to compare the extracted features of a stereo recording
    % with NCC to determine corresponding image points.
    
    %% Input parser from task 2.1
    % window_length     side length of quadratic window
    % min_corr          threshold for the correlation of two features
    % do_plot           image display variable
    % Im1, Im2          input images (double)
    input_parser
    
    %% Feature preparation
    buffer = zeros(size(Ftp1));
    adder = 0;
    for i = (1:length(Ftp1))
        current = Ftp1(:, i);
        if (included(size(Im1), window_length, [current(2), current(1)]))
            adder = adder + 1;
            buffer(:, adder) = current;
        end
    end
    Ftp1 = buffer(:, 1:adder);
    no_pts1 = adder;
    
    buffer = zeros(size(Ftp2));
    adder = 0;
    for i = (1:length(Ftp2))
        current = Ftp2(:,i);
        if (included(size(Im1), window_length, [current(2),current(1)]))
            adder = adder + 1;
            buffer(:, adder) = current;
        end
    end
    Ftp2 = buffer(:, 1:adder);
    no_pts2 = adder;
end

function included_return = included(img_dimension, window_length, current)
    included_return = false;
    max_distance = (window_length - 1)/2;
    if ((current(1) - max_distance) > 0) && ((current(1) + max_distance) <= img_dimension(1))
        if ((current(2) - max_distance) > 0) && ((current(2) + max_distance) <= img_dimension(2))
            included_return = true;
        end
    end
end

%2.3
function [Mat_feat_1, Mat_feat_2] = point_correspondence(I1, I2, Ftp1, Ftp2, varargin)
    % In this function you are going to compare the extracted features of a stereo recording
    % with NCC to determine corresponding image points.
    
    %% Input parser from task 2.1
    % window_length         side length of quadratic window
    % min_corr              threshold for the correlation of two features
    % do_plot               image display variable
    % Im1, Im2              input images (double)
    input_parser
    
    %% Feature preparation from task 2.2
    % no_pts1, no_pts 2     number of features remaining in each image
    % Ftp1, Ftp2            preprocessed features
    feature_preprocessing
    
    %% Normalization
    Mat_feat_1 = [];
    for i=(1:no_pts1)
        window = extract(Im1, Ftp1(:, i), window_length);
        window = reshape(window, [], 1);
        mue = mean(window);
        sigma = std(window);
        window = window - (ones(size(window)) * mue);
        window = window .* (1/sigma);
        Mat_feat_1 = [Mat_feat_1, window];
    end
    
    Mat_feat_2 = [];
    for i=(1:no_pts2)
        window = extract(Im2, Ftp2(:, i), window_length);
        window = reshape(window, [], 1);
        mue = mean(window);
        sigma = std(window);
        window = window - (ones(size(window)) * mue);
        window = window .* (1/sigma);
        Mat_feat_2 = [Mat_feat_2, window];
    end    
end

function window = extract(I, current, window_length)
    offset = (window_length - 1)/2;
    window = I((current(2) - offset):(current(2) + offset), (current(1) - offset):(current(1) + offset));
end

%2.4
function [NCC_matrix, sorted_index] = point_correspondence(I1, I2, Ftp1, Ftp2, varargin)
    % In this function you are going to compare the extracted features of a stereo recording
    % with NCC to determine corresponding image points.
    
    %% Input parser from task 2.1
    % window_length         side length of quadratic window
    % min_corr              threshold for the correlation of two features
    % do_plot               image display variable
    % Im1, Im2              input images (double)
    input_parser
    
    %% Feature preparation from task 2.2
    % no_pts1, no_pts 2     number of features remaining in each image
    % Ftp1, Ftp2            preprocessed features
    feature_preprocessing
    
    %% Normalization from task 2.3
    % Mat_feat_1            normalized windows in image 1
    % Mat_feat_2            normalized windows in image 2
    window_normalization
    
    %% NCC calculations
    NCC_matrix = zeros(no_pts2, no_pts1);
    n = window_length^2;
    
    for i = (1:no_pts1)
        for j = (1:no_pts2)
            NCC_matrix(j, i) = (1/(n - 1)) * (Mat_feat_2(:, j)' * Mat_feat_1(:, i));
        end
    end
    
    NCC_matrix(NCC_matrix < min_corr) = 0;
    NCC_matrix_column = reshape(NCC_matrix, [], 1);
    [correlation_ranked, index] = sort(NCC_matrix_column, 'descend');
    correlation_ranked = correlation_ranked(correlation_ranked > 0);
    sorted_index = index(1:length(correlation_ranked));   
end

%2.5
function cor = point_correspondence(I1, I2, Ftp1, Ftp2, varargin)
    % In this function you are going to compare the extracted features of a stereo recording
    % with NCC to determine corresponding image points.
    
    %% Input parser from task 2.1
    % window_length         side length of quadratic window
    % min_corr              threshold for the correlation of two features
    % do_plot               image display variable
    % Im1, Im2              input images (double)
    input_parser
    
    %% Feature preparation from task 2.2
    % no_pts1, no_pts 2     number of features remaining in each image
    % Ftp1, Ftp2            preprocessed features
    feature_preprocessing
    
    %% Normalization from task 2.3
    % Mat_feat_1            normalized windows in image 1
    % Mat_feat_2            normalized windows in image 2
    window_normalization
    
    %% NCC from task 2.4
    % NCC_matrix            matrix containing the correlation between the image points
    % sorted_index          sorted indices of NCC_matrix entries in decreasing order of intensity
    ncc_calculation
    
    %% Correspondeces
    cor = [];
    for i = (1:length(sorted_index))
        [mat_pos_y, mat_pos_x] = ind2sub(size(NCC_matrix), sorted_index(i));
        if (NCC_matrix(mat_pos_y, mat_pos_x) > 0)
            column1 = Ftp1(:,mat_pos_x);
            column2 = Ftp2(:,mat_pos_y);
            cor = [cor, [column1; column2]];
            NCC_matrix(:,mat_pos_x) = 0;
        end
    end 
end

%2.6
function cor = point_correspondence(I1, I2, Ftp1, Ftp2, varargin)
    % In this function you are going to compare the extracted features of a stereo recording
    % with NCC to determine corresponding image points.
    
    %% Input parser from task 2.1
    % window_length         side length of quadratic window
    % min_corr              threshold for the correlation of two features
    % do_plot               image display variable
    % Im1, Im2              input images (double)
    input_parser
    
    %% Feature preparation from task 2.2
    % no_pts1, no_pts 2     number of features remaining in each image
    % Ftp1, Ftp2            preprocessed features
    feature_preprocessing
    
    %% Normalization from task 2.3
    % Mat_feat_1            normalized windows in image 1
    % Mat_feat_2            normalized windows in image 2
    window_normalization
    
    %% NCC from task 2.4
    % NCC_matrix            matrix containing the correlation between the image points
    % sorted_index          sorted indices of NCC_matrix entries in decreasing order of intensity
    ncc_calculation
    
    %% Correspondeces from task 2.5
    % cor                   matrix containing all corresponding image points
    correspondence
    
    %% Visualize the correspoinding image point pairs
    if (do_plot == true)
        handle = imshow(I1);
        set(handle, 'AlphaData', ones(size(I1)) * 0.5);
        hold on
        handle2 = imshow(I2);
        set(handle2, 'AlphaData', ones(size(I1)) * 0.5);
        plot(cor(1,:), cor(2,:), 'LineStyle','none','Marker','x','MarkerEdgeColor','r');
        plot(cor(3,:), cor(4,:), 'LineStyle','none','Marker','x','MarkerEdgeColor','b');
        for i = (1:length(cor))
            plot([cor(1, i), cor(3, i)], [cor(2, i), cor(4, i)],'y');
        end
    end
end

%3.1
function [x1, x2, A, V] = epa(correspondences, K)
    % Depending on whether a calibrating matrix 'K' is given,
    % this function calculates either the essential or the fundamental matrix
    % with the eight-point algorithm.
    if (exist('K') == 1)
        x1 = correspondences(1:2,:);
        x2 = correspondences(3:4,:);
        x1 = [x1; ones(1, length(correspondences))];
        x2 = [x2; ones(1, length(correspondences))];
        x1 = inv(K) * x1;
        x2 = inv(K) * x2;
    else
        x1 = correspondences(1:2,:);
        x2 = correspondences(3:4,:);
        x1 = [x1; ones(1, length(correspondences))];
        x2 = [x2; ones(1, length(correspondences))];
    end
    
    A = [];
    for i = (1:length(x1))
        a = kron(x1(:, i), x2(:, i));
        A = [A; a'];
    end
    [U,S,V] = svd(A);
end

%3.2
function [EF] = epa(correspondences, K)
    % Depending on whether a calibrating matrix 'K' is given,
    % this function calculates either the essential or the fundamental matrix
    % with the eight-point algorithm.
    
    %% First step of the eight-point algorithm from task 3.1
    % Known variables:
    % x1, x2        homogeneous (calibrated) coordinates       
    % A             matrix A for the eight-point algorithm
    % V             right-singular vectors
    epa_part1
    
    %% Estimation of the matrices
    G_s = V(:, 9);
    G = reshape(G_s, [3, 3]);
    if (exist('K') == 1)
        [U, S, V] = svd(G);
        S = [1 0 0; 0 1 0; 0 0 0];
        EF = U * S * V';
    else
        [U, S, V] = svd(G);
        S(3, 3) = 0;
        EF = U * S * V';
    end
end

%3.3
function [epsilon, p, tolerance, x1_pixel, x2_pixel] = F_ransac(correspondences, varargin)
    % This function implements the RANSAC algorithm to determine 
    % robust corresponding image points
    parser = inputParser;
    addOptional(parser, 'epsilon', 0.5, @(x) isnumeric(x) && ((x > 0) && (x < 1)));
    addOptional(parser, 'p', 0.5, @(x) isnumeric(x) && ((x > 0) && (x < 1)));
    addOptional(parser, 'tolerance', 0.01, @(x) isnumeric(x));
    parse(parser, varargin{:});
    
    epsilon = parser.Results.epsilon;
    p = parser.Results.p;
    tolerance = parser.Results.tolerance;
    
    x1_pixel = correspondences(1:2,:);
    x1_pixel = [x1_pixel; ones(1, length(x1_pixel))];
    x2_pixel = correspondences(3:4,:);
    x2_pixel = [x2_pixel; ones(1, length(x2_pixel))];
end

%3.4
function [k, s, largest_set_size, largest_set_dist, largest_set_F] = F_ransac(correspondences, varargin)
    % This function implements the RANSAC algorithm to determine 
    % robust corresponding image points
       
    %% Input parser
    % Known variables:
    % epsilon       estimated probability
    % p             desired probability
    % tolerance     tolerance to belong to the consensus-set
    % x1_pixel      homogeneous pixel coordinates
    % x2_pixel      homogeneous pixel coordinates
    input_parser
    
    %% RANSAC algorithm preparation
    k = 8;
    s = log(1 - p)/log(1 - ((1 - epsilon)^k));
    largest_set_size = 0;
    largest_set_dist = inf;
    largest_set_F = zeros([3, 3]);
end

%3.5
function sd = sampson_dist(F, x1_pixel, x2_pixel)
    % This function calculates the Sampson distance based on the fundamental matrix F
    e3 = [0 -1 0; 1 0 0; 0 0 0];
    numerator = x2_pixel' * F * x1_pixel;
    numerator = (diag(numerator)').^2;
    left = vecnorm(e3 * F * x1_pixel);
    right = vecnorm((x2_pixel' * F * e3)');
    denominator = left.^2 + right.^2;
    sd = numerator./denominator;
end

%3.6
function [correspondences_robust, largest_set_F] = F_ransac(correspondences, varargin)
    % This function implements the RANSAC algorithm to determine 
    % robust corresponding image points
       
    %% Input parser
    % Known variables:
    % epsilon       estimated probability
    % p             desired probability
    % tolerance     tolerance to belong to the consensus-set
    % x1_pixel      homogeneous pixel coordinates
    % x2_pixel      homogeneous pixel coordinates
    input_parser
        
    %% RANSAC algorithm preparation
    % Pre-initialized variables:
    % k                     number of necessary points
    % s                     iteration number
    % largest_set_size      size of the so far biggest consensus-set
    % largest_set_dist      Sampson distance of the so far biggest consensus-set
    % largest_set_F         fundamental matrix of the so far biggest consensus-set
    ransac_preparation
    
    %% RANSAC algorithm
    for i = (1:s)
        random = randi([1 length(x1_pixel)], k, 1);
        current = [];
        
        for j = (1:k)
            current = [current, correspondences(:,random(j))];
        end
        F = epa(current);
        
        distance_array = sampson_dist(F, x1_pixel, x2_pixel);
        inlier = (distance_array < tolerance);
        num_inliers = sum(inlier);
        current_distance = sum(distance_array(inlier));
        
        if (num_inliers > largest_set_size)
            largest_set_size = num_inliers;
            largest_set_dist = current_distance;
            largest_set_F = F;
            largest_set_members = inlier;
            
        elseif (num_inliers == largest_set_size) && (largest_set_dist > current_distance)
            largest_set_size = num_inliers;
            largest_set_dist = current_distance;
            largest_set_F = F;
            largest_set_members = inlier;
        end
    end
    
    correspondences_robust = correspondences(:,largest_set_members);
end

%3.7
%% Load images
Image1 = imread('sceneL.png');
Image2 = imread('sceneR.png');
Gray1 = rgb_to_gray(Image1);
Gray2 = rgb_to_gray(Image2);

%% Calculate Harris features
features1 = harris_detector(Gray1, 'segment_length', 9, 'k', 0.05, 'min_dist', 30, 'N', 50, 'do_plot', false);
features2 = harris_detector(Gray2, 'segment_length', 9, 'k', 0.05, 'min_dist', 30, 'N', 50, 'do_plot', false);

%% Correspondence estimation
correspondences = point_correspondence(Gray1, Gray2 , features1, features2, 'window_length', 49, 'min_corr', 0.85, 'do_plot', false);

%% Determine robust corresponding image points with the RANSAC algorithm
correspondences_robust = F_ransac(correspondences, 'p', 0.5, 'tolerance', 0.1);

%% Visualize robust corresponding image points
figure
handle = imshow(Gray1);
set(handle, 'AlphaData', ones(size(Gray1)) * 0.5);
hold on
handle2 = imshow(Gray2);
set(handle2, 'AlphaData', ones(size(Gray2)) * 0.5);
plot(correspondences_robust(1,:), correspondences_robust(2,:), 'LineStyle', 'none', 'Marker', 'x', 'MarkerEdgeColor','r');
plot(correspondences_robust(3,:), correspondences_robust(4,:), 'LineStyle', 'none', 'Marker', 'x', 'MarkerEdgeColor','b');
for i = (1:length(correspondences_robust))
    plot([correspondences_robust(1,i), correspondences_robust(3,i)], [correspondences_robust(2,i), correspondences_robust(4,i)], 'y');
end
load('K.mat');
E = epa(correspondences_robust, K);
disp(E);

%4.1
function [T1, R1, T2, R2, U, V] = TR_from_E(E)
    % This function calculates the possible values for T and R 
    % from the essential matrix
    [U, S, V] = svd(E);
    Rp = [0 -1 0; 1 0 0; 0 0 1];
    Rn = [0 1 0; -1 0 0; 0 0 1];
    
    if (det(U) < 0) 
        U = U * [1 0 0; 0 1 0; 0 0 -1]; 
    end
    if (det(V') < 0) 
        V = V * [1 0 0; 0 1 0; 0 0 -1]; 
    end
    
    R1 = U * Rp' * V';
    R2 = U * Rn' * V';
    
    T1h = U * Rp * S * U';
    T1 = [T1h(3, 2); T1h(1, 3); T1h(2, 1)];
    
    T2h = U * Rn * S * U';
    T2 = [T2h(3, 2); T2h(1, 3); T2h(2, 1)];
end

%4.2
function [T_cell, R_cell, d_cell, x1, x2] = reconstruction(T1, T2, R1, R2, correspondences, K)
    %% Preparation
    n = size(correspondences, 2);
    d_cell = cell(1, 4);
    R_cell = [{R1}, {R1}, {R2}, {R2}];
    T_cell = [{T1}, {T2}, {T1}, {T2}];
    
    for i = (1:4)
        d_cell{i} = zeros(n, 2);
    end
    
    x1 = inv(K) * [correspondences(1:2,:); ones(1, n)];
    x2 = inv(K) * [correspondences(3:4,:); ones(1, n)];
end

%4.3
function [T, R, lambda, M1, M2] = reconstruction(T1, T2, R1, R2, correspondences, K)
    %% Preparation from task 4.2
    % T_cell    cell array with T1 and T2 
    % R_cell    cell array with R1 and R2
    % d_cell    cell array for the depth information
    % x1        homogeneous calibrated coordinates
    % x2        homogeneous calibrated coordinates
    preparation
    
    %% Reconstruction
    n = size(correspondences, 2);
    M1c = cell(1, 4);
    M2c = cell(1, 4);
    
    for i = (1:4)
        T = T_cell{i};
        R = R_cell{i};
        M1 = zeros(3 * n, n + 1);
        M2 = zeros(3 * n, n + 1);
        
        for j = (0:n-1)
            M1((3 * j) + 1:(3 * j) + 3, j + 1) = skew(x2(:,j + 1)) * R * x1(:,j + 1);
            M2((3 * j) + 1:(3 * j) + 3, j + 1) = skew(x1(:,j + 1)) * R' * x2(:,j + 1);
            M1((3 * j) + 1:(3 * j) + 3, n + 1) = skew(x2(:,j + 1)) * T;
            M2((3 * j) + 1:(3 * j) + 3, n + 1) = -skew(x1(:,j + 1)) * R' * T;
        end
        
        M1c{i} = M1;
        M2c{i} = M2;
        [U1, S1, V1] = svd(M1);
        d_cell{i}(:,1) = V1(1:n,end) / V1(end, end);
        
        [U2, S2, V2] = svd(M2);
        d_cell{i}(:,2) = V2(1:n,end) / V2(end, end);
    end
    
    pos = zeros(4, 1);
    
    for i = (1:4)
        pos(i) = sum(d_cell{i}(:,1) > 0) + sum(d_cell{i}(:,2) > 0);
    end
    
    [buffer, best] = max(pos);
    
    T = T_cell{best};
    R = R_cell{best};
    lambda = d_cell{best};
end

function x_hat = skew(x)
    x_hat = [0 -x(3) x(2); x(3) 0 -x(1); -x(2) x(1) 0];
end

%4.4
function [T, R, lambda, P1, camC1, camC2] = reconstruction(T1, T2, R1, R2, correspondences, K)
    % This function estimates the depth information and thereby determines the 
    % correct Euclidean movement R and T. Additionally it returns the
    % world coordinates of the image points regarding image 1 and their depth information.
    
    %% Preparation from task 4.2
    % T_cell    cell array with T1 and T2 
    % R_cell    cell array with R1 and R2
    % d_cell    cell array for the depth information
    % x1        homogeneous calibrated coordinates
    % x2        homogeneous calibrated coordinates
    preparation
    
    %% R, T and lambda from task 4.3
    % T         reconstructed translation
    % R         reconstructed rotation
    % lambda    depth information
    R_T_lambda
    
    %% Calculation and visualization of the 3D points and the cameras
    n = size(correspondences, 2);
    P1 = bsxfun(@times, lambda(:,1)', x1);
    figure('name', 'Reconstruction');
    hold on
    
    for i = (1:n)
        if (P1(3, i) > 0)
            scatter3(P1(1, i), P1(2, i), P1(3, i), '.k');
            text(P1(1, i), P1(2, i), P1(3, i), sprintf('%d', i), 'fontsize', 12, 'color', [0 0 0]);
        end
    end
    
    camC1 = [[-0.2 0.2 1]', [0.2 0.2 1]', [0.2 -0.2 1]', [-0.2 -0.2 1]'];
    camC1x = camC1(1,:)'; 
    camC1x = [camC1x; camC1x(1)];
    camC1y = camC1(2,:)'; 
    camC1y = [camC1y; camC1y(1)];
    camC1z = camC1(3,:)'; 
    camC1z = [camC1z; camC1z(1)];
    camC2 = inv(R) * (bsxfun(@plus, camC1, -T));
    camC2x = camC2(1,:)'; 
    camC2x = [camC2x; camC2x(1)];
    camC2y = camC2(2,:)'; 
    camC2y = [camC2y; camC2y(1)];
    camC2z = camC2(3,:)'; 
    camC2z = [camC2z; camC2z(1)];
    
    plot3(camC1x, camC1y, camC1z, 'g');
    text(camC1x(4), camC1y(4), camC1z(4), 'Cam1', 'color', 'g')
    
    plot3(camC2x, camC2y, camC2z, 'y');
    text(camC2x(4), camC2y(4), camC2z(4), 'Cam2', 'color', 'y')
    
    xlabel('X'); 
    ylabel('Y'); 
    zlabel('Z');
    
    grid on; 
    hold off;
    campos([43, -22, -87]);
    camup([0 -1 0]);
end

%4.5
function [repro_error, x2_repro] = backprojection(correspondences, P1, Image2, T, R, K)
    % This function calculates the mean error of the back projection
    % of the world coordinates P1 from image 1 in camera frame 2
    % and visualizes the correct feature coordinates as well as the back projected ones.
    n = size(P1, 2);
    P1C2 = zeros(3, n);
    x2 = [correspondences(3:4,:); ones(1, n)];
    P1hom = [P1; ones(1, n)];
    P1C2 = [R, T] * P1hom;
    P1C2 = P1C2 ./P1C2(3,:);
    x2_repro = K * P1C2;
    
    imshow(Image2)
    hold on
    
    plot(x2(1,:), x2(2,:), 'y', 'LineStyle', 'none', 'Marker', 'o')
    plot(x2_repro(1,:), x2_repro(2,:), 'g', 'LineStyle', 'none', 'Marker', 'x')
    
    for i = (1:n)
        text(x2(1,:), x2(2,:), sprintf('%d',i), 'fontsize', 8, 'color', 'y')
        text(x2_repro(1,:), x2_repro(2,:), sprintf('%d',i), 'fontsize', 8, 'color', 'g')
    end
    
    repro_error = sum(vecnorm(x2_repro - x2)) / n;
end