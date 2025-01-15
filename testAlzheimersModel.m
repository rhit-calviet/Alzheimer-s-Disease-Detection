% testAlzheimersModel - Test an Alzheimer’s Disease Detection Model
%
% This function allows users to test a pre-trained Alzheimer’s Disease detection 
% model on a single MRI image. The function performs the following steps:
%
% 1. **Model Selection**:
%    - Loads the latest trained model stored in the "models" directory.
%
% 2. **Image Selection**:
%    - Prompts the user to select an MRI image file (supported formats: .jpg, .png, 
%      .jpeg, .tiff, .bmp) for testing.
%
% 3. **Preprocessing**:
%    - Converts the image to grayscale (if necessary) and resizes it to match the 
%      model's input dimensions.
%    - Ensures the image has three channels by replicating the grayscale image.
%
% 4. **Prediction**:
%    - Uses the trained model to classify the input image into one of the categories: 
%      MildDemented, ModerateDemented, NonDemented, or VeryMildDemented.
%    - Outputs the predicted category and its confidence score.
%    - Displays a bar chart showing classification probabilities for all categories.
%
% 5. **Visualization**:
%    - Displays the input MRI image alongside the classification probabilities.
%
% Usage:
% Call the function `testAlzheimersModel` in the MATLAB command window. 
% The user will be prompted to select an MRI image for testing.
%
% Requirements:
% - The "models" directory must contain at least one saved model (e.g., 
%   `alzheimers_model_*.mat`).
% - The selected image must be a valid MRI scan in one of the supported formats.

function testAlzheimersModel()
clc
    % Get the model file
    modelDir = 'models';
    if ~exist(modelDir, 'dir')
        error('Models directory not found. Please ensure the model has been saved.');
    end
    
    % List all model files
    modelFiles = dir(fullfile(modelDir, 'alzheimers_model_*.mat'));
    if isempty(modelFiles)
        error('No model files found in the models directory.');
    end
    
    % Get the latest model by default
    [~, latestIdx] = max([modelFiles.datenum]);
    modelPath = fullfile(modelDir, modelFiles(latestIdx).name);
    
    % Load the model
    try
        fprintf('Loading model from: %s\n', modelPath);
        load(modelPath, 'modelInfo');
        net = modelInfo.net;
        inputSize = modelInfo.inputSize;
        categories = modelInfo.categories;
    catch ME
        error('Error loading model: %s', ME.message);
    end
    
    % Get the image file
    [filename, pathname] = uigetfile({'*.jpg;*.png;*.jpeg;*.tiff;*.bmp', 'Image Files (*.jpg,*.png,*.jpeg,*.tiff,*.bmp)'}, ...
        'Select an MRI image');
    
    if isequal(filename, 0)
        error('No image selected.');
    end
    
    imagePath = fullfile(pathname, filename);
    
    try
        % Read and preprocess the image
        img = imread(imagePath);
        
        % Convert to grayscale if RGB
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        % Resize image
        img = imresize(img, inputSize(1:2));
        
        % Convert to 3 channels
        img = repmat(img, [1 1 3]);
        
        % Create augmented datastore for single image
        testDS = augmentedImageDatastore(inputSize, img, ...
            'ColorPreprocessing', 'gray2rgb');
        
        % Make prediction
        [label, scores] = classify(net, testDS);
        
        % Display original image
        figure('Name', 'Alzheimer''s Disease Detection Result');
        
        subplot(1, 2, 1);
        imshow(img);
        title('Input MRI Image');
        
        % Create bar chart of predictions
        subplot(1, 2, 2);
        barh(scores);
        yticks(1:length(categories));
        yticklabels(categories);
        xlabel('Probability');
        title('Classification Probabilities');
        
        % Add text annotation for prediction
        [maxScore, maxIdx] = max(scores);
        fprintf('\nPrediction Results:\n');
        fprintf('Diagnosed Category: %s\n', char(label));
        fprintf('Confidence: %.2f%%\n\n', maxScore * 100);
        
        % Display all scores
        fprintf('Detailed Scores:\n');
        for i = 1:length(categories)
            fprintf('%s: %.2f%%\n', categories{i}, scores(i) * 100);
        end
        
    catch ME
        error('Error processing image: %s', ME.message);
    end
end