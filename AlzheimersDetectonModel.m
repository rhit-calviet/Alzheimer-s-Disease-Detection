% Alzheimer’s Disease Detection Using VGG19 and MRI Scans
%
% This script implements a deep learning-based approach to classify MRI scans 
% for detecting Alzheimer’s Disease. The model leverages the VGG19 convolutional 
% neural network, fine-tuned on a dataset with four categories: 
% - MildDemented
% - ModerateDemented
% - NonDemented
% - VeryMildDemented
%
% Dataset: The augmented Alzheimer's MRI dataset is sourced from Kaggle:
% https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset/data.
%
% Key Features:
% 1. **Dataset Preprocessing**: 
%    - The script organizes data into training, validation, and test sets using 
%      MATLAB's `imageDatastore` and `augmentedImageDatastore` for memory-efficient 
%      processing.
%    - Images are resized to 224x224 to match the input size of VGG19, with grayscale 
%      images converted to RGB.
%
% 2. **Model Architecture**:
%    - The last three layers of the pre-trained VGG19 network are replaced with custom 
%      layers to classify the four categories.
%    - Fine-tuning is applied to optimize for this specific task.
%
% 3. **Training Configuration**:
%    - Training is performed using stochastic gradient descent with momentum (SGDM).
%    - Custom training options, including validation patience and verbose progress 
%      display, are configured.
%    - A mini-batch size of 8 ensures compatibility with limited system resources.
%
% 4. **Evaluation and Results**:
%    - The model's performance is evaluated on a separate test dataset.
%    - A confusion matrix is displayed to visualize classification accuracy.
%
% 5. **Model Saving**:
%    - The trained model, along with metadata, is saved in the `models` directory with a 
%      timestamped filename for reproducibility.
%
% 6. **Custom Utility**:
%    - Includes a utility function `displayTrainingProgress` to log training progress 
%      and metrics for improved tracking.

% Define paths for the dataset
baseDir = '..\DATA';
trainDir = fullfile(baseDir, 'TRAIN');
valDir = fullfile(baseDir, 'VAL');
testDir = fullfile(baseDir, 'TEST');

% Define categories (subfolders in each directory)
categories = {'MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'};

% Create datastores for TRAIN and VAL
trainDS = imageDatastore(trainDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
valDS = imageDatastore(valDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
testDS = imageDatastore(testDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Set input size for VGG19
inputSize = [224 224 3];

% Create augmentedImageDatastore for memory-efficient processing
trainADS = augmentedImageDatastore(inputSize, trainDS, ...
    'ColorPreprocessing', 'gray2rgb');
valADS = augmentedImageDatastore(inputSize, valDS, ...
    'ColorPreprocessing', 'gray2rgb');
testADS = augmentedImageDatastore(inputSize, testDS, ...
    'ColorPreprocessing', 'gray2rgb');

% Load VGG19 network and modify for fine-tuning
net = vgg19;

% Extract the layer graph
lgraph = layerGraph(net);

% Remove the last three layers
lgraph = removeLayers(lgraph, {'fc8', 'prob', 'output'});

% Add new layers
numClasses = numel(categories);
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc8', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'prob')
    classificationLayer('Name', 'output')
];

% Add the new layers
lgraph = addLayers(lgraph, newLayers);

% Connect the new layers
lgraph = connectLayers(lgraph, 'drop7', 'fc8');

% Calculate iterations per epoch
miniBatchSize = 8;  % Reduced batch size for memory efficiency
iterationsPerEpoch = ceil(numel(trainDS.Files) / miniBatchSize);

% Set custom training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', valADS, ...
    'ValidationFrequency', iterationsPerEpoch, ...
    'ValidationPatience', 5, ...
    'Verbose', true, ...
    'VerboseFrequency', 50, ...  % Show progress more frequently
    'ExecutionEnvironment', 'cpu', ... % Explicitly set to CPU
    'OutputFcn', @(info)displayTrainingProgress(info, iterationsPerEpoch), ...
    'Plots', 'training-progress');

fprintf('Starting training...\n');
fprintf('Training on %d images, validating on %d images\n', numel(trainDS.Files), numel(valDS.Files));
fprintf('Mini-batch size: %d, Iterations per epoch: %d\n\n', miniBatchSize, iterationsPerEpoch);

% Train the network
trainedNet = trainNetwork(trainADS, lgraph, options);

% Evaluate on the TEST dataset
fprintf('\nEvaluating on the test dataset...\n');
[YPredTest, scores] = classify(trainedNet, testADS);
YTest = testDS.Labels;

% Calculate test accuracy
testAccuracy = mean(YPredTest == YTest) * 100;
fprintf('Test Accuracy: %.2f%%\n', testAccuracy);

% Create and display confusion matrix
figure;
confusionchart(YTest, YPredTest);
title('Confusion Matrix');

% After training, save the network and relevant info
modelInfo = struct();
modelInfo.net = trainedNet;
modelInfo.inputSize = inputSize;
modelInfo.categories = categories;
modelInfo.dateCreated = datetime('now');
modelInfo.description = 'Alzheimer''s Disease Detection Model based on VGG19';

% Create models directory if it doesn't exist
if ~exist('models', 'dir')
    mkdir('models');
end

% Save the model with timestamp
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = fullfile('models', ['alzheimers_model_' timestamp '.mat']);
save(filename, 'modelInfo', '-v7.3');
fprintf('\nModel saved to: %s\n', filename);

% Custom display function for training progress
function stop = displayTrainingProgress(info, iterationsPerEpoch)
    stop = false;
    persistent lastEpoch;
    
    % Initialize lastEpoch if needed
    if isempty(lastEpoch)
        lastEpoch = 0;
    end
    
    % Calculate current epoch
    currentEpoch = ceil(info.Iteration / iterationsPerEpoch);
    
    % Display epoch start
    if currentEpoch > lastEpoch
        fprintf('\nEpoch %d/%d:\n', currentEpoch, 1);
        lastEpoch = currentEpoch;
    end
    
    % Display iteration progress
    if mod(info.Iteration, 10) == 0
        fprintf('.');
        if mod(info.Iteration, 200) == 0
            fprintf('\n');
        end
    end
    
    % Display end of epoch metrics
    if info.State == "iteration-end" && mod(info.Iteration, iterationsPerEpoch) == 0
        fprintf('\nTraining Loss: %f\n', info.TrainingLoss);
        
        if ~isempty(info.ValidationLoss)
            fprintf('Validation Loss: %f\n', info.ValidationLoss);
        end
        
        if ~isempty(info.TrainingAccuracy)
            fprintf('Training Accuracy: %.2f%%\n', info.TrainingAccuracy);
        end
        
        if ~isempty(info.ValidationAccuracy)
            fprintf('Validation Accuracy: %.2f%%\n', info.ValidationAccuracy);
        end
        
        fprintf('\n');
    end
end