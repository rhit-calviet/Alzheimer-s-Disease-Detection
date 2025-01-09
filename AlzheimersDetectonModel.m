% Creating a model for Alzheimer’s Disease Detection from MRI scans
% Dataset: https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset/data
% Dataset has been modified on python 


% Reference: https://link.springer.com/article/10.1007/s12559-021-09946-2

% Define paths for the dataset
baseDir = 'DATA';
trainDir = fullfile(baseDir, 'TRAIN');
valDir = fullfile(baseDir, 'VAL');
testDir = fullfile(baseDir, 'TEST');

% Define categories (subfolders in each directory)
categories = {'MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'};

% Set image size (VGG19 input size is [224, 224, 3])
inputSize = [224 224 3];

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

% Apply transformations: resizing for VGG19 input size
trainDS = augmentedImageDatastore(inputSize, trainDS);
valDS = augmentedImageDatastore(inputSize, valDS);
testDS = augmentedImageDatastore(inputSize, testDS);

% Load VGG19 network and modify for fine-tuning
net = vgg19;

% Extract the layer graph
lgraph = layerGraph(net);

% Modify the last fully connected layer and classification layer
numClasses = numel(categories);
newFcLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
newClassLayer = classificationLayer('Name', 'new_output');

% Replace the old layers with new layers
lgraph = replaceLayer(lgraph, 'fc8', newFcLayer);
lgraph = replaceLayer(lgraph, 'prob', newClassLayer);

% Set custom training options
maxEpochs = 10;
initialLearnRate = 0.0001;
lrDecayFactor = 0.1;
patience = 3; % Stop if no improvement for this many epochs
minLearnRate = 1e-6;

% Training variables
bestValAccuracy = 0;
epochsWithoutImprovement = 0;
currentLearnRate = initialLearnRate;

fprintf('Training the network...\n');

% Start training loop
for epoch = 1:maxEpochs
    % Train on the training dataset
    fprintf('Epoch %d/%d\n', epoch, maxEpochs);
    
    % Update training options dynamically
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', currentLearnRate, ...
        'MaxEpochs', 1, ...
        'Verbose', false);
    
    % Train for one epoch
    trainedNet = trainNetwork(trainDS, lgraph, options);
    
    % Evaluate training accuracy
    [YPredTrain, ~] = classify(trainedNet, trainDS);
    YTrain = trainDS.Labels;
    trainAccuracy = mean(YPredTrain == YTrain) * 100;
    
    % Evaluate validation accuracy
    [YPredVal, ~] = classify(trainedNet, valDS);
    YVal = valDS.Labels;
    valAccuracy = mean(YPredVal == YVal) * 100;
    
    fprintf('Training Accuracy: %.2f%%\n', trainAccuracy);
    fprintf('Validation Accuracy: %.2f%%\n', valAccuracy);
    
    % Check for early stopping
    if valAccuracy > bestValAccuracy
        bestValAccuracy = valAccuracy;
        epochsWithoutImprovement = 0;
    else
        epochsWithoutImprovement = epochsWithoutImprovement + 1;
        if epochsWithoutImprovement >= patience
            fprintf('Stopping early due to no improvement in validation accuracy for %d epochs.\n', patience);
            break;
        end
    end
    
    % Adjust learning rate
    if epochsWithoutImprovement >= patience
        currentLearnRate = max(currentLearnRate * lrDecayFactor, minLearnRate);
        fprintf('Learning rate adjusted to %.6f\n', currentLearnRate);
    end
end

% Evaluate on the TEST dataset
fprintf('Evaluating on the test dataset...\n');
[YPredTest, ~] = classify(trainedNet, testDS);
YTest = testDS.Labels;

% Calculate test accuracy
testAccuracy = mean(YPredTest == YTest) * 100;
fprintf('Test Accuracy: %.2f%%\n', testAccuracy);

% Confusion matrix
figure;
confusionchart(YTest, YPredTest);
title('Confusion Matrix');