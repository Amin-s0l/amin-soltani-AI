clc
clear all

% input call
data = readtable('dataset_2.csv');

% define feature and labels
X = table2array(data(:, 4:8));
y_old = table2array(data(:, end));

% chang in labels
labels = {'No Failure'};
y = ismember(y_old, labels);
y = double(y);

% split data
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
X_train = X(cv.training, :);
y_train = y(cv.training, :);
X_test = X(cv.test, :);
y_test = y(cv.test, :);

% Standardization
X_train = zscore(X_train);
X_test = zscore(X_test);

% define neural net
net = feedforwardnet(500);
net.layers{1}.transferFcn = 'logsig';

% fit model
[net, tr] = train(net, X_train', y_train');

net.performFcn = 'mse';           % loss funcion  mse
net.trainFcn = 'trainlm';         % using Levenberg-Marquardt
net.trainParam.lr = 0.1;          %learning rate = 0.1
net.trainParam.max_fail = 1000;      %stop condition

% predict data and error calculate
y_pred = net(X_test')';
acc = sum(round(y_pred) == y_test) / length(y_test);
train_pred = net(X_train');
train_error = mse(y_train - train_pred);
train_rmse = sqrt(train_error);
test_pred = net(X_test');
test_error = mse(y_test - test_pred);
test_rmse = sqrt(test_error);
%showing outputs
disp(['Accuracy: ', num2str(acc)]);
disp(['RMSE for training data: ' num2str(train_rmse)]);
disp(['RMSE for test data: ' num2str(test_rmse)]);